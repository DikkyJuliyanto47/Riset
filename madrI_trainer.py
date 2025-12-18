# madrI_trainer.py (perbaikan untuk stabilitas training)
import random
import math
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.utils as nn_utils

from madrI_config import (
    NUM_AGENTS,
    NUM_EPISODES,
    LR_ACTOR,
    LR_CRITIC,
    GAMMA,
    DEVICE,
    CONGESTION_ALPHA,
)
from madrI_models import ActorNetwork, CentralizedCritic
from madrI_utils import (
    get_osrm_distance_cached,
    is_coord_in_flood_zone,
    get_elevation_at_point,
    load_osrm_cache,
    save_osrm_cache,
)
from madrI_env import OneShotMultiAgentEnv

# Hyper-penyokong untuk stabilitas
ENTROPY_COEF = 0.01
ADV_CLIP = 20.0
MAX_GRAD_NORM = 0.5
MIN_PROB = 1e-8


def build_candidate_feature(user_lat, user_lon, evac):
    e_lat, e_lon = evac["coord"]
    dist = evac.get(
        "dist_km", get_osrm_distance_cached(user_lat, user_lon, e_lat, e_lon)
    )
    flood_flag = (
        1.0
        if evac.get("flooded", False) or is_coord_in_flood_zone(None, e_lat, e_lon)
        else 0.0
    )
    # normalize roughly: lat/lon /100, dist/10
    return [
        user_lat / 100.0,
        user_lon / 100.0,
        e_lat / 100.0,
        e_lon / 100.0,
        dist / 10.0,
        flood_flag,
    ]


def compute_agent_reward_tuple(u_lat, u_lon, evac, flood_gdf):
    e_lat, e_lon = evac["coord"]
    flooded = is_coord_in_flood_zone(flood_gdf, e_lat, e_lon)
    dist = evac.get("dist_km", get_osrm_distance_cached(u_lat, u_lon, e_lat, e_lon))
    elev = evac.get("elev", 0.0)
    # less extreme flood penalty for stability
    if flooded:
        return -30.0
    if dist <= 0.5:
        r = 100.0
    elif dist <= 1.0:
        r = 70.0
    elif dist <= 1.5:
        r = 30.0
    elif dist <= 2.0:
        r = 5.0
    else:
        r = -5.0 * dist
    # elevation bonus (capped)
    u_elev = evac.get("u_elev", 0.0)
    elev_diff = elev - u_elev
    if elev_diff > 0:
        r += min(elev_diff * 2.0, 30.0)
    elif elev_diff < -1.0:
        r -= 10.0
    return float(r)


# Ganti fungsi train_madrl_one_shot dengan versi ini (stabilisasi tambahan)
def train_madrl_one_shot(
    evac_candidates,
    flood_gdf,
    tif_path,
    user_pool,
    num_agents=NUM_AGENTS,
    num_episodes=NUM_EPISODES,
):
    load_osrm_cache()
    env = OneShotMultiAgentEnv(
        evac_candidates, flood_gdf, tif_path, user_pool, num_agents=num_agents
    )

    actor = ActorNetwork(input_dim=6).to(DEVICE)
    critic = CentralizedCritic(per_agent_feat_dim=6, num_agents=num_agents).to(DEVICE)

    # sedikit lebih kecil LR actor, scheduler untuk decay
    actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR * 0.5)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC * 0.8)
    actor_scheduler = optim.lr_scheduler.StepLR(actor_opt, step_size=50, gamma=0.8)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_opt, step_size=50, gamma=0.8)

    history = {"avg_reward": []}
    running_baseline = None  # moving average baseline for returns

    # hyperparams tambahan
    ENTROPY_COEF = 0.02
    ADV_CLIP = 10.0
    MAX_GRAD_NORM = 0.5
    MIN_PROB = 1e-8
    CRITIC_UPDATE_STEPS = 4  # update critic multiple times per episode
    CRITIC_UPDATES_FIRST_EPISODES = 10  # warm-up critic only for first N episodes
    SAVE_BEST_PATH_ACTOR = "madrI_actor_best.pt"
    SAVE_BEST_PATH_CRITIC = "madrI_critic_best.pt"
    best_recent_mean = -1e9
    RECENT_WINDOW = 20
    recent_rewards = []

    for ep in range(num_episodes):
        try:
            users, candidate_lists = env.build_episode_candidate_lists()
        except Exception as e:
            print(f"[EP {ep+1}] Gagal build candidates: {e}")
            continue

        agent_actions = []
        log_probs = []
        entropies = []
        chosen_feats = []
        rewards_raw = []

        # --- Actor sampling (per agent) ---
        for ai in range(num_agents):
            user_lat, user_lon = users[ai]
            cand_list = candidate_lists[ai]
            feats = [build_candidate_feature(user_lat, user_lon, c) for c in cand_list]

            if not feats:
                idx = random.randrange(len(evac_candidates))
                cand = evac_candidates[idx]
                feats = [build_candidate_feature(user_lat, user_lon, cand)]
                cand_list = [cand]

            feat_tensor = torch.tensor(feats, dtype=torch.float32).to(DEVICE)
            scores = actor(feat_tensor)
            if torch.isnan(scores).any():
                scores = torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

            probs = torch.softmax(scores, dim=0).clamp(MIN_PROB, 1.0)
            m = Categorical(probs)
            act_idx = int(m.sample().item())

            agent_actions.append(act_idx)
            lp = m.log_prob(torch.tensor(act_idx).to(DEVICE))
            if not isinstance(lp, torch.Tensor):
                lp = torch.tensor(float(lp), device=DEVICE)
            log_probs.append(lp)
            entropies.append(m.entropy())

            # chosen feat (include u_elev)
            chosen = cand_list[act_idx]
            user_elev = (
                get_elevation_at_point(tif_path, user_lat, user_lon)
                if tif_path
                else 0.0
            )
            chosen_for_critic = build_candidate_feature(
                user_lat,
                user_lon,
                {**chosen, "elev": chosen.get("elev", 0.0), "u_elev": user_elev},
            )
            chosen_feats.append(chosen_for_critic)

            r = compute_agent_reward_tuple(user_lat, user_lon, chosen, flood_gdf)
            rewards_raw.append(r)

        # --- env.step for counts & chosen infos ---
        try:
            step_rewards_tuples, chosen_infos, counts = env.step(
                users, candidate_lists, agent_actions
            )
        except Exception as e:
            print(f"[EP {ep+1}] env.step error: {e}")
            continue

        # --- apply congestion penalty ---
        final_rewards = []
        for ai in range(num_agents):
            base_r = rewards_raw[ai]
            chosen = chosen_infos[ai]["evac"]
            key = (round(chosen["coord"][0], 6), round(chosen["coord"][1], 6))
            cnt = counts.get(key, 1)
            cong_pen = -CONGESTION_ALPHA * math.log(1.0 + cnt)
            final = base_r + cong_pen
            final_rewards.append(final)

        # --- prepare critic input and values ---
        try:
            critic_in = torch.tensor(
                np.array(chosen_feats).reshape(-1), dtype=torch.float32
            ).to(DEVICE)
            values = critic(critic_in)  # (num_agents,)
        except Exception as e:
            print(f"[EP {ep+1}] critic forward error: {e}")
            continue

        # --- returns & baseline ---
        returns = torch.tensor(final_rewards, dtype=torch.float32).to(DEVICE)
        # update running baseline (moving average of episode mean)
        ep_mean = float(returns.mean().item())
        if running_baseline is None:
            running_baseline = ep_mean
        else:
            running_baseline = 0.98 * running_baseline + 0.02 * ep_mean

        # advantages = returns - values.detach()  (use running baseline for smoother signal)
        advantages = returns - values.detach()
        # optionally blend with running baseline to reduce variance:
        advantages = (
            advantages - (running_baseline - ep_mean) * 0.0
        )  # no-op placeholder (kept for clarity)

        # normalize & clip advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-ADV_CLIP, ADV_CLIP)

        # --- CRITIC UPDATES (multiple steps) ---
        critic_loss = None
        if ep < CRITIC_UPDATES_FIRST_EPISODES:
            # warm-up: only update critic for first episodes to let V be reasonable
            for _ in range(CRITIC_UPDATE_STEPS):
                critic_opt.zero_grad()
                vals = critic(critic_in)
                l = ((vals - returns) ** 2).mean()
                l.backward()
                nn_utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                critic_opt.step()
            critic_loss = float(l.detach().item())
            # skip actor update this episode
            actor_loss = torch.tensor(0.0, device=DEVICE)
        else:
            # normal flow: optionally update critic several times then actor
            for _ in range(CRITIC_UPDATE_STEPS):
                critic_opt.zero_grad()
                vals = critic(critic_in)
                l = ((vals - returns) ** 2).mean()
                l.backward()
                nn_utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                critic_opt.step()
            critic_loss = float(l.detach().item())

            # --- Actor loss with entropy ---
            lp = torch.stack(log_probs).to(DEVICE)
            entropy_term = torch.stack(entropies).to(DEVICE).mean()
            actor_loss = -(lp * advantages).sum() - ENTROPY_COEF * entropy_term

            # actor update
            actor_opt.zero_grad()
            actor_loss.backward()
            nn_utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            actor_opt.step()

        # step schedulers
        actor_scheduler.step()
        critic_scheduler.step()

        avg_r = float(returns.mean().item())
        history["avg_reward"].append(avg_r)
        recent_rewards.append(avg_r)
        if len(recent_rewards) > RECENT_WINDOW:
            recent_rewards.pop(0)
        recent_mean = float(np.mean(recent_rewards))

        # save best
        if recent_mean > best_recent_mean and ep >= 10:
            best_recent_mean = recent_mean
            torch.save(actor.state_dict(), SAVE_BEST_PATH_ACTOR)
            torch.save(critic.state_dict(), SAVE_BEST_PATH_CRITIC)

        if (ep + 1) % 10 == 0 or ep == 0:
            # actor_loss might be tensor or scalar
            al = (
                actor_loss.item()
                if isinstance(actor_loss, torch.Tensor)
                else float(actor_loss)
            )
            print(
                f"[EP {ep+1:03d}/{num_episodes}] AvgReward: {avg_r:.2f} | RecentMean({RECENT_WINDOW}): {recent_mean:.2f} | ActorLoss: {al:.2f} | CriticLoss: {critic_loss:.2f}"
            )

    save_osrm_cache()
    return actor, critic, history

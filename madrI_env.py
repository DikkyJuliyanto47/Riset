# madrI_env.py
import random
import numpy as np
import torch

from madrI_utils import (
    get_osrm_distance_cached,
    is_coord_in_flood_zone,
    get_elevation_at_point,
)
from madrI_config import CANDIDATE_RADIUS_KM


def build_agent_candidate_list(
    user_loc, evac_candidates, flood_gdf, tif_path, radius_km=CANDIDATE_RADIUS_KM
):
    """
    Return list of candidate dicts that are within radius_km of user_loc.
    Each candidate dict must have keys: 'coord':(lat,lon), 'name', optionally 'elev'
    """
    u_lat, u_lon = user_loc
    out = []
    for cand in evac_candidates:
        e_lat, e_lon = cand["coord"]
        dist = get_osrm_distance_cached(u_lat, u_lon, e_lat, e_lon)
        if dist <= radius_km:
            elev = cand.get("elev", None)
            if elev is None and tif_path:
                try:
                    elev = get_elevation_at_point(tif_path, e_lat, e_lon)
                except Exception:
                    elev = 0.0
            out.append(
                {**cand, "dist_km": dist, "elev": elev if elev is not None else 0.0}
            )
    return out


class OneShotMultiAgentEnv:
    """
    One-shot MADRL environment:
    - sample user pool (list of (lat,lon))
    - per episode sample NUM_AGENTS users (fixed)
    - each agent picks one candidate from its candidate list
    - environment returns per-agent reward and info (congestion handled globally)
    """

    def __init__(self, evac_candidates, flood_gdf, tif_path, user_pool, num_agents):
        self.evac_candidates = evac_candidates
        self.flood_gdf = flood_gdf
        self.tif_path = tif_path
        self.user_pool = user_pool[:]  # list of (lat, lon)
        self.num_agents = num_agents

    def sample_episode_agents(self):
        # fixed number selection; if pool smaller, sample with replacement
        if len(self.user_pool) >= self.num_agents:
            return random.sample(self.user_pool, self.num_agents)
        else:
            return [random.choice(self.user_pool) for _ in range(self.num_agents)]

    def build_episode_candidate_lists(self):
        # returns list of candidate lists (length = num_agents)
        users = self.sample_episode_agents()
        candidate_lists = []
        for u in users:
            cl = build_agent_candidate_list(
                u, self.evac_candidates, self.flood_gdf, self.tif_path
            )
            if not cl:
                # fallback: include all candidates but mark large dist
                cl = []
                for cand in self.evac_candidates:
                    e_lat, e_lon = cand["coord"]
                    dist = get_osrm_distance_cached(u[0], u[1], e_lat, e_lon)
                    cl.append({**cand, "dist_km": dist, "elev": cand.get("elev", 0.0)})
            candidate_lists.append(cl)
        return users, candidate_lists

    def step(self, users, candidate_lists, agent_actions):
        """
        agent_actions: list of indices chosen into each agent's candidate_list
        returns: rewards (list), chosen_info (list)
        """
        chosen_infos = []
        # Build counts for congestion
        key_counts = {}
        for ai, idx in enumerate(agent_actions):
            cand = candidate_lists[ai][idx]
            key = (round(cand["coord"][0], 6), round(cand["coord"][1], 6))
            key_counts[key] = key_counts.get(key, 0) + 1

        # Compute per-agent reward
        from madrI_utils import is_coord_in_flood_zone

        rewards = []
        for ai, idx in enumerate(agent_actions):
            u_lat, u_lon = users[ai]
            cand = candidate_lists[ai][idx]
            e_lat, e_lon = cand["coord"]
            flooded = is_coord_in_flood_zone(self.flood_gdf, e_lat, e_lon)
            dist = cand["dist_km"]
            elev = cand.get("elev", 0.0)
            # compose chosen info
            chosen_infos.append(
                {"evac": cand, "dist_km": dist, "elev": elev, "flooded": flooded}
            )
            rewards.append((u_lat, u_lon, e_lat, e_lon, dist, elev, flooded))
        return rewards, chosen_infos, key_counts

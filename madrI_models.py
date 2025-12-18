# madrI_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    Shared actor: menerima per-candidate features (for one agent)
    dan mengembalikan satu skor per kandidat (logit).
    Input dim per candidate: 6 (S1).
    """

    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)  # per-candidate -> scalar score

    def forward(self, x):
        # x: (num_candidates, input_dim)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.out(h).squeeze(-1)  # (num_candidates,)


class CentralizedCritic(nn.Module):
    """
    Critic yang menerima fitur dari semua agen (chosen candidate features)
    dan mengeluarkan value estimate untuk setiap agen.
    Implementation: flatten per-agent chosen features -> MLP -> output NUM_AGENTS scalars.
    """

    def __init__(self, per_agent_feat_dim=6, num_agents=32, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        total_dim = per_agent_feat_dim * num_agents
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_agents)

    def forward(self, x):
        # x: (batch=1, total_dim) or (total_dim,) flatten
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        vals = self.out(h)  # (1, num_agents)
        return vals.squeeze(0)  # (num_agents,)

import torch
import torch.nn as nn


# =========================================================
# Attention Module (WITH MASKING - CRITICAL)
# =========================================================

class AttentionModule(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()

        # ---- embeddings (improved) ----
        self.ego_embed = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        self.neigh_embed = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

    def forward(self, ego, neighbors):
        """
        ego: (B, 6)
        neighbors: (B, N, 7)
        """

        # ---- create mask BEFORE embedding ----
        # True = ignore
        key_padding_mask = (neighbors.abs().sum(dim=-1) == 0)  # (B, N)

        # ---- embeddings ----
        ego_e = self.ego_embed(ego).unsqueeze(1)   # (B,1,d)
        neigh_e = self.neigh_embed(neighbors)      # (B,N,d)

        tokens = torch.cat([ego_e, neigh_e], dim=1)  # (B, N+1, d)

        # ---- extend mask to include ego token ----
        ego_mask = torch.zeros((ego.size(0), 1), dtype=torch.bool, device=ego.device)
        key_padding_mask = torch.cat([ego_mask, key_padding_mask], dim=1)

        # ---- attention ----
        Q = ego_e
        K = tokens
        V = tokens

        out, attn_weights = self.attn(
            Q, K, V,
            key_padding_mask=key_padding_mask
        )

        return out.squeeze(1), attn_weights


# =========================================================
# Policy Head (FOR STANDALONE USE ONLY)
# =========================================================

class PolicyHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# OPTIONAL: Standalone Policy (NOT used in RLlib)
# =========================================================

class AttentionPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = AttentionModule(d_model=64)
        self.policy_head = PolicyHead(64)

    def forward(self, ego, neighbors):

        context, _ = self.attention(ego, neighbors)

        action = self.policy_head(context)

        # scale to env bounds [-2, 2]
        action = torch.tanh(action) * 2.0

        return action
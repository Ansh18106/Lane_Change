import torch
import torch.nn as nn

class PolicyHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # [a_lin, a_ang]
        )

    def forward(self, x):
        return self.net(x)

class AttentionModule(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()

        self.ego_embed = nn.Linear(6, d_model)
        self.neigh_embed = nn.Linear(7, d_model)

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

        ego_e = self.ego_embed(ego).unsqueeze(1)   # (B,1,d)
        neigh_e = self.neigh_embed(neighbors)      # (B,N,d)

        tokens = torch.cat([ego_e, neigh_e], dim=1)  # (B, N+1, d)

        Q = ego_e                      # ego queries
        K = tokens
        V = tokens

        out, attn_weights = self.attn(Q, K, V)

        return out.squeeze(1), attn_weights
    

class AttentionPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = AttentionModule(d_model=64)
        self.policy_head = PolicyHead(64)

    def forward(self, ego, neighbors):

        context, _ = self.attention(ego, neighbors)

        action = self.policy_head(context)

        action = torch.tanh(action) * 2.0

        return action
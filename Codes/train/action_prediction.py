import torch
import torch.nn as nn

class SharedAgentPolicy(nn.Module):
    def __init__(self, d, num_lanes, lane_embed_dim=4):
        super().__init__()

        self.lane_embedding = nn.Embedding(num_lanes, lane_embed_dim)

        self.net = nn.Sequential(
            nn.Linear(d + lane_embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x_continuous, lane_idx):
        # x_continuous: (B, N, d)
        # lane_idx: (B, N)

        lane_emb = self.lane_embedding(lane_idx)  # (B, N, embed_dim)
        x = torch.cat([x_continuous, lane_emb], dim=-1)

        B, N, D = x.shape
        x = x.reshape(B * N, D)
        z = self.net(x)
        z = z.reshape(B, N, 2)

        return z

def map_to_action(z, a_min, a_max, alpha_min, alpha_max):
    # all tensors shape: (B, N)

    a = a_min + z[..., 0] * (a_max - a_min)
    alpha = alpha_min + z[..., 1] * (alpha_max - alpha_min)

    return torch.stack([a, alpha], dim=-1)
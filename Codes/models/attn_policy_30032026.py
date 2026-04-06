from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn

from Codes.models.local_attention_30032026 import LocalAttention

class AttentionPolicy(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.ego_embed = nn.Linear(7, 64)
        self.neigh_embed = nn.Linear(3, 64)

        self.attn = LocalAttention(64)

        self.policy = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

        self.value = nn.Linear(128, 1)

    def forward(self, input_dict, state, seq_lens):

        ego = input_dict["obs"]["ego"].float()              # (B,7)
        neigh = input_dict["obs"]["neighbors"].float()      # (B,N,3)

        ego_emb = self.ego_embed(ego)
        neigh_emb = self.neigh_embed(neigh)

        attn_out, _ = self.attn(ego_emb, neigh_emb)

        fused = torch.cat([ego_emb, attn_out], dim=-1)

        logits = self.policy(fused)
        self._value = self.value(fused).squeeze(-1)

        return logits, state

    def value_function(self):
        return self._value
from time import time
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from env.multiagent_env_rllib_30032026 import RenderCallback, env_creator


class NoAttentionPolicyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.d_model = 64
        self.num_agents = 6

        # Input: ego (6) + neighbors (5 × 7)
        self.input_dim = 6 + 5 * 7

        # Encoder (replaces attention)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.d_model),
            nn.ReLU()
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(64, 2)
        self.log_std = nn.Parameter(torch.zeros(self.num_agents * 2))

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"]

        ego = torch.tensor(obs["ego"], dtype=torch.float32)
        neighbors = torch.tensor(obs["neighbors"], dtype=torch.float32)

        # Shape handling (same as your attention model)
        if ego.dim() == 2:
            batch_size = 1
            num_agents = ego.size(0)

            ego_flat = ego
            neigh_flat = neighbors.reshape(num_agents, -1)

        elif ego.dim() == 3:
            batch_size, num_agents, _ = ego.shape

            ego_flat = ego.reshape(batch_size * num_agents, -1)
            neigh_flat = neighbors.reshape(batch_size * num_agents, -1)

        else:
            raise ValueError(f"Unexpected ego shape {ego.shape}")

        # Concatenate ego + neighbors
        inp = torch.cat([ego_flat, neigh_flat], dim=-1)

        # Encode
        context = self.encoder(inp)

        # Policy
        x = self.policy_head(context)
        mean_per_agent = self.mean_layer(x)

        # Value
        value_per_agent = self.value_head(context).squeeze(-1)

        if batch_size == 1:
            self._value_out = value_per_agent.mean().unsqueeze(0)
        else:
            self._value_out = value_per_agent.reshape(batch_size, num_agents).mean(dim=1)

        # Flatten actions
        mean = mean_per_agent.reshape(batch_size, num_agents * 2)
        log_std = self.log_std.unsqueeze(0).expand(batch_size, -1)

        return torch.cat([mean, log_std], dim=1), state

    def value_function(self):
        return self._value_out
    


# =========================================================
# Register everything
# =========================================================

ModelCatalog.register_custom_model("no_attn_model", NoAttentionPolicyModel)
register_env("lane_change_env", env_creator)

ray.init()

# =========================================================
# PPO Config
# =========================================================

config = (
    PPOConfig()
    .environment(
        env="lane_change_env",
        env_config={"num_agents": 6}
    )
    .framework("torch")
    .rollouts(num_rollout_workers=0)
    .training(
        model={
            "custom_model": "no_attn_model",
        },
        train_batch_size=4000,
        gamma=0.99,
        lr=3e-4,
        entropy_coeff=0.01,
    )
    # .callbacks(RenderCallback)
)

algo = config.build()

# =========================================================
# Training Loop
# =========================================================

reward_history = []
total_start_time = time.time()
for i in range(500):
    iter_start_time = time.time()
    result = algo.train()
    reward_mean = result["episode_reward_mean"]
    reward_history.append(reward_mean)

    iter_end_time = time.time()
    iter_duration = iter_end_time - iter_start_time
    total_elapsed = iter_end_time - total_start_time

    # Added time metrics to the print statement
    print(f"No Attention {i} | reward = {reward_mean:.2f} | iter_time = {iter_duration:.2f}s | total_time = {total_elapsed / 60:.2f}m")

    if i % 20 == 0:
        checkpoint = algo.save()
        # print("Checkpoint saved at", checkpoint)

# pprint("Evaluation Results:", config.evaluate())

# =========================================================
# Plot the reward curve
# =========================================================

plt.figure(figsize=(10, 5))
plt.plot(reward_history, marker='o', linestyle='-')
plt.title('PPO Training: Episode Reward Mean per Iteration')
plt.xlabel('Training Iteration')
plt.ylabel('Episode Reward Mean')
plt.grid(True)
plt.tight_layout()
plt.savefig('reward_curve_no_attn.png')
plt.show()
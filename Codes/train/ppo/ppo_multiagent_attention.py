from time import time
import ray
from ray import tune
import os
import logging
from ray.tune.logger import UnifiedLogger
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pprint
from env.rllib.multiagent_env_rllib_30032026 import RenderCallback, env_creator
from models.local_attention_30032026 import AttentionModule

log_dir = "./training_logs"
os.makedirs(log_dir, exist_ok=True)

# This forces all print() statements and errors into a text file
logging.basicConfig(
    filename="./training_logs/console_output.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
# =========================================================
# Custom Model (Attention + Policy)
# =========================================================

class AttentionPolicyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.d_model = 64

        # attention module
        self.attn = AttentionModule(d_model=self.d_model)

        # policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 6 agents × 2 actions per agent
        self.num_agents = 6
        self.mean_layer = nn.Linear(64, 2)
        self.log_std = nn.Parameter(torch.zeros(self.num_agents * 2))

        # value function
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"]

        ego = obs["ego"].float()
        neighbors = obs["neighbors"].float()

        # -------- shape handling --------
        if ego.dim() == 2:
            batch_size = 1
            num_agents = ego.size(0)

            ego_flat = ego
            neigh_flat = neighbors

        elif ego.dim() == 3:
            batch_size, num_agents, _ = ego.shape

            ego_flat = ego.reshape(batch_size * num_agents, -1)
            neigh_flat = neighbors.reshape(
                batch_size * num_agents,
                neighbors.size(2),
                neighbors.size(3)
            )
        else:
            raise ValueError(f"Unexpected ego shape {ego.shape}")

        # -------- attention --------
        context, _ = self.attn(ego_flat, neigh_flat)

        # -------- policy --------
        x = self.policy_head(context)
        mean_per_agent = self.mean_layer(x)

        # -------- value --------
        value_per_agent = self.value_head(context).squeeze(-1)

        if batch_size == 1:
            self._value_out = value_per_agent.mean().unsqueeze(0)
        else:
            self._value_out = value_per_agent.reshape(batch_size, num_agents).mean(dim=1)

        # -------- action output --------
        mean = mean_per_agent.reshape(batch_size, num_agents * 2)

        log_std = self.log_std.unsqueeze(0).expand(batch_size, -1)

        return torch.cat([mean, log_std], dim=1), state

    def value_function(self):
        return self._value_out

# =========================================================
# Register everything
# =========================================================

ModelCatalog.register_custom_model("attn_model", AttentionPolicyModel)
register_env("lane_change_env", env_creator)
ray.init(
    # log_to_driver=True
)

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
        model={"custom_model": "attn_model"},
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        gamma=0.99,
        lr=3e-4,
        entropy_coeff=0.01,
    )
    # .callbacks(RenderCallback)
)
algo = config.build()

# =========================================================
# Train using Ray Tune (handles logging automatically)
# =========================================================

# tune.run(
#     "PPO",
#     config=config.to_dict(),
#     local_dir="./training_logs",
#     name="ppo_attention_run1",
#     stop={"training_iteration": 2000},
#     checkpoint_freq=1,
#     checkpoint_at_end=True
# )
# =========================================================
# Training Loop
# =========================================================

reward_history = []
collision_history = []
start_time = time()

for i in range(100):

    iter_start_time = time()
    result = algo.train()
    iter_end_time = time()
    reward_mean = result["episode_reward_mean"]
    collision_rate = result.get("custom_metrics", {}).get("collision_rate_mean", 0.0)
    collision_episode = result.get("custom_metrics", {}).get("collision_episode_mean", 0.0)
    reward_history.append(reward_mean)
    collision_history.append(collision_rate)
    iter_duration = iter_end_time - iter_start_time
    total_elapsed = iter_end_time - start_time
    print(
        f"PPO (A) {i} | "
        f"Reward: {reward_mean:.2f} | "
        f"CollisionRate: {collision_rate:.4f} | "
        f"CollisionEpisode: {collision_episode:.4f} | "
        f"IterTime: {iter_duration:.2f}s | "
        f"Total: {total_elapsed/60:.2f}m"
    )

    if i % 50 == 0:
        checkpoint = algo.save()

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(10, 12))

# -------- Reward --------
plt.subplot(3, 1, 1)
plt.plot(reward_history)
plt.title("Reward")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.grid()

# -------- Collision Rate --------
plt.subplot(3, 1, 2)
plt.plot(collision_history)
plt.title("Collision Rate (per step)")
plt.xlabel("Iteration")
plt.ylabel("Rate")
plt.grid()

# -------- Collision Episode --------
plt.subplot(3, 1, 3)
# plt.plot(collision_episode_history)
plt.title("Collision Episode Rate")
plt.xlabel("Iteration")
plt.ylabel("Rate")
plt.grid()

plt.tight_layout()
plt.savefig("ppo_all_metrics_with_attention.png")
plt.show()
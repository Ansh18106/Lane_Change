from time import time
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from env.rllib.multiagent_env_rllib_mappo_12042026 import env_creator
from env.utils.collision_callback import CollisionCallback


NUM_AGENTS = 16

# =========================================================
# MAPPO MODEL WITHOUT ATTENTION (FOR ABLATION)
# =========================================================

class MAPPONoAttentionModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.d_model = 64
        self.num_agents = NUM_AGENTS

        # -------- Simple Encoder (NO ATTENTION) --------
        self.encoder = nn.Sequential(
            nn.Linear(6 + 5*7, 128),   # ego + flattened neighbors
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # -------- Actor --------
        self.actor = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(64, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

        # -------- Centralized Critic --------
        self.critic = nn.Sequential(
            nn.Linear(self.num_agents * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_dict, state, seq_lens):

        obs = input_dict["obs"]

        ego = obs["ego"].float()              # (B, 6)
        neighbors = obs["neighbors"].float()  # (B, N, 7)
        global_state = obs["global"].float()

        # -------- Flatten neighbors --------
        B = neighbors.shape[0]
        neighbors_flat = neighbors.reshape(B, -1)  # (B, 5*7)

        # -------- Encode --------
        x = torch.cat([ego, neighbors_flat], dim=1)
        context = self.encoder(x)

        # -------- Actor --------
        x = self.actor(context)
        mean = self.mean_layer(x)
        log_std = self.log_std.expand_as(mean)

        # -------- Critic --------
        self._value_out = self.critic(global_state).squeeze(-1)

        return torch.cat([mean, log_std], dim=1), state

    def value_function(self):
        return self._value_out


# =========================================================
# Register
# =========================================================

ModelCatalog.register_custom_model("mappo_no_attn", MAPPONoAttentionModel)
register_env("lane_env", env_creator)

ray.init()

# =========================================================
# CONFIG (MAPPO)
# =========================================================

config = (
    PPOConfig()
    .environment("lane_env", env_config={"num_agents": NUM_AGENTS})
    .framework("torch")
    .rollouts(num_rollout_workers=0)
    .multi_agent(
        policies={
            "shared_policy": (
                None,
                env_creator({"num_agents": NUM_AGENTS}).observation_space,
                env_creator({"num_agents": NUM_AGENTS}).action_space,
                {}
            )
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
    )
    .training(
        model={"custom_model": "mappo_no_attn"},
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        gamma=0.99,
        lr=1e-4,
        entropy_coeff=0.01,
    )
    .callbacks(CollisionCallback)
)

algo = config.build()

# =========================================================
# TRAIN
# =========================================================
env = env_creator({"num_agents": NUM_AGENTS})
obs, _ = env.reset()

reward_history = []
collision_history = []
collision_episode_history = []
start_time = time()

for i in range(200):

    iter_start_time = time()
    result = algo.train()
    iter_end_time = time()
    reward_mean = result["episode_reward_mean"]
    collision_rate = result.get("custom_metrics", {}).get("collision_rate_mean", 0.0)
    collision_episode = result.get("custom_metrics", {}).get("collision_episode_mean", 0.0)
    reward_history.append(reward_mean)
    collision_history.append(collision_rate)
    collision_episode_history.append(collision_episode)
    iter_duration = iter_end_time - iter_start_time
    total_elapsed = iter_end_time - start_time
    print(
        f"MAPPO (No Attention) {i} | "
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
plt.plot(collision_episode_history)
plt.title("Collision Episode Rate")
plt.xlabel("Iteration")
plt.ylabel("Rate")
plt.grid()

plt.tight_layout()
plt.savefig("mappo_no_attn_metrics.png")
plt.show()
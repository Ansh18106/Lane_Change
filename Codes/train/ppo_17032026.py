import ray
from tqdm import tqdm
from env.multiagent_env_rllib_30032026 import *
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from models.attn_policy_30032026 import AttentionPolicy  

ModelCatalog.register_custom_model("local_attn", AttentionPolicy)


ray.init()

register_env("lane_change_env", env_creator)

config = (
    PPOConfig()
    .environment(
        env="lane_change_env",
        env_config={"num_agents": NUM_AGENTS}
    )
    .framework("torch")
    .rollouts(num_rollout_workers=0)
    .callbacks(RenderCallback)
)

algo = config.build()

for i in range(2):

    result = algo.train()

    print(
        f"Iteration {i}",
        result["episode_reward_mean"]
    )

print("Starting Test Run...")
for __ in range(10):
    env = MultiAgentLaneChangeEnv(num_agents=NUM_AGENTS)
    obs, _ = env.reset()
    play = int(input())
    if (play):
        for _ in tqdm(range(500)):

            action = algo.get_policy().compute_single_action(obs)

            obs, reward, terminated, truncated, _ = env.step(action[0])

            env.render()

            if terminated or truncated:
                break
    else: continue
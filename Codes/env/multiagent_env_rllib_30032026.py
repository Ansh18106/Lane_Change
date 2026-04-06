import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from .pygame_visualization import LaneRenderer
from .reward_function_17032026 import RewardFunction
from .compute_safe_acc_30032026 import ComputeSafeACC

NUM_AGENTS = 6

class RenderCallback(DefaultCallbacks):

    def on_episode_step(self, *, worker, base_env, episode, **kwargs):

        env = base_env.get_sub_environments()[0]

        env.render()


def env_creator(config):
    return MultiAgentLaneChangeEnv(**config)

class MultiAgentLaneChangeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_lanes=3, num_agents=3, gamma=0.99, lambda_x=1, lambda_y=2, lambda_collision=5, lambda_control=0.05, d_safe=0.2, target_lane_reward=20):
        super().__init__()

        # --------------------------
        # Environment Parameters
        # --------------------------
        self.num_lanes = num_lanes
        self.num_agents = num_agents
        self.dt = 0.1
        self.max_steps = 500

        self.x_start = 0.0
        self.x_end = 30.0

        self.lane_centers = np.array([0.525, 1.575, 2.625])
        self.lane_width = 1.05

        # --------------------------
        # Constraints
        # --------------------------

        # Velocity constraints
        self.v_min = 0.0
        self.v_max = 5.0

        # Angular velocity constraints
        self.omega_min = -np.pi/4
        self.omega_max = np.pi/4

        # State constraints
        self.y_min = self.lane_centers[0] - self.lane_width/2
        self.y_max = self.lane_centers[-1] + self.lane_width/2
        self.theta_min = -np.pi/6
        self.theta_max = np.pi/6

        # Acceleration bounds (input constraints)
        self.a_min = -2.0
        self.a_max = 2.0
        self.alpha_min = -2.0
        self.alpha_max = 2.0

        # --------------------------
        # Reward Coefficients
        # --------------------------

        self.gamma = gamma

        # potential function weights
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y

        # collision penalty weight
        self.lambda_collision = lambda_collision

        # control penalty weight
        self.lambda_control = lambda_control

        # safety distance
        self.d_safe = d_safe

        # terminal reward
        self.target_lane_reward = target_lane_reward

        # --------------------------
        # Gym Spaces
        # --------------------------

        self.action_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(self.num_agents * 2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "ego": spaces.Box(low=-1.0, high=1.0, shape=(self.num_agents, 6), dtype=np.float32),
            "neighbors": spaces.Box(low=-1.0, high=1.0, shape=(self.num_agents, 5, 7), dtype=np.float32)
        })

        # Rendering
        self.renderer = LaneRenderer(
        self.x_end,
        self.y_min,
        self.y_max,
        self.lane_centers,
        self.lane_width
    )

        self.reset()

    # ------------------------------------------------
    # Reset
    # ------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.agents = {}

        for i in range(self.num_agents):
            self.agents[i] = {
                "x": (self.x_end - self.x_start) * self.np_random.random() + self.x_start,
                "y": self.lane_centers[self.np_random.integers(0, 3)],
                "target_lane": self.np_random.integers(0, 3),
                "v": 0.5,
                "theta": 0.0,
                "omega": 0.0,
            }
            # print(self.agents[i])

        return self._get_joint_obs(), {}

    # ------------------------------------------------
    # Observation
    # ------------------------------------------------

    def _get_ego_state(self, i):
        ego = self.agents[i]

        current_lane = np.argmin(np.abs(self.lane_centers - ego["y"]))
        delta_lane = ego["target_lane"] - current_lane
        d_deadline = (self.x_end - ego["x"]) / self.x_end

        ego_feat = np.array([
            ego["v"] / self.v_max,
            ego["theta"] / self.theta_max,
            ego["omega"] / self.omega_max,
            current_lane / (self.num_lanes - 1),
            delta_lane / self.num_lanes,
            d_deadline
        ], dtype=np.float32)

        return ego_feat
    
    def _get_nearby_agents(self, i, max_neighbors=5, dx_thresh=0.3, dy_thresh=0.3):
        ego = self.agents[i]
        neighbors = []

        for j in range(self.num_agents):
            if i == j:
                continue

            other = self.agents[j]

            dx = (other["x"] - ego["x"]) / self.x_end
            dy = (other["y"] - ego["y"]) / (self.y_max - self.y_min)

            if abs(dx) < dx_thresh and abs(dy) < dy_thresh:
                neighbors.append((j, dx, dy))

        # sort by distance (important for stability)
        neighbors.sort(key=lambda x: abs(x[1]))

        return neighbors[:max_neighbors]
    
    def _get_neighbor_state(self, i, neighbor_list, max_neighbors=5):
        ego = self.agents[i]

        neighbor_feats = []

        for (j, dx, dy) in neighbor_list:
            other = self.agents[j]

            dv = (other["v"] - ego["v"]) / self.v_max

            current_lane_j = np.argmin(np.abs(self.lane_centers - other["y"]))
            target_flag = 1.0 if current_lane_j == ego["target_lane"] else 0.0

            feat = np.array([
                dx,
                dy,
                dv,
                other["theta"] / self.theta_max,
                other["omega"] / self.omega_max,
                current_lane_j / (self.num_lanes - 1),
                target_flag
            ], dtype=np.float32)

            neighbor_feats.append(feat)

        # padding (critical for transformer)
        while len(neighbor_feats) < max_neighbors:
            neighbor_feats.append(np.zeros(7, dtype=np.float32))

        return np.array(neighbor_feats)

    def _get_joint_obs(self):

        ego_list = []
        neighbors_list = []

        for i in range(self.num_agents):

            ego_feat = self._get_ego_state(i)

            neighbors_idx = self._get_nearby_agents(i)

            neighbor_feats = self._get_neighbor_state(i, neighbors_idx)

            ego_list.append(ego_feat)
            neighbors_list.append(neighbor_feats)

        return {
            "ego": np.array(ego_list, dtype=np.float32),
            "neighbors": np.array(neighbors_list, dtype=np.float32)
        }

    # ------------------------------------------------
    # SAFE SET FOR COLLISION AVOIDANCE
    # ------------------------------------------------

    def _compute_safe_accel_bounds(self, i):
        """
        Simple safety model:
        Maintain minimum longitudinal gap.
        Returns tightened (a_min_i, a_max_i)
        """
        return ComputeSafeACC.compute_safe_acc(self, self.agents[i])
    
    # ------------------------------------------------
    # Reward Functions
    # ------------------------------------------------

    def _compute_reward(self, i, prev_agent, a_lin, a_ang):
        return RewardFunction(self.__dict__)._compute_reward(i, prev_agent, a_lin, a_ang)
    
    # ------------------------------------------------
    # Check Collision
    # ------------------------------------------------

    def _check_collision(self):
        positions = np.array([[a["x"], a["y"]] for a in self.agents.values()])

        for i in range(len(positions)):
            dists = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
            if np.any(dists < self.d_safe):
                return True
        return False

    # ------------------------------------------------
    # STEP
    # ------------------------------------------------

    def step(self, action):

        prev_agents = {i: self.agents[i].copy() for i in range(self.num_agents)}

        self.step_count += 1
        total_reward = 0.0

        for i in range(self.num_agents):

            agent = self.agents[i]

            # ----------------------------
            # Safe-set projection
            # ----------------------------
            a_min_i, a_max_i = self._compute_safe_accel_bounds(i)

            a_lin = np.clip(action[2*i], a_min_i, a_max_i)
            a_ang = np.clip(action[2*i+1],
                            self.alpha_min, self.alpha_max)

            # ----------------------------
            # Velocity update
            # v_{t+1} = v_t + a dt
            # ----------------------------
            agent["v"] = np.clip(
                agent["v"] + a_lin * self.dt,
                self.v_min,
                self.v_max
            )

            agent["omega"] = np.clip(
                agent["omega"] + a_ang * self.dt,
                self.omega_min,
                self.omega_max
            )

            # ----------------------------
            # State update
            # x_{t+1}, y_{t+1}, theta_{t+1}
            # ----------------------------
            agent["theta"] += agent["omega"] * self.dt
            agent["theta"] = np.clip(
                agent["theta"],
                self.theta_min,
                self.theta_max
            )
            
            agent["x"] += agent["v"] * np.cos(agent["theta"]) * self.dt
            agent["x"] = np.clip(agent["x"], self.x_start, self.x_end)


            agent["y"] += agent["v"] * np.sin(agent["theta"]) * self.dt

            # State constraint on y
            agent["y"] = np.clip(agent["y"],
                                 self.y_min,
                                 self.y_max)

            # ----------------------------
            # Reward
            # ----------------------------
            
            total_reward += self._compute_reward(i, prev_agents[i], a_lin, a_ang)

        collision = self._check_collision()

        if collision:
            total_reward -= self.lambda_collision * 10  # large penalty for collision

        terminated = (
            collision or 
            all(self.agents[i]["x"] >= self.x_end for i in range(self.num_agents))
        )

        truncated = self.step_count >= self.max_steps

        return self._get_joint_obs(), total_reward, terminated, truncated, {}

    # ------------------------------------------------
    # RENDER
    # ------------------------------------------------

    def render(self):
        self.renderer.render(self.agents)

    # ------------------------------------------------
    # GRID OBSERVATION (for CNN policy)
    # ------------------------------------------------

    def _get_grid_obs(self):
        H, W = 10, 30   # (lanes × longitudinal discretization)
        C = 4           # channels

        grid = np.zeros((H, W, C), dtype=np.float32)

        for i in range(self.num_agents):
            a = self.agents[i]

            # discretize
            row = int((a["y"] - self.y_min) / (self.y_max - self.y_min) * (H - 1))
            col = int(a["x"] / self.x_end * (W - 1))

            # channels
            grid[row, col, 0] = 1.0                  # occupancy
            grid[row, col, 1] = a["v"] / self.v_max  # velocity
            grid[row, col, 2] = a["omega"] / self.omega_max
            grid[row, col, 3] = i / self.num_agents  # agent id encoding

        return grid

# # --- Execution ---
# ray.init()

# register_env("lane_change_env", env_creator)

# config = (
#     PPOConfig()
#     .environment(
#         env="lane_change_env",
#         env_config={"num_agents": NUM_AGENTS}
#     )
#     .framework("torch")
#     .rollouts(num_rollout_workers=0)
#     .callbacks(RenderCallback)
# )

# algo = config.build()

# for i in range(200):

#     result = algo.train()

#     print(
#         f"Iteration {i}",
#         result["episode_reward_mean"]
#     )

# print("Starting Test Run...")
# for __ in range(10):
#     env = MultiAgentLaneChangeEnv(num_agents=NUM_AGENTS)
#     obs, _ = env.reset()
#     play = int(input())
#     if (play):
#         for _ in tqdm(range(500)):

#             action = algo.get_policy().compute_single_action(obs)

#             obs, reward, terminated, truncated, _ = env.step(action[0])

#             env.render()

#             if terminated or truncated:
#                 break
#     else: continue
from .utils.pygame_visualization import LaneRenderer
from .utils.reward_function_17032026 import RewardFunction
from .utils.compute_safe_acc_30032026 import ComputeSafeACC
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

NUM_AGENTS = 6

class RenderCallback(DefaultCallbacks):

    def on_episode_step(self, *, worker, base_env, episode, **kwargs):

        env = base_env.get_sub_environments()[0]

        env.render()


def env_creator(config):
    return MultiAgentLaneChangeEnv(**config)

class MultiAgentLaneChangeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_lanes=3, num_agents=3, gamma=0.99, lambda_x=1, lambda_y=2, lambda_collision=5, lambda_control=0.05, d_safe=1, target_lane_reward=20):
        super().__init__()

        # --------------------------
        # Environment Parameters
        # --------------------------
        self.agents = {}
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

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents * 8,),
            dtype=np.float32
        )

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
                "x": self.x_start,
                "y": self.lane_centers[self.np_random.integers(0, 3)],
                "target_lane": self.np_random.integers(0, 3),
                "v": 0.5,
                "theta": 0.0,
                "omega": 0.0,
            }

        return self._get_joint_obs(), {}

    # ------------------------------------------------
    # Observation
    # ------------------------------------------------

    def _get_joint_obs(self):

        obs = []

        for i in range(self.num_agents):
            a = self.agents[i]
            x_norm = a["x"] / self.x_end
            y_norm = (a["y"] - self.y_min) / (self.y_max - self.y_min)
            v_norm = a["v"] / self.v_max
            omega_norm = a["omega"] / self.omega_max
            theta_sin = np.sin(a["theta"])
            theta_cos = np.cos(a["theta"])
            target_lane_norm = a["target_lane"] / (len(self.lane_centers) - 1)
            dist_goal_norm = (self.x_end - a["x"]) / self.x_end

            obs.append([
                x_norm,
                y_norm,
                v_norm,
                theta_sin,
                theta_cos,
                omega_norm,
                target_lane_norm,
                dist_goal_norm
            ])

        return np.array(obs, dtype=np.float32).flatten()

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

        terminated = all(
            self.agents[i]["x"] >= self.x_end
            for i in range(self.num_agents)
        )
        # if self.step_count >= 1:
            # terminated = True

        truncated = self.step_count >= self.max_steps

        return self._get_joint_obs(), total_reward, terminated, truncated, {}

    # ------------------------------------------------
    # RENDER
    # ------------------------------------------------

    def render(self):
        self.renderer.render(self.agents)


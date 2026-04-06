import matplotlib
matplotlib.use("TkAgg")   # critical for real-time animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]   # unwrap VecEnv
        env.render()
        return True

class MultiAgentLaneChangeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_agents=3, gamma=0.99, lambda_x=1, lambda_y=2, lambda_collision=5, lambda_control=0.05, d_safe=1, target_lane_reward=20):
        super().__init__()

        # --------------------------
        # Environment Parameters
        # --------------------------
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
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        plt.ion()

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
            print(self.agents[i])

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

        # min_gap = 2.0
        # ego = self.agents[i]

        # a_min_i = self.a_min
        # a_max_i = self.a_max

        # for j in range(self.num_agents):
        #     if i == j:
        #         continue

        #     other = self.agents[j]

        #     dx = other["x"] - ego["x"]
        #     dy = abs(other["y"] - ego["y"])

        #     # Same lane check
        #     if dy < self.lane_width/2:
        #         if 0 < dx < min_gap:
        #             # Too close → forbid acceleration
        #             a_max_i = 0.0

        return self.a_min, self.a_max
    
    # ------------------------------------------------
    # Reward Functions
    # ------------------------------------------------

    def _potential(self, agent):
        dx = self.x_end - agent["x"]
        target_y = self.lane_centers[agent["target_lane"]]
        ey = agent["y"] - target_y

        phi = -self.lambda_x * dx - self.lambda_y * ey**2

        return phi
    
    def _collision_penalty(self, ego):
        penalty = 0.0

        for j in range(self.num_agents):

            if ego == self.agents[j]: continue

            other = self.agents[j]

            dx = ego["x"] - other["x"]
            dy = ego["y"] - other["y"]

            d = np.sqrt(dx**2 + dy**2)

            penalty += 1 / (d - self.d_safe + 1e-3)

        return - (self.lambda_collision * penalty)
    

    def _control_penalty(self, a_lin, a_ang):

        r_comfort = self.lambda_control * (a_lin**2 + a_ang**2)

        return -r_comfort
    
    def _goal_reward(self, agent):

        target_y = self.lane_centers[agent["target_lane"]]

        if abs(agent["y"] - target_y) < 0.1 :
            if (agent["x"] > self.x_end):
                return self.target_lane_reward
            return self.target_lane_reward / 2
        
        return 0.0
    
    def _compute_reward(self, i, prev_agent, a_lin, a_ang):
        reward = 0.0

        agent = self.agents[i]

        phi_prev = self._potential(prev_agent)
        phi_now = self._potential(agent)

        shaping = self.gamma*phi_now - phi_prev

        control_cost = self._control_penalty(a_lin, a_ang)

        collision_cost = self._collision_penalty(agent)

        goal = self._goal_reward(agent)

        reward += shaping + collision_cost + control_cost + goal

        return reward


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

        truncated = self.step_count >= self.max_steps

        return self._get_joint_obs(), total_reward, terminated, truncated, {}

    # ------------------------------------------------
    # RENDER
    # ------------------------------------------------

    def render(self):
        self.ax.clear()

        # Draw lanes
        for center in self.lane_centers:
            self.ax.axhline(center - self.lane_width/2, color='black', lw=2)
            self.ax.axhline(center + self.lane_width/2, color='black', lw=2)
            self.ax.axhline(center, color='gray', linestyle='--', alpha=0.5)

        car_len, car_wid = 0.8, 0.4

        for i, agent in self.agents.items():

            rect = patches.Rectangle(
                (-car_len/2, -car_wid/2),
                car_len,
                car_wid,
                color='blue'
            )

            transform = (
                patches.transforms.Affine2D()
                .rotate(agent["theta"])
                .translate(agent["x"], agent["y"])
                + self.ax.transData
            )

            rect.set_transform(transform)
            self.ax.add_patch(rect)

            self.ax.text(agent["x"], agent["y"]+0.3, f"A{i}", fontsize=8)

        self.ax.set_xlim(0, self.x_end)
        self.ax.set_ylim(self.y_min-0.5, self.y_max+0.5)
        self.ax.set_aspect('equal')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

# --- Execution ---
env = MultiAgentLaneChangeEnv(num_agents=9)

model = PPO(
    "MlpPolicy",
    env,
    n_steps=256,
    batch_size=64,
    verbose=1
)

model.learn(
    total_timesteps=2048,
    callback=RenderCallback(),
    progress_bar=True
)

# Final Test Run
print("Starting Test Run...")
obs, _ = env.reset()

for _ in tqdm(range(300), desc="Test Run"):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        break

plt.ioff()
plt.show()

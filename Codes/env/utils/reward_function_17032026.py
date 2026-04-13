import numpy as np

class RewardFunction:
    def __init__(self, env_config):
        self.x_end = env_config["x_end"]
        self.lane_centers = env_config["lane_centers"]
        self.lambda_x = env_config["lambda_x"]
        self.lambda_y = env_config["lambda_y"]
        self.lambda_collision = env_config["lambda_collision"]
        self.lambda_control = env_config["lambda_control"]
        self.gamma = env_config["gamma"]
        self.d_safe = env_config["d_safe"]
        self.target_lane_reward = env_config["target_lane_reward"]
        self.agents = env_config["agents"]
        self.num_agents = env_config["num_agents"]

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

        # collision_cost = self._collision_penalty(agent)

        goal = self._goal_reward(agent)

        reward += shaping + control_cost + goal
        # reward += shaping + collision_cost + control_cost + goal

        return reward
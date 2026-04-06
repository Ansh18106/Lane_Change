import numpy as np

class ComputeSafeACC:
    def __init__(self, env_config):
        self.d_safe = env_config["d_safe"]
        self.v_max = env_config["v_max"]
        self.dt = env_config["dt"]
        self.a_min = env_config["a_min"]
        self.a_max = env_config["a_max"]
        self.agents = env_config["agents"]
        self.num_agents = env_config["num_agents"]

    def compute_safe_acc(self, ego):
        # dx = other["x"] - ego["x"]
        # dy = other["y"] - ego["y"]

        # d = np.sqrt(dx**2 + dy**2)

        # if d < self.d_safe:
        #     # Compute the required deceleration to maintain safe distance
        #     v_rel = ego["v"] - other["v"]
        #     a_safe = - (v_rel**2) / (2 * (d - self.d_safe + 1e-3))
        #     return max(a_safe, -self.v_max / self.dt)  # Limit deceleration to max possible
        # else:
        #     return 0.0
        return self.a_min, self.a_max
        

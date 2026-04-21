from ray.rllib.algorithms.callbacks import DefaultCallbacks

class CollisionCallback(DefaultCallbacks):

    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["collisions"] = 0
        episode.user_data["steps"] = 0

    def on_episode_step(self, *, episode, **kwargs):
        episode.user_data["steps"] += 1

        # Use the official public API to check Agent 0's info dict safely
        agent_info = episode.last_info_for("__common__")
        
        if agent_info and agent_info.get("collision", 0) > 0:
            # We don't need a for-loop because if one crashes, they all crash.
            episode.user_data["collisions"] += 1 

    def on_episode_end(self, *, episode, **kwargs):
        # THE FIX: Check the final info dict one last time. 
        # If the environment terminated on a crash, on_episode_step likely missed it.
        agent_info = episode.last_info_for("__common__")
        
        if agent_info and agent_info.get("collision", 0) > 0:
            # Ensure it counts as at least 1 crash
            if episode.user_data["collisions"] == 0:
                episode.user_data["collisions"] = 1

        collisions = episode.user_data["collisions"]
        steps = episode.user_data["steps"]

        # -------- metric 1: step-wise --------
        collision_rate = collisions / max(steps, 1)

        # -------- metric 2: episode-wise (better) --------
        collision_episode = int(collisions > 0)

        # -------- log --------
        episode.custom_metrics["collision_rate"] = collision_rate
        episode.custom_metrics["collision_episode"] = collision_episode
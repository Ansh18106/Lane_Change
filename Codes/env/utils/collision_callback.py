from ray.rllib.algorithms.callbacks import DefaultCallbacks

class CollisionCallback(DefaultCallbacks):

    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["collisions"] = 0
        episode.user_data["steps"] = 0

    def on_episode_step(self, *, episode, **kwargs):

        infos = getattr(episode, "_last_infos", {})  # RLlib-safe

        if infos:
            for agent_id, info in infos.items():
                if "collision" in info:
                    episode.user_data["collisions"] += info["collision"]
                    break  # count once per step

        episode.user_data["steps"] += 1

    def on_episode_end(self, *, episode, **kwargs):

        collisions = episode.user_data["collisions"]
        steps = episode.user_data["steps"]

        # -------- metric 1: step-wise --------
        collision_rate = collisions / max(steps, 1)

        # -------- metric 2: episode-wise (better) --------
        collision_episode = int(collisions > 0)

        # -------- log --------
        episode.custom_metrics["collision_rate"] = collision_rate
        episode.custom_metrics["collision_episode"] = collision_episode
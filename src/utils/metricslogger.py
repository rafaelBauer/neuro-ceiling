import wandb
from collections import deque

from utils.human_feedback import HumanFeedback


class EpisodeMetrics:
    def __init__(self, episode_number: int):
        self.__reward: float = 0
        self.__num_steps: int = 0
        self.__corrected_steps = 0
        self.__good_steps = 0
        self.__bad_steps = 0
        self.__EPISODE_NUMBER = episode_number

    def log_step(self, reward, feedback: HumanFeedback):
        self.__reward += reward.item()
        self.__num_steps += 1
        if feedback == HumanFeedback.CORRECTED:
            self.__corrected_steps += 1
        elif feedback == HumanFeedback.GOOD:
            self.__good_steps += 1
        elif feedback == HumanFeedback.BAD:
            self.__bad_steps += 1
        return

    @property
    def reward(self):
        return self.__reward

    @property
    def num_steps(self):
        return self.__num_steps

    @property
    def corrected_steps(self):
        return self.__corrected_steps

    @property
    def good_steps(self):
        return self.__good_steps

    @property
    def bad_steps(self):
        return self.__bad_steps

    @property
    def corrected_rate(self):
        if self.__num_steps == 0:
            return 0
        return self.__corrected_steps / self.__num_steps

    @property
    def good_rate(self):
        if self.__num_steps == 0:
            return 0
        return self.__good_steps / self.__num_steps

    @property
    def bad_rate(self):
        if self.__num_steps == 0:
            return 0
        return self.__bad_steps / self.__num_steps

    @property
    def episode_number(self):
        return self.__EPISODE_NUMBER

    def __str__(self):
        return (
            f"Episode {self.episode_number}: "
            f"      Reward: {self.reward}"
            f"      Steps: {self.num_steps}"
            f"      Corrected Rate: {self.corrected_rate}"
            f"      Good Rate: {self.good_rate},"
            f"      Bad Rate: {self.bad_rate}"
        )


class MetricsLogger:
    def __init__(self):
        self.total_successes = 0
        self.total_episodes = 0
        self.total_steps = 0
        self.total_corrected_steps = 0
        self.total_good_steps = 0
        self.total_bad_steps = 0
        self.episode_metrics = deque(maxlen=1)

        return

    def log_episode(self, episode_metrics: EpisodeMetrics):
        log_episode_metrics = {
            "reward": episode_metrics.reward,
            "ep_corrected_rate": episode_metrics.corrected_rate,
            "ep_good_rate": episode_metrics.good_rate,
            "ep_bad_rate": episode_metrics.bad_rate,
            "episode": episode_metrics.episode_number,
        }
        self.append(log_episode_metrics)
        self.total_episodes += 1
        if episode_metrics.reward > 0:
            self.total_successes += 1
        self.total_steps += episode_metrics.num_steps
        self.total_corrected_steps += episode_metrics.corrected_steps
        self.total_good_steps += episode_metrics.good_steps
        self.total_bad_steps += episode_metrics.bad_steps
        return

    def log_session(self):
        if self.total_episodes == 0:
            success_rate = 0
        else:
            success_rate = self.total_successes / self.total_episodes
        if self.total_steps == 0:
            corrected_rate = 0
            good_rate = 0
            bad_rate = 0
        else:
            corrected_rate = self.total_corrected_steps / self.total_steps
            good_rate = self.total_good_steps / self.total_steps
            bad_rate = self.total_bad_steps / self.total_steps
        wandb.run.summary["success_rate"] = success_rate
        wandb.run.summary["total_corrected_rate"] = corrected_rate
        wandb.run.summary["total_good_rate"] = good_rate
        wandb.run.summary["total_bad_rate"] = bad_rate
        return

    def append(self, episode_metrics):
        self.episode_metrics.append(episode_metrics)
        return

    def pop(self):
        return self.episode_metrics.popleft()

    def empty(self):
        return len(self.episode_metrics) == 0

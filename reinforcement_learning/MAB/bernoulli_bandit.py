import numpy as np
from logger_utils import logger


class BernoulliBandit:
    def __init__(self, n_arms: int, seed: int = 1):
        self.seed = seed
        np.random.seed(self.seed)
        self.n_arms = n_arms
        self.probs = np.random.uniform(size=n_arms)
        self.best_arm = np.argmax(self.probs)
        self.best_reward = np.max(self.probs)

    def step(self, arm: int) -> float:
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[arm]:
            return 1
        else:
            return 0


if __name__ == "__main__":
    bandit = BernoulliBandit(n_arms=10, seed=1)
    logger.debug(f"probs: {bandit.probs}")
    logger.debug(f"best_arm: {bandit.best_arm}")
    logger.debug(f"best_reward: {bandit.best_reward}")

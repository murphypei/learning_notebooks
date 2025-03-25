import numpy as np
from bernoulli_bandit import BernoulliBandit
from logger_utils import logger
from solver import Solver, plot_results


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob: float = 1.0, seed: int = 1):
        super().__init__(bandit, seed=seed)
        self.estimates = np.array([init_prob] * self.bandit.n_arms)
        self.total_count = 0
        self.epsilon = 1

    def update_epsilon(self):
        self.epsilon = 1 / self.total_count

    def run_one_step(self):
        self.total_count += 1

        # 随着时间推移，探索的概率越来越小
        self.update_epsilon()
        logger.debug(f"epsilon: {self.epsilon}")

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.bandit.n_arms)
        else:
            action = np.argmax(self.estimates)

        reward = self.bandit.step(action)
        self.estimates[action] += 1.0 / (self.counts[action] + 1) * (reward - self.estimates[action])  # +1 防止除零
        return action


if __name__ == "__main__":
    np.random.seed(1)
    bandit = BernoulliBandit(n_arms=10)
    decaying_epsilon_greedy = DecayingEpsilonGreedy(bandit, seed=1)
    decaying_epsilon_greedy.run(num_steps=5000)
    plot_results([decaying_epsilon_greedy], ["decaying epsilon greedy"])

    
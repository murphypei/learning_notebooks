import numpy as np
from bernoulli_bandit import BernoulliBandit
from solver import Solver, plot_results


class UCB(Solver):
    def __init__(self, bandit, coef, init_prob: float = 1.0, seed: int = 1):
        super().__init__(bandit, seed=seed)
        self.coef = coef
        self.estimates = np.array([init_prob] * self.bandit.n_arms)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        ucb_values = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        action = np.argmax(ucb_values)
        reward = self.bandit.step(action)
        self.estimates[action] += 1.0 / (self.counts[action] + 1) * (reward - self.estimates[action])
        return action


if __name__ == "__main__":
    np.random.seed(1)
    bandit = BernoulliBandit(n_arms=10)
    ucb_solver = UCB(bandit, coef=1, seed=1)
    ucb_solver.run(5000)
    plot_results([ucb_solver], ["UCB"])

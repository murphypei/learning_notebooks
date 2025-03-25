import numpy as np
from bernoulli_bandit import BernoulliBandit
from solver import Solver, plot_results


class ThompsonSampling(Solver):
    def __init__(self, bandit, seed: int = 1):
        super().__init__(bandit, seed=seed)
        self.alpha = np.ones(self.bandit.n_arms)
        self.beta = np.ones(self.bandit.n_arms)

    def run_one_step(self):
        samples = np.random.beta(self.alpha, self.beta)
        action = np.argmax(samples)
        reward = self.bandit.step(action)
        self.alpha[action] += reward
        self.beta[action] += 1 - reward
        return action


if __name__ == "__main__":
    np.random.seed(1)
    bandit = BernoulliBandit(n_arms=10)
    thompson_sampling_solver = ThompsonSampling(bandit, seed=1)
    thompson_sampling_solver.run(5000)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])

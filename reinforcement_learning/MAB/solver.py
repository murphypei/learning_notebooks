import matplotlib.pyplot as plt
import numpy as np
from bernoulli_bandit import BernoulliBandit
from logger_utils import logger


class Solver:
    def __init__(self, bandit: BernoulliBandit, seed: int = 1):
        self.seed = seed
        np.random.seed(self.seed)
        self.bandit = bandit
        self.counts = np.zeros(bandit.n_arms)  # 记录每个拉杆被选择的次数
        self.cumulative_regret = 0  # 累积遗憾
        self.actions = []  # 记录每次选择的拉杆
        self.regrets = []  # 记录每次的遗憾

    def update_regret(self, action: int):
        # 遗憾就是最优拉杆的期望奖励和当前拉杆的期望奖励之差
        self.cumulative_regret += self.bandit.best_reward - self.bandit.probs[action]
        logger.debug(
            f"action: {action} | cumulative_regret: {self.cumulative_regret} | {self.bandit.best_reward - self.bandit.probs[action]}"
        )
        self.regrets.append(self.cumulative_regret)

    def run_one_step(self):
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, num_steps: int):
        for _ in range(num_steps):
            action = self.run_one_step()
            self.counts[action] += 1
            self.actions.append(action)
            self.update_regret(action)


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        step_list = range(len(solver.regrets))
        plt.plot(step_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("step")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.n_arms)
    plt.legend()
    plt.savefig(f"{solver_names[0]}.png")
    plt.close()

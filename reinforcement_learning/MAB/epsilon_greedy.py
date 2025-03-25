import numpy as np
from bernoulli_bandit import BernoulliBandit
from solver import Solver, plot_results


class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon: float = 0.01, init_prob: float = 1.0, seed: int = 1):
        super().__init__(bandit, seed=seed)
        # 探索概率，越大越容易探索
        self.epsilon = epsilon
        # 初始化每个拉杆的期望奖励
        self.estimates = np.array([init_prob] * self.bandit.n_arms)

    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            # 随机选择一个拉杆，探索
            action = np.random.randint(0, self.bandit.n_arms)
        else:
            # 按照贪心算法选择期望奖励最大的拉杆
            action = np.argmax(self.estimates)

        # 根据选择的拉杆获得奖励
        reward = self.bandit.step(action)
        # 更新每个action的期望奖励
        self.estimates[action] += 1.0 / (self.counts[action] + 1) * (reward - self.estimates[action])  # +1 防止除零
        # action 就是本次选择的拉杆
        return action


if __name__ == "__main__":
    bandit = BernoulliBandit(n_arms=10, seed=1)
    epsilon_greedy_solver = EpsilonGreedy(bandit, epsilon=0.01, seed=1)
    epsilon_greedy_solver.run(5000)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [EpsilonGreedy(bandit, epsilon=e, seed=0) for e in epsilons]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

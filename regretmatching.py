import numpy as np
from itertools import combinations_with_replacement, permutations

class RegretMatchingPlayer:
    # player num is 0-idx
    def __init__(self, strategy_name, payoff_3d, player_num):
        self.NUM_ACTIONS = payoff_3d.shape[player_num]
        self.strategy_name = strategy_name
        self.actions = [i for i in range(self.NUM_ACTIONS)]

        self.regret_sum = np.zeros(self.NUM_ACTIONS)
        self.strategy = np.zeros(self.NUM_ACTIONS)
        self.strategy_sum = np.zeros(self.NUM_ACTIONS)

        self.payoff = payoff_3d[:, :, player_num]

        # Change utility of P1(0-idx) to from his own perspective
        if player_num == 1:
            self.payoff = self.payoff.T


    # Get current strategy based on positive regrets
    def get_strategy(self):
        self.strategy = np.maximum(self.regret_sum, 0)

        if np.sum(self.strategy) > 0:
            self.strategy /= np.sum(self.strategy)
        else:
            self.strategy = np.ones(self.NUM_ACTIONS) / self.NUM_ACTIONS
        
        self.strategy_sum += self.strategy
        return self.strategy

    def get_average_strategy(self):
        if np.sum(self.strategy_sum) > 0:
            return self.strategy_sum / np.sum(self.strategy_sum)
        else:
            return np.ones(self.NUM_ACTIONS) / self.NUM_ACTIONS

    def get_top_strategies(self, n=5):
        avg_strategy = self.get_average_strategy()
        sorted_indices = np.argsort(-avg_strategy)
        return [(self.strategy_name[i], avg_strategy[i]) for i in sorted_indices[:n]]

    def get_utility(self, other_strategy):
        return self.get_average_strategy() @ self.payoff @ other_strategy.T

    def update_regrets_against_strategy(self, opp_strategy):
        current_util = self.strategy @ self.payoff @ opp_strategy.T
        self.regret_sum += self.payoff @ opp_strategy.T - current_util
        self.regret_sum = np.maximum(self.regret_sum, 0)
    
    def accumulate_strategy(self, strategy):
        """Track average strategy over time"""
        self.strategy_sum += strategy


def play_match(p1, p2, num_iterations):
    for _ in range(num_iterations):
        p1_strat = p1.get_strategy()
        p2_strat = p2.get_strategy()
        
        p1.update_regrets_against_strategy(p2_strat)
        p2.update_regrets_against_strategy(p1_strat)
        
        p1.accumulate_strategy(p1_strat)
        p2.accumulate_strategy(p2_strat)

def calc_nash_equilibrium(strategy_name, game_payoff, num_iterations=1000):
    p1 = RegretMatchingPlayer(strategy_name, game_payoff, 0)
    p2 = RegretMatchingPlayer(strategy_name, game_payoff, 1)

    play_match(p1, p2, num_iterations)

    print(p1.get_utility(p2.get_average_strategy()))
    print(p2.get_utility(p1.get_average_strategy()))

    print("\nPlayer 1 top strategies:")
    for strat, prob in p1.get_top_strategies(20):
        print(f"{strat}: {prob:.4f}")
    
    print("\nPlayer 2 top strategies:")
    for strat, prob in p2.get_top_strategies(20):
        print(f"{strat}: {prob:.4f}")

RPS_strategy = ("Rock", "Paper", "Scissors")
RPS_payoff = np.array([
    [[0, 0], [-1, 1], [1, -1]],
    [[1, -1], [0, 0], [-1, 1]],
    [[-1, 1], [1, -1], [0, 0]]
])


# Battle of the Sexes
BotS_strategy = ("Movie", "Game")
BotS_payoff = np.array([
    [[2, 1], [0, 0]],
    [[0, 0], [1, 2]]
])

mp_strategy = ("Left", "Right")
mp_payoff = np.array([
    [[0, 0], [0.5, -0.5]],
    [[1, -1], [0, 0]]
])


def blotto(S, N):
    strategies = set()
    for c in combinations_with_replacement(range(S + 1), N):
        if sum(c) == S:
            # Generate all unique permutations
            strategies.update(permutations(c))
    sorted_strategies = sorted(strategies)

    def calculate_payoff(s1, s2):
        p1_wins = sum(1 if a > b else 0 for a, b in zip(s1, s2))
        p2_wins = sum(1 if b > a else 0 for a, b in zip(s1, s2))
        
        if p1_wins > p2_wins:
            return (1, -1)
        elif p2_wins > p1_wins:
            return (-1, 1)
        return (0, 0)

    NUM_ACTIONS = len(sorted_strategies)

    payoff_matrix = np.zeros((NUM_ACTIONS, NUM_ACTIONS, 2))
    for i, s1 in enumerate(sorted_strategies):
        for j, s2 in enumerate(sorted_strategies):
            payoff_matrix[i, j] = calculate_payoff(s1, s2)
    return sorted_strategies, payoff_matrix

# calc_nash_equilibrium(RPS_strategy, RPS_payoff)
# calc_nash_equilibrium(BotS_strategy, BotS_payoff)
# calc_nash_equilibrium(*blotto(5, 3))
calc_nash_equilibrium(mp_strategy, mp_payoff, num_iterations=10000)
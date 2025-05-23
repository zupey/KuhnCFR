import numpy as np
from kuhnpoker import export_game_details

class Treeplex:
    def __init__(self, strategy_name, payoff, strategy_children, infoset_parent, infoset_children):
        self.NUM_SEQ = payoff.shape[0]
        self.NUM_INFOSET = len(infoset_parent)
        self.strategy = np.zeros(self.NUM_SEQ)
        self.strategy_sum = np.zeros(self.NUM_SEQ)
        self.regret_sum = np.zeros(self.NUM_SEQ)

        self.strategy_utility = np.zeros(self.NUM_SEQ)
        self.strategy_children = strategy_children
        self.infoset_parent = infoset_parent
        self.infoset_children = infoset_children

        self.strategy[0] = 1
        self.strategy = self.get_behavioral_strategy()

        assert(len(strategy_children) ==  self.NUM_SEQ)
        assert(len(infoset_parent) == len(infoset_children) == self.NUM_INFOSET)

        self.payoff = payoff

    def get_behavioral_strategy(self):
        """Gets the behavioural form strategy

        Returns:
            _type_: np array of behavioural forms strat
        """
        # Top down
        for infoset_idx in range(self.NUM_INFOSET):
            total = 0
            for child_seq in self.infoset_children[infoset_idx]:
                total += max(self.regret_sum[child_seq], 0)

            if total == 0:
                for child_seq in self.infoset_children[infoset_idx]:
                    self.strategy[child_seq] = 1.0 / len(self.infoset_children[infoset_idx])
            else:
                for child_seq in self.infoset_children[infoset_idx]:
                    self.strategy[child_seq] = max(self.regret_sum[child_seq], 0) / total
        return self.strategy

    def get_sequence_form_strategy(self):
        sequence_form_strategy = np.copy(self.strategy)
        stack = [0]
        while stack:
            idx = stack.pop()
            for child_infoset in self.strategy_children[idx]:
                for child_seq in self.infoset_children[child_infoset]:
                    sequence_form_strategy[child_seq] *= sequence_form_strategy[idx]
                    stack.append(child_seq)
        return sequence_form_strategy

    def convert_seqform_to_behav(self, strat):
        behavioural_form_strategy = np.copy(strat)
        stack = [0]
        while stack:
            idx = stack.pop()
            for child_infoset in self.strategy_children[idx]:
                for child_seq in self.infoset_children[child_infoset]:
                    behavioural_form_strategy[child_seq] /= behavioural_form_strategy[idx]
                    stack.append(child_seq)
        return behavioural_form_strategy
        

    def update_regrets_against_strategy(self, opp_strat):
        # Initialize utility at leaf nodes
        self.strategy_utility = self.payoff @ opp_strat

        # from btm up
        # for each infoset
        # 1. calc cur_utility (by taking sum of children) and add cur_utility to parent_strat
        # 2. calc regret (utility of taking pure strat - cur_util)
        # 3. max regret with 0

        for infoset_idx in range(self.NUM_INFOSET-1, -1, -1):
            # step 1
            all_infoset_util = np.array([self.strategy_utility[strat_idx] * self.strategy[strat_idx] for strat_idx in self.infoset_children[infoset_idx]])
            assert(len(all_infoset_util) <= 2)
            infoset_util = np.sum(all_infoset_util)
            self.strategy_utility[self.infoset_parent[infoset_idx]] += infoset_util

            # step 2
            for strat_idx in self.infoset_children[infoset_idx]:
                pure_strat_util = self.strategy_utility[strat_idx]
                self.regret_sum[strat_idx] += pure_strat_util - infoset_util
            # step 3
            self.regret_sum = np.maximum(self.regret_sum, 0)
        
        self.strategy = self.get_behavioral_strategy()

        
    def accumulate_strategy(self, strategy):
        self.strategy_sum += strategy

    def get_average_strategy(self, num_iterations):
        return self.strategy_sum / num_iterations

def train(p1: Treeplex, p2: Treeplex, num_iterations=1000):
    for _ in range(num_iterations):
        p1_strat = p1.get_sequence_form_strategy()
        p2_strat = p2.get_sequence_form_strategy()
        
        p1.update_regrets_against_strategy(p2_strat)
        p2.update_regrets_against_strategy(p1_strat)

        p1.accumulate_strategy(p1_strat)
        p2.accumulate_strategy(p2_strat)
    return p1.get_average_strategy(num_iterations), p2.get_average_strategy(num_iterations)

payoff_matrix, p1_strat_children, p1_infoset_parent, p1_infoset_children, p1_strat_name, p2_strat_children, p2_infoset_parent, p2_infoset_children, p2_strat_name = export_game_details()

p1_treeplex = Treeplex(p1_strat_name, payoff_matrix, p1_strat_children, p1_infoset_parent, p1_infoset_children)
p2_treeplex = Treeplex(p2_strat_name, -payoff_matrix.T, p2_strat_children, p2_infoset_parent, p2_infoset_children)

p1_strat, p2_strat = train(p1_treeplex, p2_treeplex, num_iterations=10000)
print("P1 Seq_form strat:", p1_strat)
print("P2 Seq_form strat:", p2_strat)
print("P1 Behav_form strat:", p1_treeplex.convert_seqform_to_behav(p1_strat))
print("P2 Behav_form strat:", p2_treeplex.convert_seqform_to_behav(p2_strat))


print("Utility of Kuhn Poker is", p1_strat @ payoff_matrix @ p2_strat.T)
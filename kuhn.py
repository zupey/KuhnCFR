from enum import Enum
from typing import Optional, Dict, List, Tuple
from itertools import permutations
from collections import deque
from dataclasses import dataclass
import numpy as np

starting_states = list(permutations("JQK", 2))

class Action(Enum):
    BET = "BET"
    CALL = "CALL"
    CHECK = "CHECK"
    FOLD = "FOLD"
    CHANCE = "chance"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

class Player(Enum):
    RANDOM = -1
    PLAYER_1 = 0
    PLAYER_2 = 1

    def other(self) -> "Player":
        if self == Player.RANDOM:
            return Player.RANDOM
        return Player.PLAYER_1 if self == Player.PLAYER_2 else Player.PLAYER_2

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

class Cards(Enum):
    Jack = "J"
    Queen = "Q"
    King = "K"
    Blank = None


class Node:
    def __init__(self, parent: Optional["Node"], player: Player, actions: List[Action]):
        self.parent = parent
        self.player = player
        self.actions = actions
        self.children: Dict[Action, "Node"] = {}
        self.information_set: Optional[Tuple[Cards, Tuple[Action,...]]]= None
        self.history: List[Action] = []

    def is_terminal_state(self):
        return False

    def play(self, action):
        return self.children[action]


class ChanceNode(Node):
    def __init__(self):
        super().__init__(parent=None, player=Player.RANDOM, actions=starting_states)
        self.children = {
            cards: PlayerNode(self, Player.PLAYER_1, [Action.BET, Action.CHECK], cards, [Action.CHANCE]) for cards in self.actions
        }

    def __str__(self):
        return f"""ChanceNode"""


class PlayerNode(Node):
    def __init__(
        self, parent: Node, player: Player, actions: List[Action], cards: Tuple[Cards], history: List[Action]
    ):
        super().__init__(parent, player, actions)

        self.history: List[Action] = history
        self.cards: Tuple[Cards] = cards
        self.card = cards[self.player.value]
        self.children = {
            action: PlayerNode(
                self,
                player.other(),
                self.next_actions(action),
                self.cards,
                self.history + [action],
            )
            for action in actions
        }
        self.information_set : Tuple[Cards, Tuple[Action, ...]] = (self.card, tuple(self.history))

    def next_actions(self, action: Action):
        if len(self.history) == 1 and action == Action.BET:
            return [Action.FOLD, Action.CALL]
        elif len(self.history) == 1 and action == Action.CHECK:
            return [Action.BET, Action.CHECK]
        elif self.history[-1] == Action.CHECK and action == Action.BET:
            return [Action.CALL, Action.FOLD]
        elif action == Action.CALL or action == Action.FOLD or (self.history[-1] == Action.CHECK and action == Action.CHECK):
            return []
        
    def is_terminal_state(self):
        return self.actions == []

    def evaluate(self):
        if not self.is_terminal_state():
            return None
        base = 0
        if self.cards[0] == "K" or self.cards[1] == "J":
            base = 1
        elif self.cards[0] == "J" or self.cards[1] == "K":
            base = -1

        if (self.history[-2] == Action.CHECK and self.history[-1] == Action.CHECK):
            return base

        elif self.history[-2] == Action.BET and self.history[-1] == Action.CALL:
            return base * 2
        
        elif (self.history[-2] == Action.BET and self.history[-1] == Action.FOLD):
            if self.player == Player.PLAYER_1:
                return 1
            elif self.player == Player.PLAYER_2:
                return -1

    def __repr__(self):
        return f"""PlayerNode: History:{self.history}, {self.player}, Cards:{''.join(self.cards)}, Value:{self.evaluate()}"""

    def __str__(self):
        return f"""PlayerNode: History:{self.history}, {self.player}, Cards:{''.join(self.cards)}, Value:{self.evaluate()}"""


class Infoset:
    def __init__(self, card: Cards, history: List[Action], seq_id_list: List[int], parent_seq_id: int):
        # Card and history form the unique identifier for Infoset
        self.card: Cards = card
        self.history: List[Action] = history
        self.seq_id_list: List[int] = seq_id_list
        self.parent_seq_id: int = parent_seq_id

    def __repr__(self):
        history_string = ", ".join([str(action) for action in self.history])
        return f"({self.card}/{history_string})"

class Sequence:
    def __init__(self, sequence: List[Action] = [], value: float = 1.0):
        self.sequence: List[Action] = sequence
        self.value: float = value
        self.child_infosets: List[Infoset] = []
        self.infoset: Optional[Infoset] = None

    def add_parent(self, infoset: Infoset):
        self.infoset = infoset

    def add_child_infoset(self, infoset: Infoset):
        self.child_infosets.append(infoset)

    def __repr__(self):
        actions = ", ".join([str(action) for action in self.sequence])
        return f"({self.infoset.card if self.infoset else None} {actions}, {str(self.value)})"

    def __hash__(self):
        return hash(tuple(self.sequence))

    def __eq__(self, other):
        if isinstance(other, Sequence):
            return self.sequence == other.sequence
        return False
    
    def __add__(self, other):
        if isinstance(other, Action):
            return Sequence(self.sequence + [other])
        elif isinstance(other, Sequence):
            return Sequence(self.sequence + other.sequence)

class Treeplex:
    def __init__(self, infosets: List[Infoset] = [], sequences: List[Sequence] = []):
        self.infosets: List[Infoset] = infosets
        self.sequences: List[Sequence] = sequences
    
    def add_infoset(self, infoset: Infoset):
        self.infosets.append(infoset)

    def add_sequence(self, sequence: Sequence) -> int:
        idx = len(self.sequences)
        self.sequences.append(sequence)
        return idx

    # can be improved to use hashmap instead of linear search
    def find_infoset(self, card, history) -> Optional[Infoset]:
        for infoset in self.infosets:
            if infoset.card == card and infoset.history == history:
                return infoset
        return None

    def convert_to_sequence_form(self) -> None:
        EPS = 1e-6
        stack: List[Sequence] = [self.sequences[0]]
        while stack:
            seq = stack.pop()

            for child_infoset in seq.child_infosets:
                total = sum([self.sequences[seq_id].value for seq_id in child_infoset.seq_id_list])
                if abs(total-1) > EPS:
                    print(total)
                    assert(abs(total - 1) <= EPS)
                for seq_id in child_infoset.seq_id_list:
                    self.sequences[seq_id].value = seq.value * self.sequences[seq_id].value / total
                    stack.append(self.sequences[seq_id])

@dataclass
class Leaf:
    value: int
    seq1_id: int
    seq2_id: int

root = ChanceNode()
# Generate List of Sequences in topological order (BFS)
player1_treeplex = Treeplex([], [Sequence()])
player2_treeplex = Treeplex([], [Sequence()])
seen: set[Tuple[Cards, Tuple[Action,...]]] = set()
queue: deque[Tuple[Node, int, int]] = deque([(root, 0, 0)])
leaves: List[Leaf] = []
while queue:
    node, p1_idx, p2_idx = queue.popleft()
    p1parent_seq, p2parent_seq = player1_treeplex.sequences[p1_idx], player2_treeplex.sequences[p2_idx]

    infoset = None
    if node.information_set in seen and isinstance(node, PlayerNode):
        if node.player == Player.PLAYER_1:
            infoset = player1_treeplex.find_infoset(node.card, node.history)
        elif node.player == Player.PLAYER_2:
            infoset = player2_treeplex.find_infoset(node.card, node.history)
            
    if isinstance(node, PlayerNode) and not node.is_terminal_state() and node.information_set not in seen:
        children_seq: List[int] = []
        children: List[Node] = []
        if node.player == Player.PLAYER_1:
            # Generate Children Sequences
            for action, child in node.children.items():
                child_idx = player1_treeplex.add_sequence(p1parent_seq + action)
                children_seq.append(child_idx)
                children.append(child)
            infoset = Infoset(node.card, node.history, children_seq, p1_idx)
            player1_treeplex.add_infoset(infoset)
            p1parent_seq.add_child_infoset(infoset)
            for child_seq in children_seq:
                player1_treeplex.sequences[child_seq].add_parent(infoset)
        elif node.player == Player.PLAYER_2:
            # Generate Children Sequences
            for action in node.children.keys():
                child_idx = player2_treeplex.add_sequence(p2parent_seq + action)
                children_seq.append(child_idx)
                children.append(child)
            infoset = Infoset(node.card, node.history, children_seq, p2_idx)
            player2_treeplex.add_infoset(infoset)
            p2parent_seq.add_child_infoset(infoset)
            for child_seq in children_seq:
                player2_treeplex.sequences[child_seq].add_parent(infoset)

        # add infoset into seen so we process each infoset once
        seen.add(node.information_set)
    
    # Process leaf logics
    if node.is_terminal_state() and isinstance(node, PlayerNode):
        leaves.append(Leaf(node.evaluate(), p1_idx, p2_idx))
    if isinstance(node, ChanceNode):
        for action, child in node.children.items():
                queue.append((child, p1_idx, p2_idx))
    elif isinstance(node, PlayerNode) and infoset:
        if node.player == Player.PLAYER_1:
            for id in infoset.seq_id_list:
                action = player1_treeplex.sequences[id].sequence[-1]
                child = node.children[action]
                queue.append((child, id, p2_idx))
        if node.player == Player.PLAYER_2:
            for id in infoset.seq_id_list:
                action = player2_treeplex.sequences[id].sequence[-1]
                child = node.children[action]
                queue.append((child, p1_idx, id))


# print(player1_treeplex.infosets)
# print(player1_treeplex.sequences)
# print(player2_treeplex.infosets)
# print(player2_treeplex.sequences)
# print(leaves)

payoff_matrix: List[List[float]] = [[0. for _ in range(len(player2_treeplex.sequences))] for _ in range(len(player1_treeplex.sequences))]
for leaf in leaves:
    seq1 = player1_treeplex.sequences[leaf.seq1_id]
    seq2 = player2_treeplex.sequences[leaf.seq2_id]
    # print(seq1, seq1.infoset.card, seq2, seq2.infoset.card, leaf.value)
    payoff_matrix[leaf.seq1_id][leaf.seq2_id] += leaf.value / 6


np_payoff = np.array(payoff_matrix)
# print(np_payoff)
# print('\n'.join([', '.join([str(cell) for cell in row]) for row in np_payoff]))


# As a sanity check, when playing a uniform strategy at every infoset, the value of the game should be 0.125
def calc_utility(treeplex1: Treeplex, treeplex2: Treeplex) -> float:
    # convert to sequence form just in case
    treeplex1.convert_to_sequence_form()
    treeplex2.convert_to_sequence_form()
    p1_strat = np.array([seq.value for seq in treeplex1.sequences])
    p2_strat = np.array([seq.value for seq in treeplex2.sequences])
    p1_strat = np.array([0.0,0.2,0.8,0.,1.,0.6,0.4,0.,1.,0.57,0.43,1.,0.])
    p2_strat = np.array([0.0,0.64,0.36,0.,1.,0.,1.,1.,0.,1.,0.,0.33,0.67])
    print(p1_strat)
    print(p2_strat)
    print("utility is", p1_strat@np_payoff@p2_strat)
    return p1_strat @ np_payoff @ p2_strat

# print(calc_utility(player1_treeplex, player2_treeplex))

def calc_p1_best_response(treeplex1: Treeplex, treeplex2: Treeplex):
    # p2_strat = np.array([seq.value for seq in treeplex2.sequences])
    p2_strat = np.array([1.0,0.36,0.64,0.,1.,0.,1.,1.,0.,1.,0.,0.33,0.67])
    p1_utility = np_payoff @ p2_strat
    p1_bestresponse = [0] * len(treeplex1.sequences)
    for idx in range(len(treeplex1.sequences)-1,-1,-1):
        if treeplex1.sequences[idx].child_infosets:
            for child_infoset in treeplex1.sequences[idx].child_infosets:
                best_value, best_idx = -float('inf'), -1
                for child_seq_idx in child_infoset.seq_id_list:
                    if p1_utility[child_seq_idx] > best_value:
                        best_value = p1_utility[child_seq_idx]
                        best_idx = child_seq_idx
                # plays strat with best utility 100% of the time as best response
                p1_bestresponse[best_idx] = 1
            p1_utility[idx] += best_value
    return p1_bestresponse

def calc_p2_best_response(treeplex1: Treeplex, treeplex2: Treeplex):
    p1_strat = np.array([seq.value for seq in treeplex1.sequences])
    p2_utility = p1_strat @ np_payoff
    p2_bestresponse = [0] * len(treeplex2.sequences)
    for idx in range(len(treeplex2.sequences)-1,-1,-1):
        if treeplex2.sequences[idx].child_infosets:
            for child_infoset in treeplex2.sequences[idx].child_infosets:
                best_value, best_idx = float('inf'), -1
                for child_seq_idx in child_infoset.seq_id_list:
                    if p2_utility[child_seq_idx] < best_value:
                        best_value = p2_utility[child_seq_idx]
                        best_idx = child_seq_idx
                # plays strat with best utility 100% of the time as best response
                p2_bestresponse[best_idx] = 1
            p2_utility[idx] += best_value
    return p2_bestresponse
p1_strat = np.array([seq.value for seq in player1_treeplex.sequences])
p2_strat = np.array([seq.value for seq in player2_treeplex.sequences])

# print(calc_p1_best_response(player1_treeplex, player2_treeplex))
# print(player1_treeplex.sequences)

# print(calc_p2_best_response(player1_treeplex, player2_treeplex))
# print(player2_treeplex.sequences)

"""
do this 2 times (once for each player)
for each infoset, 
1. calculate current utility (by playing current strategy)
2. calculate utility for playing each action (will be in the array similar to calculating best response)
3. take max(utility - current utility, 0) as regret and add it to the regret for that sequence
4. update strategy at each infoset in proportional to this regret

convert to sequence form

take averages of strategies
run regret minimizer on 1 person and should converge to best response
"""
def counterfactual_regret_minimizer(treeplex1: Treeplex, treeplex2: Treeplex, iterations: int):
    # initalize regret
    p1_regret = [0.] * len(treeplex1.sequences)
    p2_regret = [0.] * len(treeplex2.sequences)

    p1_strat = np.array([seq.value for seq in treeplex1.sequences])
    p2_strat = np.array([seq.value for seq in treeplex2.sequences])
    # p1_strat = np.array([1.0,0.2,0.8,0.,1.,0.6,0.4,0.,1.,0.57,0.43,1.,0.])
    # p2_strat = np.array([1.0,0.36,0.64,0.,1.,0.,1.,1.,0.,1.,0.,0.33,0.67])
    p1_avg = np.zeros_like(p1_strat)
    p2_avg = np.zeros_like(p2_strat)

    for _ in range(iterations):
        p1_utility = np_payoff @ p2_strat
        p2_utility = p1_strat @ np_payoff

        for idx in range(len(treeplex1.sequences)-1,-1,-1):
            if treeplex1.sequences[idx].child_infosets:
                for child_infoset in treeplex1.sequences[idx].child_infosets:
                    cur_utility = 0.
                    total_regret = 0.
                    # calc utility by playing current strategy
                    for child_seq_idx in child_infoset.seq_id_list:
                        cur_utility += p1_strat[child_seq_idx] * p1_utility[child_seq_idx]
                    # update regret and store total regret
                    for child_seq_idx in child_infoset.seq_id_list:
                        p1_regret[child_seq_idx] += p1_utility[child_seq_idx] - cur_utility
                        p1_regret[child_seq_idx] = max(p1_regret[child_seq_idx], 0.0001)
                        total_regret += p1_regret[child_seq_idx]
                    # update strat
                    for child_seq_idx in child_infoset.seq_id_list:
                        if total_regret == 0:
                            p1_strat[child_seq_idx] = 0
                        else:
                            p1_strat[child_seq_idx] = p1_regret[child_seq_idx] / total_regret

        for idx in range(len(treeplex2.sequences)-1,-1,-1):
            if treeplex2.sequences[idx].child_infosets:
                for child_infoset in treeplex2.sequences[idx].child_infosets:
                    cur_utility = 0.
                    total_regret = 0.
                    # calc utility by playing current strategy
                    for child_seq_idx in child_infoset.seq_id_list:
                        cur_utility -= p2_strat[child_seq_idx] * p2_utility[child_seq_idx]
                    # update regret and store total regret
                    for child_seq_idx in child_infoset.seq_id_list:
                        p2_regret[child_seq_idx] += -p2_utility[child_seq_idx] + cur_utility # opp sign for p2
                        p2_regret[child_seq_idx] = max(p2_regret[child_seq_idx], 0.0001)
                        total_regret += p2_regret[child_seq_idx]
                    # update strat
                    for child_seq_idx in child_infoset.seq_id_list:
                        if total_regret == 0:
                            p2_strat[child_seq_idx] = 0
                        else: 
                            p2_strat[child_seq_idx] = p2_regret[child_seq_idx] / total_regret
        for idx, seq in enumerate(treeplex1.sequences):
            seq.value = p1_strat[idx] 
        for idx, seq in enumerate(treeplex2.sequences):
            seq.value = p2_strat[idx] 
        treeplex1.convert_to_sequence_form()
        treeplex2.convert_to_sequence_form()
        p1_avg += p1_strat
        p2_avg += p2_strat
    p1_avg /= iterations
    p2_avg /= iterations
    return p1_avg, p2_avg

print(calc_p1_best_response(player1_treeplex, player2_treeplex))


x, y = counterfactual_regret_minimizer(player1_treeplex, player2_treeplex, 100)
print(np.round(x, 2))
print(np.round(y, 2))


# not getting -1/18 (yet)
print(calc_utility(player1_treeplex, player2_treeplex))


'''
Debugging steps
1. Check that leaf payoffs are correct (done)
2. 
'''

# col generation


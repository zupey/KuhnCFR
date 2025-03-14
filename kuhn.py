from enum import Enum
import typing
from typing import Optional, Dict, List, Tuple
from itertools import permutations
from collections import deque, defaultdict
import random
from dataclasses import dataclass
from fractions import Fraction
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
    def __init__(self, sequence: List[Action] = []):
        self.sequence: List[Action] = sequence

    def add_parent(self, infoset: Infoset):
        self.infoset = infoset

    def __repr__(self):
        return "(" + ", ".join([str(action) for action in self.sequence]) + ")"

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

    def find_infoset(self, card, history) -> Optional[Infoset]:
        for infoset in self.infosets:
            if infoset.card == card and infoset.history == history:
                return infoset
        return None

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
print(player1_treeplex.sequences)
# print(player2_treeplex.infosets)
print(player2_treeplex.sequences)
# print(leaves)

payoff_matrix: List[List[float]] = [[0. for _ in range(len(player2_treeplex.sequences))] for _ in range(len(player1_treeplex.sequences))]
print(len(leaves))
for leaf in leaves:
    seq1 = player1_treeplex.sequences[leaf.seq1_id]
    seq2 = player2_treeplex.sequences[leaf.seq2_id]
    # print(seq1, seq1.infoset.card, seq2, seq2.infoset.card, leaf.value)
    payoff_matrix[leaf.seq1_id][leaf.seq2_id] += leaf.value / 6


print(np.array(payoff_matrix))


"""
TODO
1. Initialize strategy for each seq
2. Enforce Checks (strategy between 0 and 1 + sum to parent)
3. Calculate xPy (utility)
4. Calculate best reponse by backward induction
5.++ CFR
"""
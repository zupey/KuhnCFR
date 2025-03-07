from enum import Enum
import typing
from typing import Optional, Dict, List, Tuple
from itertools import permutations

starting_states = list(permutations("JQK", 2))

class Action(Enum):
    BET = "BET"
    CALL = "CALL"
    CHECK = "CHECK"
    FOLD = "FOLD"


class Player(Enum):
    RANDOM = -1
    PLAYER_1 = 0
    PLAYER_2 = 1

    def other(self) -> "Player":
        if self == Player.RANDOM:
            return Player.RANDOM
        return Player.PLAYER_1 if self == Player.PLAYER_2 else Player.PLAYER_2

class Cards(Enum):
    Jack = "J"
    Queen = "Q"
    King = "K"


class Node:
    def __init__(self, parent: Optional["Node"], player: Player, actions: List[Action]):
        self.parent = parent
        self.player = player
        self.actions = actions
        self.children: Dict[Action, "Node"] = {}

    def play(self, action):
        return self.children[action]


class ChanceNode(Node):
    def __init__(self):
        super().__init__(parent=None, player=Player.RANDOM, actions=starting_states)
        self.children = {
            cards: PlayerNode(self, Player.PLAYER_1, [Action.BET, Action.CHECK], cards, []) for cards in self.actions
        }

    def __str__(self):
        return f"""
    ChanceNode
    {[(key, str(value)) for key, value in self.children.items()]}
    """


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
                Player.other(player),
                self.next_actions(action),
                self.cards,
                self.history + [action],
            )
            for action in actions
        }
    
        self.information_set = f"{self.card}, {self.history}"

    def next_actions(self, action: Action):
        if len(self.history) == 0 and action == Action.BET:
            return [Action.FOLD, Action.CALL]
        elif len(self.history) == 0 and action == Action.CHECK:
            return [Action.BET, Action.CHECK]
        elif self.history[-1] == Action.CHECK and action == Action.BET:
            return [Action.CALL, Action.FOLD]
        elif action == Action.CALL or action == Action.FOLD or (self.history[-1] == Action.CHECK and action == Action.CHECK):
            return []

    def __str__(self):
        return f"""
    PlayerNode 
    {[child for child in self.children]}
    """


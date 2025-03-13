from enum import Enum
import typing
from typing import Optional, Dict, List, Tuple
from itertools import permutations
from collections import deque, defaultdict
import random

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
        self.information_set: Optional[Tuple[Cards, List[Action]]]= None

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
        self.information_set : Tuple[Cards, List[Action]] = (self.card, self.history)

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

        if (self.history[-2] == Action.CHECK and self.history[-1] == Action.CHECK) or (self.history[-2] == Action.BET and self.history[-1] == Action.FOLD):
            return base

        if self.history[-2] == Action.BET and self.history[-1] == Action.CALL:
            return base * 2

    def __repr__(self):
        return f"""PlayerNode: History:{self.history}, {self.player}, Cards:{''.join(self.cards)}, Value:{self.evaluate()}"""

    def __str__(self):
        return f"""PlayerNode: History:{self.history}, {self.player}, Cards:{''.join(self.cards)}, Value:{self.evaluate()}"""


### Form Treeplex

def dfs(root):
    """Prints Nodes in DFS Order

    Args:
        root (Node): Root of Game Tree
    """
    stack = [root]
    while stack:
        node = stack.pop()
        print(node)
        for child_node in node.children:
            stack.append(child_node)

def generate_leaves(root: Node) -> List[PlayerNode]:
    stack: List[Node] = [root]
    res: List[PlayerNode] = []
    while stack:
        node = stack.pop()
        if node.is_terminal_state():
            res.append(node) # type: ignore
        for child_node in node.children.values():
            stack.append(child_node)
    return res

class TreeplexNode:
    def __init__(self, card: Cards, parent: Optional['TreeplexNode'], history:List[Action]):
        self.card: Cards = card
        self.history: List[Action] = history
        self.infoset = f"{card}, {history}"
        self.parent: Optional[TreeplexNode] = parent
        self.children: List[TreeplexNode]= []
        self.actions: List[Action] = self.get_actions()
        # initialize uniform strategy (behavioural form)
        if self.actions:
            x = random.randint(1, 10)
            self.behavioural_strategy: Dict[Action, float] = {
                self.actions[0] : 0.7,
                self.actions[1] : 0.3
                # action: 1/len(self.actions) for action in self.actions
            }
        self.sequence_form_strategy: Dict[Action, float] = {}
        for action in self.actions:
            if self.parent and self.parent.actions:
                self.sequence_form_strategy[action] = (1/len(self.actions)) * self.parent.behavioural_strategy[self.history[-2]]
            else:
                self.sequence_form_strategy[action] = self.behavioural_strategy[action]
                

    def is_terminal_state(self):
        return self.children == []

    def __str__(self):
        if self.parent is None:
            return "ROOT"
        return f"(<{self.infoset}>, strat: {self.sequence_form_strategy})"

    def __repr__(self):
        return f"(<{self.infoset}>, strat: {self.sequence_form_strategy})"

    def get_actions(self) -> List[Action]:
        """Get Actions at this node

        Returns:
            List[Action]: List of possible actions at this decision node (alphabetical order)
        """
        # history = [chance], P1 turn
        if len(self.history) == 1:
            return [Action.BET, Action.CHECK]
        # history = [chance, check/bet] P2 turn
        if len(self.history) == 2:
            if self.history[-1] == Action.CHECK:
                return [Action.BET, Action.CHECK]
            elif self.history[-1] == Action.BET:
                return [Action.CALL, Action.FOLD]
        # history = [chance, check, bet] P1 turn
        if len(self.history) == 3:
            return [Action.CALL, Action.FOLD]
        # catchall at chance node
        return []

    def display_tree(self) -> None:
        "Pretty printing of tree"
        levels = defaultdict(list)
        queue = deque([self])
        while queue:
            node = queue.popleft()
            levels[len(node.history)].append(node)
            for child in node.children:
                queue.append(child)
        keys = levels.keys()
        for key in sorted(keys):
            print(*levels[key])
    
def hash(card, history):
    return f"{card}, {history}"

node_map: Dict[str, TreeplexNode] = {}

def generate_treeplex(player: Player, root: Node):
    stack : List[Node] = [root]
    treeplex_root = TreeplexNode(Cards.Blank, None, [])
    map = {"Blank" : treeplex_root}
    while stack:
        node = stack.pop()
        if node.player == player and not node.is_terminal_state():
            if node.information_set is None:
                continue
            card, history = node.information_set
            s = hash(card, history)
            if s not in map:
                parent = "Blank" if len(history) <= 2 else hash(card, history[:-2])
                treeplex_node = TreeplexNode(card, map[parent], history)
                node_map[treeplex_node.infoset] = treeplex_node
                map[parent].children.append(treeplex_node)
                map[s] = treeplex_node
        for child_node in node.children.values():
            stack.append(child_node)
    return treeplex_root

root = ChanceNode()
# dfs(root)
tree1 = generate_treeplex(Player.PLAYER_1, root)
tree2 = generate_treeplex(Player.PLAYER_2, root)
# tree1.display_tree()
# tree2.display_tree()

leaves: List[PlayerNode] = generate_leaves(root)

# for key,val in node_map.items():
#     print(key,val)

# Currently does not take into account chance nodes
def calc_utility_game():
    expected_val = 0
    for leaf in leaves:
        if leaf.player == Player.PLAYER_1:
            val = leaf.evaluate() * node_map[hash(leaf.cards[1], leaf.history[:-1])].sequence_form_strategy[leaf.history[-1]] * node_map[hash(leaf.cards[0], leaf.history[:-2])].sequence_form_strategy[leaf.history[-2]]
        elif leaf.player == Player.PLAYER_2:
            val = leaf.evaluate() * node_map[hash(leaf.cards[0], leaf.history[:-1])].sequence_form_strategy[leaf.history[-1]] * node_map[hash(leaf.cards[1], leaf.history[:-2])].sequence_form_strategy[leaf.history[-2]]
        print(leaf, val)
        expected_val += val
    return expected_val

def best_response():
    pass

print(calc_utility_game())
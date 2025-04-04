from enum import Enum
from itertools import permutations
from collections import defaultdict
import numpy as np

class Action(Enum):
    BET = "BET"
    CALL = "CALL"
    CHECK = "CHECK"
    FOLD = "FOLD"
    CHANCE = "CHANCE"

class Player(Enum):
    PLAYER_1 = 0
    PLAYER_2 = 1

    def other(self):
        return Player.PLAYER_1 if self == Player.PLAYER_2 else Player.PLAYER_2

# Generate all possible card combinations (6 possible deals)
starting_states = list(permutations("JQK", 2))

class GameNode:
    def __init__(self, parent, player, cards, history):
        self.parent = parent
        self.player = player
        self.cards = cards
        self.history = history
        self.children = {}
        self.terminal = False
        self.value = None
        
    def is_terminal(self):
        return self.terminal
        
    def evaluate(self):
        if not self.is_terminal():
            return None
            
        # Determine winner based on cards and history
        card1, card2 = self.cards
        if (card1 == 'K' and card2 == 'J') or (card1 == 'Q' and card2 == 'J') or (card1 == 'K' and card2 == 'Q'):
            base = 1
        elif (card1 == 'J' and card2 == 'K') or (card1 == 'J' and card2 == 'Q') or (card1 == 'Q' and card2 == 'K'):
            base = -1
        else:
            base = 0  # Shouldn't happen in Kuhn Poker
            
        # Check terminal conditions
        if len(self.history) >= 2:
            if self.history[-2] == Action.CHECK and self.history[-1] == Action.CHECK:
                return base
            elif self.history[-2] == Action.BET and self.history[-1] == Action.CALL:
                return base * 2
            elif self.history[-1] == Action.FOLD:
                return 1 if self.player == Player.PLAYER_2 else -1
        return 0

def build_game_tree():
    root_nodes = []
    
    for cards in starting_states:
        root = GameNode(None, Player.PLAYER_1, cards, [Action.CHANCE])
        root_nodes.append(root)
        
        # Player 1's first move
        p1_check = GameNode(root, Player.PLAYER_2, cards, root.history + [Action.CHECK])
        p1_bet = GameNode(root, Player.PLAYER_2, cards, root.history + [Action.BET])
        root.children = {Action.CHECK: p1_check, Action.BET: p1_bet}
        
        # Player 2's responses to check
        p2_bet = GameNode(p1_check, Player.PLAYER_1, cards, p1_check.history + [Action.BET])
        p2_check = GameNode(p1_check, Player.PLAYER_1, cards, p1_check.history + [Action.CHECK])
        p2_check.terminal = True
        p2_check.value = p2_check.evaluate()
        p1_check.children = {Action.BET: p2_bet, Action.CHECK: p2_check}
        
        # Player 1's responses to bet after check
        p1_call = GameNode(p2_bet, Player.PLAYER_2, cards, p2_bet.history + [Action.CALL])
        p1_fold = GameNode(p2_bet, Player.PLAYER_2, cards, p2_bet.history + [Action.FOLD])
        p1_call.terminal = True
        p1_fold.terminal = True
        p1_call.value = p1_call.evaluate()
        p1_fold.value = p1_fold.evaluate()
        p2_bet.children = {Action.CALL: p1_call, Action.FOLD: p1_fold}
        
        # Player 2's responses to initial bet
        p2_call = GameNode(p1_bet, Player.PLAYER_1, cards, p1_bet.history + [Action.CALL])
        p2_fold = GameNode(p1_bet, Player.PLAYER_1, cards, p1_bet.history + [Action.FOLD])
        p2_call.terminal = True
        p2_fold.terminal = True
        p2_call.value = p2_call.evaluate()
        p2_fold.value = p2_fold.evaluate()
        p1_bet.children = {Action.CALL: p2_call, Action.FOLD: p2_fold}
    
    return root_nodes


def build_sequence_form(game_trees):
    # Initialize data structures
    p1_sequences = []
    p2_sequences = []
    
    # Maps for sequence IDs and infoset IDs
    sequence_to_id = {}
    infoset_to_id = {}
    current_seq_id = 0
    current_infoset_id = 0
    
    # Parent and child relationships
    p1_parent = []
    p1_children = defaultdict(list)
    p1_infoset_parent = []
    p1_infoset_children = defaultdict(list)
    
    p2_parent = []
    p2_children = defaultdict(list)
    p2_infoset_parent = []
    p2_infoset_children = defaultdict(list)
    
    # First pass: assign IDs to all sequences and infosets
    for tree in game_trees:
        queue = [(tree, -1, Player.PLAYER_1)]  # (node, parent_seq_id, current_player)
        
        while queue:
            node, parent_seq_id, player = queue.pop(0)
            
            # Assign sequence ID to node
            node.seq_id = current_seq_id
            current_seq_id += 1
            
            if player == Player.PLAYER_1:
                p1_parent.append(parent_seq_id)
                if parent_seq_id != -1:
                    p1_children[parent_seq_id].append(node.seq_id)
            else:
                p2_parent.append(parent_seq_id)
                if parent_seq_id != -1:
                    p2_children[parent_seq_id].append(node.seq_id)
            
            # Create infoset if needed
            infoset_key = (node.cards[player.value], tuple(node.history))
            if infoset_key not in infoset_to_id:
                infoset_id = current_infoset_id
                current_infoset_id += 1
                infoset_to_id[infoset_key] = infoset_id
                
                if player == Player.PLAYER_1:
                    p1_infoset_parent.append(node.seq_id)
                else:
                    p2_infoset_parent.append(node.seq_id)
            else:
                infoset_id = infoset_to_id[infoset_key]
            
            # Add to infoset children
            if player == Player.PLAYER_1:
                p1_infoset_children[infoset_id].append(node.seq_id)
            else:
                p2_infoset_children[infoset_id].append(node.seq_id)
            
            # Enqueue children
            for action, child in node.children.items():
                queue.append((child, node.seq_id, player.other()))
    
    # Count sequences for each player
    num_p1_sequences = len(p1_parent)
    num_p2_sequences = len(p2_parent)
    
    # Build payoff matrix
    payoff = np.zeros((num_p1_sequences, num_p2_sequences))
    
    # Second pass: fill payoff matrix
    for tree in game_trees:
        terminal_nodes = []
        queue = [tree]
        
        while queue:
            node = queue.pop(0)
            if node.is_terminal():
                terminal_nodes.append(node)
            else:
                queue.extend(node.children.values())
        
        for terminal in terminal_nodes:
            # Trace back to find sequences
            p1_seq = -1
            p2_seq = -1
            current = terminal
            
            while current.parent is not None:
                if current.player == Player.PLAYER_1:
                    p1_seq = current.parent.seq_id
                else:
                    p2_seq = current.parent.seq_id
                current = current.parent
            
            # Verify sequence IDs are within bounds
            if p1_seq != -1 and p2_seq != -1:
                if p1_seq < num_p1_sequences and p2_seq < num_p2_sequences:
                    payoff[p1_seq][p2_seq] += terminal.value / len(game_trees)
                else:
                    print(f"Warning: Sequence ID out of bounds (P1:{p1_seq}, P2:{p2_seq})")
    
    return {
        'p1': {
            'payoff': payoff,
            'strategy_parent': p1_parent,
            'strategy_children': p1_children,
            'infoset_parent': p1_infoset_parent,
            'infoset_children': p1_infoset_children,
            'num_sequences': num_p1_sequences
        },
        'p2': {
            'payoff': -payoff.T,
            'strategy_parent': p2_parent,
            'strategy_children': p2_children,
            'infoset_parent': p2_infoset_parent,
            'infoset_children': p2_infoset_children,
            'num_sequences': num_p2_sequences
        }
    }

class KuhnPokerTreeplex:
    def __init__(self, payoff, strategy_parent, strategy_children, infoset_parent, infoset_children):
        self.NUM_SEQUENCES = payoff.shape[0]
        self.NUM_INFOSETS = len(infoset_parent)
        
        # Strategy and regret tracking
        self.strategy = np.ones(self.NUM_SEQUENCES)  # Initial uniform strategy
        self.strategy_sum = np.zeros(self.NUM_SEQUENCES)
        self.regret_sum = np.zeros(self.NUM_SEQUENCES)
        self.utility = np.zeros(self.NUM_SEQUENCES)
        
        # Game structure
        self.strategy_parent = strategy_parent  # sequence -> parent sequence
        self.strategy_children = strategy_children  # sequence -> list of child infosets
        self.infoset_parent = infoset_parent  # infoset -> parent sequence
        self.infoset_children = infoset_children  # infoset -> list of child sequences
        self.payoff = payoff
        
        # Validate structure
        assert len(strategy_parent) == self.NUM_SEQUENCES
        assert len(infoset_parent) == self.NUM_INFOSETS
        assert len(infoset_children) == self.NUM_INFOSETS

    def get_strategy(self):
        """Update strategy based on current regrets"""
        for infoset_idx in range(self.NUM_INFOSETS):
            children = self.infoset_children[infoset_idx]
            total_regret = sum(max(self.regret_sum[c], 0) for c in children)
            
            if total_regret > 0:
                for c in children:
                    self.strategy[c] = max(self.regret_sum[c], 0) / total_regret
            else:
                # Uniform if no positive regret
                uniform_prob = 1.0 / len(children)
                for c in children:
                    self.strategy[c] = uniform_prob
                    
        self.normalize_strategy()
        return self.strategy

    def normalize_strategy(self):
        """Convert strategy to sequence form"""
        # Process in topological order (root first)
        processed = [False] * self.NUM_SEQUENCES
        processed[0] = True  # Root sequence
        
        for seq in range(1, self.NUM_SEQUENCES):
            parent = self.strategy_parent[seq]
            if not processed[parent]:
                raise ValueError("Invalid sequence ordering - parent not processed before child")
            
            # Multiply by parent's probability
            self.strategy[seq] *= self.strategy[parent]
            processed[seq] = True

    def update_regrets(self, opponent_strategy):
        """Update regrets against opponent's strategy"""
        # Calculate leaf utilities
        self.utility = self.payoff @ opponent_strategy
        
        # Process infosets in reverse order (leaves first)
        for infoset_idx in reversed(range(self.NUM_INFOSETS)):
            parent_seq = self.infoset_parent[infoset_idx]
            children = self.infoset_children[infoset_idx]
            
            # Calculate counterfactual utility for this infoset
            cf_utility = 0.0
            for c in children:
                cf_utility += self.strategy[c] * self.utility[c]
            
            # Update parent's utility
            self.utility[parent_seq] += cf_utility
            
            # Update regrets
            for c in children:
                # Counterfactual regret = (child utility - infoset utility) * reach probability
                self.regret_sum[c] += (self.utility[c] - cf_utility) * self.strategy[parent_seq]
        
        # Update strategy for next iteration
        self.get_strategy()
        
    def accumulate_strategy(self):
        """Accumulate strategy sums for average strategy"""
        self.strategy_sum += self.strategy
        
    def get_average_strategy(self):
        """Normalize accumulated strategy"""
        avg_strategy = np.zeros_like(self.strategy_sum)
        
        # Need to normalize within each infoset
        for infoset_idx in range(self.NUM_INFOSETS):
            children = self.infoset_children[infoset_idx]
            total = sum(self.strategy_sum[c] for c in children)
            
            if total > 0:
                for c in children:
                    avg_strategy[c] = self.strategy_sum[c] / total
            else:
                uniform = 1.0 / len(children)
                for c in children:
                    avg_strategy[c] = uniform
                    
        return avg_strategy

def train_kuhn_poker(iterations=1000):
    # Build the game tree
    game_trees = build_game_tree()
    
    # Convert to sequence form
    structure = build_sequence_form(game_trees)
    
    # Initialize both players
    player1 = KuhnPokerTreeplex(
        payoff=structure['p1']['payoff'],
        strategy_parent=structure['p1']['strategy_parent'],
        strategy_children=structure['p1']['strategy_children'],
        infoset_parent=structure['p1']['infoset_parent'],
        infoset_children=structure['p1']['infoset_children']
    )
    
    player2 = KuhnPokerTreeplex(
        payoff=structure['p2']['payoff'],
        strategy_parent=structure['p2']['strategy_parent'],
        strategy_children=structure['p2']['strategy_children'],
        infoset_parent=structure['p2']['infoset_parent'],
        infoset_children=structure['p2']['infoset_children']
    )
    
    # Run CFR
    for i in range(iterations):
        player1.update_regrets(player2.strategy)
        player2.update_regrets(player1.strategy)
        
        player1.accumulate_strategy()
        player2.accumulate_strategy()
    
    # Get average strategies
    p1_avg = player1.get_average_strategy()
    p2_avg = player2.get_average_strategy()
    
    return p1_avg, p2_avg, structure

def analyze_strategy(avg_strategy, structure, player_num):
    print(f"\nPlayer {player_num} strategy:")
    for infoset_id, children in structure[f'p{player_num}']['infoset_children'].items():
        parent_seq = structure[f'p{player_num}']['infoset_parent'][infoset_id]
        history = None
        
        # Find the history for this infoset (would need to track this in build_sequence_form)
        # For now just print the sequence IDs and probabilities
        print(f"Infoset {infoset_id}:")
        for seq_id in children:
            prob = avg_strategy[seq_id]
            print(f"  Sequence {seq_id}: {prob:.4f}")

# Run everything
p1_avg, p2_avg, structure = train_kuhn_poker(iterations=10000)
analyze_strategy(p1_avg, structure, 1)
analyze_strategy(p2_avg, structure, 2)

# Calculate expected value
expected_value = p1_avg @ structure['p1']['payoff'] @ p2_avg
print(f"\nExpected value: {expected_value:.6f}")
import numpy as np
import torch
import math
import copy
from config import Config

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {} # {action_key: Node}
        self.visits = 0
        self.value_sum = 0
        self.prior = 0
        
    def is_expanded(self):
        return len(self.children) > 0

    def ucb(self):
        if self.visits == 0:
            return float('inf')
        q = self.value_sum / self.visits
        # Standard PUCT
        u = Config.C_PUCT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u

class MCTS:
    def __init__(self, model):
        self.model = model
        
    def search(self, game_state, game_engine, my_score, opp_score):
        root = Node(game_state)
        
        # We need a fresh engine copy for simulation
        sim_engine = copy.deepcopy(game_engine)
        
        # 1. Expand Root
        self._expand(root, sim_engine, my_score, opp_score)
        
        for _ in range(Config.NUM_SIMULATIONS):
            node = root
            sim_engine = copy.deepcopy(game_engine) # Reset engine for traversal
            
            # SELECT
            while node.is_expanded():
                action, node = self._select_child(node)
                # Apply move in sim
                sim_engine.step_atomic(action)
                # If dice exhausted, sim engine might need logic? 
                # For this atomic MCTS, we assume dice is fixed at root for the current turn.
            
            # EXPAND & EVALUATE
            v = self._expand(node, sim_engine, my_score, opp_score)
            
            # BACKPROP
            self._backprop(node, v)
            
        return root

    def _select_child(self, node):
        best_score = -float('inf')
        best_act = None
        best_child = None
        
        for act, child in node.children.items():
            score = child.ucb()
            if score > best_score:
                best_score = score
                best_act = act
                best_child = child
        return best_act, best_child

    def _expand(self, node, engine, ms, os):
        # Check terminal
        winner, win_type = engine.check_win()
        if winner != 0:
            # Calculate reward from current player's perspective
            points = 1
            if win_type == 2: points = Config.R_GAMMON
            if win_type == 3: points = Config.R_BACKGAMMON
            points *= engine.cube
            
            # Return positive if current player won, negative otherwise
            if winner == engine.turn:
                return points
            else:
                return -points

        # Get Legal Atomic Moves (in real coordinates)
        moves = engine.get_legal_moves()
        
        if not moves:
            return 0  # No moves possible
            
        # Get canonical board representation (always from current player's view)
        board_vec, ctx_vec = engine.get_vector(ms, os, canonical=True)
        
        b_t = torch.from_numpy(board_vec).unsqueeze(0).long().to(Config.DEVICE)
        c_t = torch.from_numpy(ctx_vec).unsqueeze(0).float().to(Config.DEVICE)
        
        with torch.no_grad():
            p_from, p_to, v, _ = self.model(b_t, c_t)
            
        value = v.item()
        
        # Convert logits to priors
        p_from = torch.softmax(p_from, dim=1).cpu().numpy()[0]
        p_to = torch.softmax(p_to, dim=1).cpu().numpy()[0]
        
        policy_sum = 0
        for real_move in moves:
            # Convert real move to canonical for policy lookup
            canonical_move = engine.real_action_to_canonical(real_move)
            s, e = canonical_move
            
            # Map 'bar' and 'off' to indices
            s_idx = Config.BAR_IDX if s == 'bar' else s
            e_idx = Config.OFF_IDX if e == 'off' else e
            
            prob = p_from[s_idx] * p_to[e_idx]
            
            # Store with REAL move as key (for step_atomic)
            node.children[real_move] = Node(None, parent=node)
            node.children[real_move].prior = prob
            policy_sum += prob
            
        # Normalize priors
        for child in node.children.values():
            if policy_sum > 0:
                child.prior /= policy_sum
            else:
                child.prior = 1.0 / len(moves)
                
        return value

    def _backprop(self, node, value):
        while node:
            node.visits += 1
            node.value_sum += value
            value = -value # Flip for opponent
            node = node.parent
import torch
import math
from config import Config

class Node:
    __slots__ = ['visits', 'value_sum', 'prior', 'children']
    def __init__(self):
        self.children = {} 
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
        
    def is_expanded(self):
        return len(self.children) > 0

    def ucb(self, sqrt_parent_visits):
        # Optimized UCB calculation (passed pre-calc sqrt)
        if self.visits == 0:
            return float('inf')
        
        q = self.value_sum / self.visits
        u = Config.C_PUCT * self.prior * sqrt_parent_visits / (1 + self.visits)
        return q + u

class MCTS:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def search(self, game_state, game_engine, my_score, opp_score):
        root = Node()
        
        # Save original state once
        original_snapshot = game_engine.fast_save()
        
        # 1. Expand Root
        self._expand(root, game_engine, my_score, opp_score)
        
        for _ in range(Config.NUM_SIMULATIONS):
            node = root
            game_engine.fast_restore(original_snapshot)
            
            # SELECT
            path = []
            while node.is_expanded():
                sqrt_v = math.sqrt(node.visits)
                
                # Fast Python max logic
                best_score = -float('inf')
                best_act = None
                best_child = None
                
                for act, child in node.children.items():
                    score = child.ucb(sqrt_v)
                    if score > best_score:
                        best_score = score
                        best_act = act
                        best_child = child
                
                if best_child is None: break 
                
                game_engine.step_atomic(best_act)
                node = best_child
                path.append(node)
            
            # EXPAND & EVALUATE
            v = self._expand(node, game_engine, my_score, opp_score)
            
            # BACKPROP
            # Backpropagate value v up the path
            for p_node in path:
                p_node.visits += 1
                p_node.value_sum += v
                v = -v
            
            # Root update
            root.visits += 1
            root.value_sum += v
        
        # Restore engine to actual state before returning
        game_engine.fast_restore(original_snapshot)
        return root

    def _expand(self, node, engine, ms, os):
        # Check terminal state
        winner, win_type = engine.check_win()
        if winner != 0:
            points = 1.0
            if win_type == 2: points = Config.R_GAMMON
            if win_type == 3: points = Config.R_BACKGAMMON
            points *= engine.cube
            
            if winner == engine.turn: return points
            else: return -points

        moves = engine.get_legal_moves()
        if not moves:
            return 0.0
            
        # Get tensors directly on device (No Numpy)
        b_t, c_t = engine.get_vector(ms, os, device=self.device, canonical=True)
        
        # Add batch dim
        b_t = b_t.unsqueeze(0)
        c_t = c_t.unsqueeze(0)
        
        with torch.no_grad():
            p_from, p_to, v, _ = self.model(b_t, c_t)
            
        value = v.item()
        
        # Softmax on device
        p_from = torch.softmax(p_from, dim=1).squeeze(0) # [26]
        p_to = torch.softmax(p_to, dim=1).squeeze(0)     # [26]
        
        policy_sum = 0.0
        
        # Map legal moves to children
        for real_move in moves:
            canonical_move = engine.real_action_to_canonical(real_move)
            s, e = canonical_move
            s_idx = Config.BAR_IDX if s == 'bar' else s
            e_idx = Config.OFF_IDX if e == 'off' else e
            
            # Tensor indexing (scalar access)
            prob = (p_from[s_idx] * p_to[e_idx]).item()
            
            child = Node()
            child.prior = prob
            node.children[real_move] = child
            policy_sum += prob
            
        # Normalize priors
        if policy_sum > 0:
            inv_sum = 1.0 / policy_sum
            for child in node.children.values():
                child.prior *= inv_sum
        else:
            # Fallback uniform
            prob = 1.0 / len(moves)
            for child in node.children.values():
                child.prior = prob
                
        return value
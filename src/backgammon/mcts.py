import math
import torch
from src.backgammon.config import Config

class MCTSNode:
    __slots__ = ("parent", "action", "children", "visits", "value_sum", "prior")
    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action  # Now stores ((start, end), die)
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

class MCTS:
    def __init__(self, model, cpuct=1.5, num_sims=50, device="cpu", batch_size=16):
        self.model = model
        self.device = device
        self.cpuct = float(cpuct)
        self.num_sims = int(num_sims)
        self.batch_size = int(batch_size)
        self.root = MCTSNode()

    def reset(self):
        """Hard reset of the search tree for a new game."""
        self.root = MCTSNode()

    def advance_to_child(self, action_with_die):
        """
        Reuses the existing subtree for the chosen action. 
        """
        for child in self.root.children:
            if child.action == action_with_die:
                child.parent = None
                self.root = child
                return
        self.reset()

    def search(self, game, my_score, opp_score, reset_tree=True):
            if reset_tree:
                self.root = MCTSNode() # Force a fresh start for this specific dice state
            
            root_snapshot = game.fast_save()
            eval_queue = []

            for _ in range(self.num_sims):
                node = self.root
                game.fast_restore(root_snapshot)

                # --- 1. SELECTION ---
                # Follow the tree until we hit a leaf or the turn ends (no dice left)
                while node.children and game.dice:
                    sqrt_n = math.sqrt(node.visits + 1e-6)
                    best_score = -1e9
                    best_child = None
                    
                    for child in node.children:
                        # UCB1 Calculation
                        q = child.value_sum / (child.visits + 1e-6)
                        u = self.cpuct * child.prior * sqrt_n / (1 + child.visits)
                        score = q + u
                        if score > best_score:
                            best_score = score
                            best_child = child
                    
                    if not best_child: break
                    
                    # Apply the move to the temporary game state
                    game.step_atomic(best_child.action)
                    node = best_child

                # --- 2. EXPANSION ---
                # If we reached a leaf and the game/turn isn't over, expand it
                winner, _ = game.check_win()
                if not node.children and winner == 0 and game.dice:
                    # CRITICAL: get_legal_moves now accurately reflects 
                    # the remaining dice in the 'game' object
                    moves = game.get_legal_moves()
                    if moves:
                        prior = 1.0 / len(moves)
                        node.children = [MCTSNode(node, m, prior) for m in moves]

                # --- 3. EVALUATION ---
                # Obtain value for the resulting state
                b_t, c_t = game.get_vector(my_score, opp_score, self.device)
                eval_queue.append((node, b_t, c_t))

                if len(eval_queue) >= self.batch_size:
                    self._flush_evals(eval_queue)
                    eval_queue = []

            if eval_queue: 
                self._flush_evals(eval_queue)
                
            game.fast_restore(root_snapshot)
            return self.root

    def _flush_evals(self, queue):
        boards = torch.stack([x[1] for x in queue])
        ctxs = torch.stack([x[2] for x in queue])
        with torch.no_grad():
            out = self.model(boards, ctxs)
            values = out[2].view(-1).tolist() 
        
        for i, (node, _, _) in enumerate(queue):
            self._backprop(node, values[i])

    def _backprop(self, node, v):
        while node:
            node.visits += 1
            node.value_sum += v
            v = -v
            node = node.parent

    def get_visit_targets(self, action_space_size=26):
            target_f = torch.zeros(action_space_size)
            target_t = torch.zeros(action_space_size)
            
            total_visits = sum(child.visits for child in self.root.children)
            if total_visits == 0:
                return target_f, target_t

            for child in self.root.children:
                # UNPACK WRAPPER: ((src, dst), die)
                (src, dst), _ = child.action 
                
                # Map "bar" and "off" to indices
                idx_f = 24 if src == "bar" else src
                idx_t = 25 if dst == "off" else dst
                
                prob = child.visits / total_visits
                target_f[idx_f] += prob
                target_t[idx_t] += prob
                
            return target_f, target_t

    def _to_idx(self, val):
        """Safely maps board positions (ints or strings) to tensor indices."""
        if val == "bar": return 24
        if val == "off": return 25
        return int(val)
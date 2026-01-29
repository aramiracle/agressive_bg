import math
import torch
from src.backgammon.config import Config

class MCTSNode:
    __slots__ = ("parent", "action", "children", "visits", "value_sum", "prior")
    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
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

    def advance_to_child(self, action):
        """
        Reuses the existing subtree for the chosen action. 
        This is a major speed optimization.
        """
        for child in self.root.children:
            if child.action == action:
                child.parent = None
                self.root = child
                return
        self.reset()

    def search(self, game, my_score, opp_score):
        root_snapshot = game.fast_save()
        eval_queue = []

        # Optimization: cache variables locally for the hot loop
        num_sims = self.num_sims
        batch_size = self.batch_size
        cpuct = self.cpuct

        for _ in range(num_sims):
            node = self.root
            game.fast_restore(root_snapshot)

            # 1. Selection
            while node.children:
                sqrt_n = math.sqrt(node.visits + 1e-6)
                best_score = -1e9
                best_child = None
                
                for child in node.children:
                    q = child.value_sum / (child.visits + 1e-6)
                    u = cpuct * child.prior * sqrt_n / (1 + child.visits)
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_child = child
                
                if not best_child: break
                game.step_atomic(best_child.action)
                node = best_child
                if not game.dice: break

            # 2. Expansion
            winner, _ = game.check_win()
            if not node.children and winner == 0:
                moves = game.get_legal_moves()
                if moves:
                    prob = 1.0 / len(moves)
                    node.children = [MCTSNode(node, m, prob) for m in moves]

            # 3. Batched Inference Queue
            b_t, c_t = game.get_vector(my_score, opp_score, self.device)
            eval_queue.append((node, b_t, c_t))

            if len(eval_queue) >= batch_size:
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
        if not self.root.children:
            return target_f, target_t

        for child in self.root.children:
            f, t = child.action
            target_f[f] += child.visits
            target_t[t] += child.visits

        total_f = target_f.sum()
        if total_f > 0: target_f /= total_f
        total_t = target_t.sum()
        if total_t > 0: target_t /= total_t
        return target_f, target_t
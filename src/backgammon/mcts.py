import math
import torch
from src.backgammon.config import Config

class MCTSNode:
    __slots__ = ("parent", "action", "children", "visits", "value_sum", "prior")
    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action  # stores ((start, end), die)
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
    
    def value(self):
        """Returns averaged value (-1 to 1) seen by this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

class MCTS:
    def __init__(self, model, cpuct=1.5, num_sims=50, device="cpu", batch_size=16):
        self.model = model
        self.device = device
        self.cpuct = float(cpuct)
        self.num_sims = int(num_sims)
        self.batch_size = int(batch_size)
        self.root = MCTSNode()

    def reset(self):
        self.root = MCTSNode()

    def advance_to_child(self, action_with_die):
        for child in self.root.children:
            if child.action == action_with_die:
                child.parent = None
                self.root = child
                return
        self.reset()

    def search(self, game, my_score, opp_score, reset_tree=True):
        if reset_tree:
            self.root = MCTSNode()
            
        root_snapshot = game.fast_save()
        eval_queue = []

        for _ in range(self.num_sims):
            node = self.root
            game.fast_restore(root_snapshot)

            # 1. Selection
            while node.children and game.dice:
                sqrt_n = math.sqrt(node.visits + Config.MIN_PRIOR)
                best_score = -math.inf
                best_child = None
                
                for child in node.children:
                    q = child.value()
                    u = self.cpuct * child.prior * sqrt_n / (1 + child.visits)
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_child = child
                
                if not best_child: break
                game.step_atomic(best_child.action)
                node = best_child

            # 2. Expansion
            winner, _ = game.check_win()
            if not node.children and winner == 0 and game.dice:
                moves = game.get_legal_moves()
                if moves:
                    prior = 1.0 / len(moves)
                    node.children = [MCTSNode(node, m, prior) for m in moves]

            # 3. Evaluation
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
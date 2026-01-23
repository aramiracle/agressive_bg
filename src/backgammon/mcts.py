import math
import torch


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
    def __init__(self, model, cpuct=1.5, num_sims=100, device="cpu", batch_size=8):
        self.model = model
        self.cpuct = cpuct
        self.num_sims = num_sims
        self.device = device
        self.batch_size = batch_size
        self.root = MCTSNode()

    def reset(self):
        self.root = MCTSNode()

    def advance_to_child(self, action):
        for child in self.root.children:
            if child.action == action:
                child.parent = None
                self.root = child
                return
        self.reset()

    def search(self, game, my_score, opp_score):
        if not self.root.children:
            self._expand_node(self.root, game)

        root_snapshot = game.fast_save()
        eval_queue = []

        for _ in range(self.num_sims):
            node = self.root
            game.fast_restore(root_snapshot)

            # --- SELECTION PHASE ---
            while node.children:
                best = None
                best_score = -float('inf') # Use -inf, cleaner than -1e9
                sqrt_visits = math.sqrt(node.visits + 1)

                for child in node.children:
                    u = self.cpuct * child.prior * sqrt_visits / (1 + child.visits)
                    q = child.value_sum / child.visits if child.visits else 0.0
                    score = q + u

                    # Check for NaNs to prevent poisoning
                    if math.isnan(score): 
                        score = -float('inf')

                    if score > best_score:
                        best_score = score
                        best = child
                
                # SAFETY CHECK: If best is still None, break to avoid crash
                if best is None:
                    break

                # Handle Game Step
                try:
                    # Move atomic logic here to separate variable access from game logic
                    action = best.action 
                    game.step_atomic(action)
                    node = best
                except Exception:
                    # This catches illegal moves from game logic, not AttributeError
                    if best in node.children:
                        node.children.remove(best)
                    # If we pruned the only child, or selection failed, stop this sim
                    if not node.children: 
                        break
                    continue # Try selecting a different sibling

                if not game.dice:
                    break
            
            # ... (Rest of the loop: checking win, backprop, expansion) ... 

            winner, _ = game.check_win()
            if winner != 0:
                value = 1.0 if winner == game.turn else -1.0
                self._backpropagate(node, value)
                continue

            if not node.children:
                self._expand_node(node, game)

            b_t, c_t = game.get_vector(
                my_score=my_score,
                opp_score=opp_score,
                device=self.device,
                canonical=False,
            )

            eval_queue.append((node, b_t, c_t))

            if len(eval_queue) >= self.batch_size:
                self._flush_batch(eval_queue)
                eval_queue.clear()

        if eval_queue:
            self._flush_batch(eval_queue)

        game.fast_restore(root_snapshot)
        return self.root

    def _flush_batch(self, batch):
        boards = torch.stack([b for _, b, _ in batch]).to(self.device)
        ctxs = torch.stack([c for _, _, c in batch]).to(self.device)

        with torch.no_grad():
            out = self.model(boards, ctxs)
            values = out[2].squeeze(-1)

        for i, (node, _, _) in enumerate(batch):
            v = float(values[i].item())
            self._backpropagate(node, v)

    def _expand_node(self, node, game):
        legal = game.get_legal_moves()
        if not legal:
            return

        prior = 1.0 / len(legal)
        node.children = [MCTSNode(node, mv, prior) for mv in legal]

    def _backpropagate(self, node, value):
        v = value
        while node is not None:
            node.visits += 1
            node.value_sum += v
            v = -v
            node = node.parent

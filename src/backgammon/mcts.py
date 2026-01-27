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
                best_score = -float('inf')
                sqrt_visits = math.sqrt(node.visits + 1)

                for child in node.children:
                    # Force cpuct to float just in case
                    cpuct = float(self.cpuct)
                    
                    # Force prior to float (handle list/tensor/float inputs)
                    try:
                        if hasattr(child.prior, 'item'):
                            # Handle torch tensors or numpy scalars
                            prior_val = float(child.prior.item())
                        elif isinstance(child.prior, (list, tuple)):
                            # Handle accidental list wrapping
                            prior_val = float(child.prior[0])
                        else:
                            prior_val = float(child.prior)
                    except (TypeError, IndexError, ValueError):
                        # Fallback if data is corrupted
                        prior_val = 1.0

                    sqrt_visits = math.sqrt(node.visits + 1)
                    
                    # Calculate UCB using local safe variables
                    u = cpuct * prior_val * sqrt_visits / (1 + child.visits)
                    q = child.value_sum / child.visits if child.visits > 0 else 0.0
                    score = q + u

                    if math.isnan(score):
                        score = -float('inf')

                    if score > best_score:
                        best_score = score
                        best = child
                
                # If selection failed (should not happen if children exist), break
                if best is None:
                    break

                try:
                    game.step_atomic(best.action)
                    node = best
                except Exception:
                    # Handle rare cases where a cached move becomes invalid
                    if best in node.children:
                        node.children.remove(best)
                    if not node.children:
                        break
                    continue

                # Stop selection if turn ends (dice used up)
                if not game.dice:
                    break
            
            # --- EVALUATION / BACKPROP ---
            winner, _ = game.check_win()
            if winner != 0:
                # Value is +1 if 'node' player won (current turn), else -1
                # Note: This logic assumes standard zero-sum perspective relative to node.parent
                value = 1.0 if winner == game.turn else -1.0
                self._backpropagate(node, value)
                continue

            # Expansion
            if not node.children:
                self._expand_node(node, game)

            # If node is a leaf (just expanded or terminal), queue for evaluation
            # If expansion failed (no legal moves but no winner?), we treat as draw or loss depending on rules
            # Here we just evaluate the state value.
            b_t, c_t = game.get_vector(
                my_score=my_score,
                opp_score=opp_score,
                device=self.device,
                canonical=False,
            )

            eval_queue.append((node, b_t, c_t))

            # Batch Processing
            if len(eval_queue) >= self.batch_size:
                self._flush_batch(eval_queue)
                eval_queue.clear()

        # Flush remaining
        if eval_queue:
            self._flush_batch(eval_queue)

        game.fast_restore(root_snapshot)
        return self.root

    def _flush_batch(self, batch):
        if not batch:
            return

        boards = torch.stack([b for _, b, _ in batch]).to(self.device)
        ctxs = torch.stack([c for _, _, c in batch]).to(self.device)

        with torch.no_grad():
            # Standard model forward pass: returns (p_from, p_to, value, cube)
            out = self.model(boards, ctxs)
            values = out[2].squeeze(-1) # Value head is index 2

        for i, (node, _, _) in enumerate(batch):
            v = float(values[i].item())
            self._backpropagate(node, v)

    def _expand_node(self, node, game):
        legal = game.get_legal_moves()
        if not legal:
            return

        # Uniform prior if not using policy head for selection (simplification)
        # We ensure prior is strictly a float
        prior = float(1.0 / len(legal))
        
        # KEY FIX: Use keyword arguments to ensure 'action' gets 'mv' and 'prior' gets 'prior'
        node.children = [
            MCTSNode(parent=node, action=mv, prior=prior) 
            for mv in legal
        ]

    def _backpropagate(self, node, value):
        v = value
        while node is not None:
            node.visits += 1
            node.value_sum += v
            v = -v # Flip perspective for parent
            node = node.parent
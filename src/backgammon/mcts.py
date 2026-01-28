# =============================
# MINIMAL, PRECISE NaN-SAFE MODIFICATIONS
# Structure and behavior preserved
# =============================

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
        # --- SAFETY: ensure prior is always finite float ---
        try:
            self.prior = float(prior)
        except Exception:
            self.prior = 0.0


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

            # ---------------- SELECTION ----------------
            while node.children:
                best = None
                best_score = -float("inf")
                sqrt_visits = math.sqrt(node.visits + 1)

                for child in node.children:
                    # --- SAFETY: protect UCB math ---
                    prior = child.prior
                    if not math.isfinite(prior) or prior < 0.0:
                        prior = 0.0

                    u = self.cpuct * prior * sqrt_visits / (1 + child.visits)
                    q = child.value_sum / child.visits if child.visits > 0 else 0.0

                    score = q + u
                    if not math.isfinite(score):
                        score = -float("inf")

                    if score > best_score:
                        best_score = score
                        best = child

                if best is None:
                    break

                try:
                    game.step_atomic(best.action)
                    node = best
                except Exception:
                    # Invalid cached move: drop child and retry
                    if best in node.children:
                        node.children.remove(best)
                    if not node.children:
                        break
                    continue

                if not game.dice:
                    break

            # ---------------- TERMINAL CHECK ----------------
            winner, _ = game.check_win()
            if winner != 0:
                value = 1.0 if winner == game.turn else -1.0
                self._backpropagate(node, value)
                continue

            # ---------------- EXPANSION ----------------
            if not node.children:
                self._expand_node(node, game)

            # ---------------- EVALUATION ----------------
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
        if not batch:
            return

        boards = torch.stack([b for _, b, _ in batch]).to(self.device)
        ctxs = torch.stack([c for _, _, c in batch]).to(self.device)

        with torch.no_grad():
            out = self.model(boards, ctxs)
            values = out[2].squeeze(-1)

        for i, (node, _, _) in enumerate(batch):
            v = float(values[i].item())

            # --- SAFETY: clamp value head ---
            if not math.isfinite(v):
                v = 0.0
            v = max(-1.0, min(1.0, v))

            self._backpropagate(node, v)

    def _expand_node(self, node, game):
        legal = game.get_legal_moves()
        if not legal:
            return

        # Uniform prior (unchanged behavior)
        prior = 1.0 / len(legal)

        node.children = [
            MCTSNode(parent=node, action=mv, prior=prior)
            for mv in legal
        ]

    def _backpropagate(self, node, value):
        v = float(value)
        # --- SAFETY: sanitize backprop value ---
        if not math.isfinite(v):
            v = 0.0
        v = max(-1.0, min(1.0, v))

        while node is not None:
            node.visits += 1
            node.value_sum += v
            v = -v
            node = node.parent

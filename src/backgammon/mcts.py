# ============================================================
# MCTS — HARD DEBUG EDITION
# Will CRASH on first NaN or invalid policy
# ============================================================

import math
import torch
import traceback


# ------------------------------------------------------------
# NODE
# ------------------------------------------------------------
class MCTSNode:
    __slots__ = ("parent", "action", "children", "visits", "value_sum", "prior")

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value_sum = 0.0

        try:
            p = float(prior)
        except Exception:
            p = 0.0

        if not math.isfinite(p) or p < 0.0:
            p = 0.0

        self.prior = p


# ------------------------------------------------------------
# MCTS
# ------------------------------------------------------------
class MCTS:
    def __init__(self, model, cpuct=1.5, num_sims=100, device="cpu", batch_size=8):
        self.model = model
        self.cpuct = float(cpuct)
        self.num_sims = int(num_sims)
        self.device = device
        self.batch_size = int(batch_size)
        self.root = MCTSNode()

    # --------------------------------------------------------
    # DEBUG UTIL
    # --------------------------------------------------------
    def _die(self, msg):
        print("\n" + "=" * 60)
        print("🚨 MCTS HARD FAILURE")
        print(msg)
        print("TRACEBACK:")
        traceback.print_stack(limit=20)
        print("=" * 60)
        raise RuntimeError(msg)

    # --------------------------------------------------------
    # SAFE NORMALIZE (DEBUG)
    # --------------------------------------------------------
    def _safe_normalize(self, t, size, label):
        if torch.isnan(t).any() or torch.isinf(t).any():
            self._die(f"{label}: Input tensor already NaN/Inf:\n{t}")

        s = t.sum()

        if not torch.isfinite(s):
            self._die(f"{label}: Sum is NaN/Inf. Tensor:\n{t}")

        if s <= 0:
            print(f"⚠ {label}: ZERO SUM — FALLING BACK TO UNIFORM")
            return torch.ones(size, dtype=torch.float32) / float(size)

        p = t / s

        if torch.isnan(p).any() or torch.isinf(p).any():
            self._die(f"{label}: Normalization produced NaN:\n{p}")

        return p

    # --------------------------------------------------------
    def reset(self):
        self.root = MCTSNode()

    # --------------------------------------------------------
    def advance_to_child(self, action):
        for child in self.root.children:
            if child.action == action:
                child.parent = None
                self.root = child
                return
        self.reset()

    # --------------------------------------------------------
    def search(self, game, my_score, opp_score):
        if not self.root.children:
            self._expand_node(self.root, game)

        root_snapshot = game.fast_save()
        eval_queue = []

        for sim in range(self.num_sims):
            node = self.root
            game.fast_restore(root_snapshot)

            # ---------------- SELECTION ----------------
            while node.children:
                best = None
                best_score = -float("inf")
                sqrt_visits = math.sqrt(node.visits + 1.0)

                for child in node.children:
                    p = child.prior
                    if not math.isfinite(p) or p < 0:
                        self._die(f"Child prior corrupted: {p}")

                    u = self.cpuct * p * sqrt_visits / (1 + child.visits)

                    q = child.value_sum / child.visits if child.visits > 0 else 0.0

                    score = q + u

                    if not math.isfinite(score):
                        self._die(f"UCB score NaN: q={q}, u={u}, p={p}")

                    if score > best_score:
                        best_score = score
                        best = child

                if best is None:
                    break

                try:
                    game.step_atomic(best.action)
                    node = best
                except Exception:
                    if best in node.children:
                        node.children.remove(best)
                    if not node.children:
                        break
                    continue

                if not game.dice:
                    break

            # ---------------- TERMINAL ----------------
            winner, _ = game.check_win()
            if winner != 0:
                v = 1.0 if winner == game.turn else -1.0
                self._backpropagate(node, v)
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

            if torch.isnan(b_t).any() or torch.isnan(c_t).any():
                self._die("NaN in game.get_vector() output")

            eval_queue.append((node, b_t, c_t))

            if len(eval_queue) >= self.batch_size:
                self._flush_batch(eval_queue)
                eval_queue.clear()

        if eval_queue:
            self._flush_batch(eval_queue)

        game.fast_restore(root_snapshot)
        return self.root

    # --------------------------------------------------------
    def _flush_batch(self, batch):
        boards = torch.stack([b for _, b, _ in batch]).to(self.device)
        ctxs = torch.stack([c for _, _, c in batch]).to(self.device)

        if torch.isnan(boards).any():
            self._die("NaN in stacked boards")

        if torch.isnan(ctxs).any():
            self._die("NaN in stacked contexts")

        with torch.no_grad():
            out = self.model(boards, ctxs)
            values = out[2].squeeze(-1)

        for i, (node, _, _) in enumerate(batch):
            v = values[i].item()

            if not math.isfinite(v):
                self._die(f"Model returned NaN value head: {v}")

            v = max(-1.0, min(1.0, float(v)))
            self._backpropagate(node, v)

    # --------------------------------------------------------
    def _expand_node(self, node, game):
        try:
            legal = game.get_legal_moves()
        except Exception as e:
            self._die(f"get_legal_moves crashed: {e}")

        if not legal:
            return

        for mv in legal:
            if not isinstance(mv, (tuple, list)) or len(mv) != 2:
                self._die(f"Invalid move format: {mv}")

        n = len(legal)
        prior = 1.0 / float(n)

        node.children = [
            MCTSNode(parent=node, action=mv, prior=prior)
            for mv in legal
        ]

    # --------------------------------------------------------
    def _backpropagate(self, node, value):
        v = float(value)

        if not math.isfinite(v):
            self._die(f"Backprop value NaN: {v}")

        v = max(-1.0, min(1.0, v))

        while node is not None:
            if node.visits < 0:
                self._die("Node visits negative")

            node.visits += 1
            node.value_sum += v
            v = -v
            node = node.parent

    # --------------------------------------------------------
    # HARD POLICY TARGET
    # --------------------------------------------------------
    def get_visit_targets(self, action_space_size=26):
        if not self.root.children:
            self._die("get_visit_targets called with EMPTY ROOT")

        target_f = torch.zeros(action_space_size)
        target_t = torch.zeros(action_space_size)

        total_visits = 0

        for child in self.root.children:
            if child.visits < 0:
                self._die(f"Negative visits: {child.visits}")

            try:
                f, t = child.action
            except Exception:
                self._die(f"Invalid action in child: {child.action}")

            if not (0 <= f < action_space_size):
                self._die(f"f out of bounds: {f}")

            if not (0 <= t < action_space_size):
                self._die(f"t out of bounds: {t}")

            target_f[f] += child.visits
            target_t[t] += child.visits
            total_visits += child.visits

        print(f"DEBUG POLICY: total_visits={total_visits}")

        tf = self._safe_normalize(target_f, action_space_size, "FROM")
        tt = self._safe_normalize(target_t, action_space_size, "TO")

        return tf, tt

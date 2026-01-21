import math
import torch


class MCTSNode:
    __slots__ = (
        "parent",
        "action",
        "children",
        "visits",
        "value_sum",
        "prior",
    )

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action          # atomic move
        self.children = []            # list[MCTSNode]
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior


class MCTS:
    def __init__(self, model, cpuct=1.5, num_sims=100, device="cpu"):
        self.model = model
        self.cpuct = cpuct
        self.num_sims = num_sims
        self.device = device
        self.root = MCTSNode()

    # =====================================================
    # TREE MANAGEMENT
    # =====================================================

    def reset(self):
        self.root = MCTSNode()

    def advance_to_child(self, action):
        for child in self.root.children:
            if child.action == action:
                child.parent = None
                self.root = child
                return
        self.reset()

    # =====================================================
    # PUBLIC SEARCH
    # =====================================================

    def search(self, game, my_score, opp_score):
        if not self.root.children:
            self._expand_node(self.root, game)

        root_snapshot = game.fast_save()

        for _ in range(self.num_sims):
            node = self.root
            game.fast_restore(root_snapshot)

            # -----------------------
            # SELECTION
            # -----------------------
            while node.children:
                best = None
                best_score = -1e9
                sqrt_visits = math.sqrt(node.visits + 1)

                for child in node.children:
                    u = self.cpuct * child.prior * sqrt_visits / (1 + child.visits)
                    q = child.value_sum / child.visits if child.visits else 0.0
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best = child

                try:
                    game.step_atomic(best.action)
                except Exception:
                    node.children.remove(best)
                    break

                node = best
                if not game.dice:
                    break

            # -----------------------
            # TERMINAL CHECK
            # -----------------------
            winner, _ = game.check_win()
            if winner != 0:
                value = 1.0 if winner == game.turn else -1.0
                self._backpropagate(node, value)
                continue

            # -----------------------
            # EXPANSION
            # -----------------------
            if not node.children:
                self._expand_node(node, game)

            # -----------------------
            # EVALUATION
            # -----------------------
            value = self._evaluate(game, my_score, opp_score)
            self._backpropagate(node, value)

        game.fast_restore(root_snapshot)
        return self.root

    # =====================================================
    # CORE OPERATIONS
    # =====================================================

    def _expand_node(self, node, game):
        legal = game.get_legal_moves()
        if not legal:
            return

        prior = 1.0 / len(legal)
        children = []
        for mv in legal:
            children.append(MCTSNode(node, mv, prior))
        node.children = children

    def _evaluate(self, game, my_score, opp_score):
        with torch.no_grad():
            b_t, c_t = game.get_vector(
                my_score=my_score,
                opp_score=opp_score,
                device=self.device,
                canonical=False,
            )
            out = self.model(b_t.unsqueeze(0), c_t.unsqueeze(0))
            if isinstance(out, tuple):
                return float(out[2].item())
            return float(out.item())

    def _backpropagate(self, node, value):
        v = value
        while node is not None:
            node.visits += 1
            node.value_sum += v
            v = -v
            node = node.parent

    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def best_action(self, temperature=0.0):
        if not self.root.children:
            return None

        if temperature <= 0:
            best = None
            max_visits = -1
            for c in self.root.children:
                if c.visits > max_visits:
                    max_visits = c.visits
                    best = c
            return best.action

        visits = torch.tensor([c.visits for c in self.root.children], dtype=torch.float)
        probs = visits ** (1.0 / temperature)
        probs /= probs.sum()
        idx = torch.multinomial(probs, 1).item()
        return self.root.children[idx].action

    # =====================================================
    # SAFETY
    # =====================================================

    def sanity_check(self, game):
        legal = set(game.get_legal_moves())
        self.root.children = [c for c in self.root.children if c.action in legal]

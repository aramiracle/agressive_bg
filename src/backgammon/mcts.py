import math
import torch

"""
AlphaZero-style Monte Carlo Tree Search for Backgammon
------------------------------------------------------

Design goals:
- Works in REAL board space (no canonical confusion)
- Atomic-move aware (multiple moves per turn / dice consumption)
- Strictly legal: never calls step_atomic() with illegal actions
- Safe for Web / MCTS / NN integration
- Preserves tree across multi-atomic moves in same turn

Model Interface Expectations:
-----------------------------
Your model(model(board, ctx)) can return:
  1) value tensor
  OR
  2) tuple: (policy_from, policy_to, value, cube_logits)

This implementation only USES the value head.
Policy is handled by uniform priors over legal moves (robust + simple).

If you later want true policy guidance, I can wire from/to heads.
"""


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
        self.action = action  # REAL atomic move: (start, end)
        self.children = {}    # {action: MCTSNode}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTS:
    def __init__(self, model, cpuct=1.5, num_sims=100, device="cpu"):
        self.model = model
        self.cpuct = cpuct
        self.num_sims = num_sims
        self.device = device
        self.root = MCTSNode()

    # =========================
    # TREE MANAGEMENT
    # =========================

    def reset(self):
        """Clear tree (call on new game)."""
        self.root = MCTSNode()

    def advance_to_child(self, action):
        """
        Advance root after a REAL move is played.
        Preserves statistics for multi-step turns.
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.reset()

    # =========================
    # PUBLIC SEARCH API
    # =========================

    def search(self, game, my_score, opp_score):
        """
        Run MCTS simulations from current game state.
        Returns the root node.
        Uses batched neural network evaluation for efficiency.
        """
        # If root is unexpanded, expand once
        if not self.root.children:
            self._expand_root(game)

        # Save initial state once - we'll restore to this after each sim
        initial_snapshot = game.fast_save()

        # Batch size for neural network evaluation
        batch_size = min(32, self.num_sims)
        
        sim_idx = 0
        while sim_idx < self.num_sims:
            # Collect a batch of leaf nodes to evaluate
            current_batch_size = min(batch_size, self.num_sims - sim_idx)
            
            leaf_nodes = []
            leaf_snapshots = []
            leaf_values = []  # For terminal nodes
            
            for _ in range(current_batch_size):
                # Restore to initial state
                game.fast_restore(initial_snapshot)
                node = self.root

                # -----------------
                # 1. SELECTION
                # -----------------
                while True:
                    if not node.children:
                        break

                    best_child = self._select_child(node)
                    if best_child is None:
                        break

                    # Apply move safely
                    try:
                        game.step_atomic(best_child.action)
                    except Exception:
                        # Prune illegal branch
                        node.children.pop(best_child.action, None)
                        break

                    node = best_child

                    # If turn ended, stop descent
                    if not game.dice:
                        break

                # -----------------
                # 2. CHECK TERMINAL / EXPAND
                # -----------------
                winner, _ = game.check_win()

                if winner != 0:
                    # Terminal state - no need for NN eval
                    v = 1.0 if winner == game.turn else -1.0
                    leaf_values.append((node, v))
                else:
                    # Expand if possible
                    self._expand_node(node, game)
                    # Save state for batched evaluation
                    leaf_nodes.append(node)
                    leaf_snapshots.append(game.fast_save())

            # -----------------
            # 3. BATCHED EVALUATION
            # -----------------
            if leaf_nodes:
                values = self._evaluate_batch(game, leaf_snapshots, my_score, opp_score)
                for node, v in zip(leaf_nodes, values):
                    self._backpropagate(node, v)

            # Backprop terminal values
            for node, v in leaf_values:
                self._backpropagate(node, v)

            sim_idx += current_batch_size

        # Restore game to initial state
        game.fast_restore(initial_snapshot)
        return self.root

    # =========================
    # CORE MCTS OPS
    # =========================

    def _select_child(self, node):
        """PUCT selection."""
        best_score = -float("inf")
        best_child = None

        sqrt_visits = math.sqrt(node.visits + 1)

        for child in node.children.values():
            u = self.cpuct * child.prior * sqrt_visits / (1 + child.visits)
            score = child.value + u

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand_root(self, game):
        legal = game.get_legal_moves()
        if not legal:
            return

        prior = 1.0 / len(legal)
        for mv in legal:
            self.root.children[mv] = MCTSNode(
                parent=self.root,
                action=mv,
                prior=prior
            )

    def _expand_node(self, node, game):
        """Expand leaf node with all legal atomic moves."""
        if node.children:
            return

        legal = game.get_legal_moves()
        if not legal:
            return

        prior = 1.0 / len(legal)
        for mv in legal:
            node.children[mv] = MCTSNode(
                parent=node,
                action=mv,
                prior=prior
            )

    def _evaluate(self, game, my_score, opp_score):
        """Neural value evaluation (REAL board, no canonical)."""
        with torch.no_grad():
            b_t, c_t = game.get_vector(
                my_score=my_score,
                opp_score=opp_score,
                device=self.device,
                canonical=False
            )

            out = self.model(b_t.unsqueeze(0), c_t.unsqueeze(0))

            if isinstance(out, tuple):
                v = out[2]
            else:
                v = out

            return float(v.item())

    def _evaluate_batch(self, game, snapshots, my_score, opp_score):
        """Batched neural value evaluation for multiple states."""
        if not snapshots:
            return []
        
        batch_size = len(snapshots)
        
        # Pre-allocate tensors for the batch
        board_tensors = []
        ctx_tensors = []
        
        for snapshot in snapshots:
            game.fast_restore(snapshot)
            b_t, c_t = game.get_vector(
                my_score=my_score,
                opp_score=opp_score,
                device=self.device,
                canonical=False
            )
            board_tensors.append(b_t)
            ctx_tensors.append(c_t)
        
        # Stack into batches
        batch_board = torch.stack(board_tensors, dim=0)
        batch_ctx = torch.stack(ctx_tensors, dim=0)
        
        with torch.no_grad():
            out = self.model(batch_board, batch_ctx)
            
            if isinstance(out, tuple):
                v = out[2]
            else:
                v = out
            
            # Handle both scalar and tensor outputs
            if v.dim() == 0:
                return [float(v.item())]
            
            return [float(v[i].item()) for i in range(batch_size)]

    def _backpropagate(self, node, value):
        """
        Backpropagate value up the tree.
        Perspective flips each level.
        """
        v = value
        while node is not None:
            node.visits += 1
            node.value_sum += v
            v = -v
            node = node.parent

    # =========================
    # ACTION SELECTION
    # =========================

    def best_action(self, temperature=0.0):
        """
        Pick action from root.
        temperature = 0  -> greedy
        temperature > 0  -> stochastic
        """
        if not self.root.children:
            return None

        actions = list(self.root.children.keys())
        visits = [self.root.children[a].visits for a in actions]

        if temperature <= 0:
            # Greedy: pick action with most visits
            max_visits = -1
            best_idx = 0
            for i, v in enumerate(visits):
                if v > max_visits:
                    max_visits = v
                    best_idx = i
            return actions[best_idx]

        # Stochastic selection with temperature
        visits_tensor = torch.tensor(visits, dtype=torch.float)
        probs = visits_tensor ** (1.0 / temperature)
        probs = probs / probs.sum()

        idx = torch.multinomial(probs, 1).item()
        return actions[idx]

    # =========================
    # DEBUG / SAFETY
    # =========================

    def sanity_check(self, game):
        """
        Removes illegal children from root (debug / web safety)
        """
        legal = set(game.get_legal_moves())
        for a in list(self.root.children.keys()):
            if a not in legal:
                self.root.children.pop(a)

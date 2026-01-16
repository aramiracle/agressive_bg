import torch
import math
from collections import deque
from config import Config


class Node:
    __slots__ = ['visits', 'value_sum', 'prior', 'children', 'is_expanded']

    def __init__(self, prior=0.0):
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
        self.is_expanded = False


class MCTS:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.c_puct = Config.C_PUCT
        self.batch_size = Config.MCTS_BATCH

    def search(self, game, my_score, opp_score):
        root = Node()
        root_snapshot = game.fast_save()

        # Initial expansion of root
        self._evaluate_batch([deque([root])], [root_snapshot], game, my_score, opp_score)

        # Run simulations
        for _ in range(0, Config.NUM_SIMULATIONS, self.batch_size):
            paths, snaps = [], []

            for _ in range(self.batch_size):
                game.fast_restore(root_snapshot)
                node = root
                path = deque([node])

                # Selection: traverse tree using UCB
                while node.is_expanded and node.children:
                    sqrt_v = math.sqrt(node.visits) if node.visits > 0 else 1.0
                    best_act, best_child = None, -float('inf')
                    
                    for act, child in node.children.items():
                        # UCB calculation
                        q = child.value_sum / child.visits if child.visits > 0 else 0.0
                        u = self.c_puct * child.prior * sqrt_v / (1 + child.visits)
                        score = q + u
                        
                        if score > best_child:
                            best_child = score
                            best_act = (act, child)

                    if best_act is None: break
                    game.step_atomic(best_act[0])
                    node = best_act[1]
                    path.append(node)

                paths.append(path)
                snaps.append(game.fast_save())

            # Batch evaluation and backpropagation
            self._evaluate_batch(paths, snaps, game, my_score, opp_score)

        game.fast_restore(root_snapshot)
        return root

    def _evaluate_batch(self, paths, snaps, game, ms, os):
        boards, ctxs, idxs = [], [], []

        for i, snap in enumerate(snaps):
            game.fast_restore(snap)
            winner, _ = game.check_win()

            if winner != 0:
                # Terminal node - immediate backprop
                val = 1.0 if winner == game.turn else -1.0
                self._backprop(paths[i], val)
            else:
                # Non-terminal - need neural network evaluation
                b, c = game.get_vector(ms, os, self.device, canonical=True)
                boards.append(b)
                ctxs.append(c)
                idxs.append(i)

        if not boards:
            return

        # Batch inference
        with torch.no_grad():
            pb = torch.stack(boards)
            pc = torch.stack(ctxs)
            pf, pt, v, _ = self.model(pb, pc)
            pf = torch.softmax(pf, dim=1)
            pt = torch.softmax(pt, dim=1)
            values = v.squeeze(-1).tolist()

        # Expand nodes and backpropagate
        for j, i in enumerate(idxs):
            path = paths[i]
            leaf = path[-1]
            game.fast_restore(snaps[i])

            moves = game.get_legal_moves()
            if moves:
                total = 0.0
                probs = []

                for m in moves:
                    canon_m = game.real_action_to_canonical(m)
                    s, e = canon_m
                    si = Config.BAR_IDX if s == 'bar' else s
                    ei = Config.OFF_IDX if e == 'off' else e
                    p = (pf[j, si] * pt[j, ei]).item()
                    probs.append((m, p))
                    total += p

                # Normalize probabilities
                for m, p in probs:
                    leaf.children[m] = Node(prior=p / total if total > 0 else 1.0 / len(moves))
                leaf.is_expanded = True

            self._backprop(path, values[j])

    # Inside MCTS class

    def _backprop(self, path, val):
        """
        Backpropagate value through the path.
        In Backgammon, since we use canonical board states (perspective of current player),
        the value should alternate signs for each level of the tree.
        """
        for n in reversed(path):
            n.visits += 1
            n.value_sum += val
            val = -val  # Flip value for the opponent's node
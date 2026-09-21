"""
Efficient within-turn MCTS.

A backgammon roll has at most a few dozen distinct complete plays, so the tree
is one ply of full turns (not atomic dice steps). That also fixes A1: every
child of the root belongs to the same player, so values are NEVER negated
during backup.

Search:

  1. Enumerate every legal full turn (engine.get_legal_turns).
  2. Batch-evaluate afterstates with the network's 6-way outcome head.
     Win/equity of each position is the leaf judgment (A3: value-only priors).
  3. Prune losing plays whose equity is more than SEARCH_PRUNE_MARGIN below
     the best, keeping at most SEARCH_PRUNE_TOP_K (B1).
  4. Optional 2-ply expectimax on the survivors (B2).
  5. PUCT among the survivors. Leaf values are already known, so this is a
     bandit over complete plays (NUM_SIMULATIONS visits, Dirichlet noise in
     self-play). Race leaves use the analytic pip formula (B6).

Perspective: the network always evaluates a position for the player about to
roll. A mover's afterstate is therefore scored from the opponent's seat and
flipped back.
"""

import math
import torch
from src.config import Config
from src.engine import BackgammonGame
from src.utils.outcome import (
    flip, money_equity, match_equity_batch, terminal_distribution,
    race_distribution, race_win_probability,
)

# 21 distinct rolls with their probabilities
ALL_ROLLS = []
for _d1 in range(1, Config.DICE_SIDES + 1):
    for _d2 in range(_d1, Config.DICE_SIDES + 1):
        if _d1 == _d2:
            ALL_ROLLS.append(([_d1] * 4, 1.0 / 36.0))
        else:
            ALL_ROLLS.append(([_d1, _d2], 2.0 / 36.0))


class Candidate:
    __slots__ = ("path", "position", "probs", "equity", "visits", "value_sum", "prior")

    def __init__(self, path, position):
        self.path = path            # tuple of ((start, end), die)
        self.position = position    # (board, bar, off) after the play
        self.probs = None           # outcome distribution, mover's perspective [6]
        self.equity = 0.0
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0

    def value(self):
        """Mean backed-up equity. Same-player tree: never negated (A1)."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class SearchResult:
    def __init__(self, candidates):
        self.candidates = sorted(
            candidates,
            key=lambda c: (c.visits, c.equity),
            reverse=True,
        )

    def best(self):
        return self.candidates[0] if self.candidates else None

    def __len__(self):
        return len(self.candidates)


def prune_losing_moves(candidates, top_k, margin):
    """
    Keep plays within `margin` equity of the best, at most `top_k` of them.
    Losers stay in `candidates` with prior/visits left at 0 so exploration
    can still see them, but PUCT only runs on the returned survivors.
    """
    if not candidates:
        return []
    order = sorted(range(len(candidates)), key=lambda i: candidates[i].equity, reverse=True)
    best_eq = candidates[order[0]].equity
    kept = []
    for i in order:
        if len(kept) >= top_k:
            break
        if candidates[i].equity >= best_eq - margin:
            kept.append(candidates[i])
    return kept


def select_candidate(result, explore, temperature=None):
    """
    Pick a candidate from a SearchResult.

    explore=False -> most visits, then equity (greedy).
    explore=True  -> sample with softmax((equity - best) / temperature) (B5).
    """
    cands = result.candidates
    if not cands:
        return None
    if not explore or len(cands) == 1:
        return cands[0]

    t = Config.EXPLORE_TEMPERATURE if temperature is None else temperature
    best = max(c.equity for c in cands)
    weights = torch.tensor([math.exp((c.equity - best) / max(t, 1e-6)) for c in cands])
    idx = torch.multinomial(weights / weights.sum(), 1).item()
    return cands[idx]


class MCTS:
    def __init__(self, model, device="cpu", ply=None, top_k=None, margin=None,
                 equity_table=None, batch_size=None, num_sims=None, cpuct=None):
        self.model = model
        self.device = device
        self.ply = Config.SEARCH_PLY if ply is None else int(ply)
        self.top_k = Config.SEARCH_PRUNE_TOP_K if top_k is None else int(top_k)
        self.margin = Config.SEARCH_PRUNE_MARGIN if margin is None else float(margin)
        self.batch_size = Config.SEARCH_EVAL_BATCH if batch_size is None else int(batch_size)
        self.num_sims = Config.NUM_SIMULATIONS if num_sims is None else int(num_sims)
        self.cpuct = Config.C_PUCT if cpuct is None else float(cpuct)
        self.equity_table = equity_table
        self.root = None

    def reset(self):
        """API compatibility; the search is rebuilt every call."""
        self.root = None

    def advance_to_child(self, action_with_die):
        """No tree reuse across atomic steps: complete turns are the nodes."""
        self.reset()

    # ------------------------------------------------------------------
    # Equity
    # ------------------------------------------------------------------
    def equity(self, probs, my_score, opp_score, cube):
        """probs [..., 6] from the mover's perspective -> equity in [-1, 1]."""
        if self.equity_table is not None and Config.MATCH_TARGET > 1:
            return match_equity_batch(probs, my_score, opp_score, cube, self.equity_table)
        return money_equity(probs)

    # ------------------------------------------------------------------
    # Leaf evaluation
    # ------------------------------------------------------------------
    def _leaf(self, position, on_roll, ctx_static, on_roll_score, opp_score, pending):
        """
        Outcome distribution for `position` from the perspective of `on_roll`
        (the player about to roll), or None after queueing a network eval.
        """
        board, bar, off = position
        idx_me = 0 if on_roll == 1 else 1
        idx_opp = 1 - idx_me

        if off[idx_me] >= Config.CHECKERS_PER_PLAYER:
            return terminal_distribution(True, BackgammonGame.win_type_of(board, bar, off, on_roll))
        if off[idx_opp] >= Config.CHECKERS_PER_PLAYER:
            return terminal_distribution(False, BackgammonGame.win_type_of(board, bar, off, -on_roll))

        if Config.RACE_EARLY_TERMINATION and BackgammonGame.gammon_free_race(board, bar, off):
            p = race_win_probability(
                BackgammonGame.pips(board, bar, on_roll),
                BackgammonGame.pips(board, bar, -on_roll),
            )
            return race_distribution(p)

        cube, cube_owner, crawford = ctx_static
        vec, ctx = BackgammonGame.encode_state(
            board, bar, off, on_roll, cube, cube_owner, crawford,
            on_roll_score, opp_score, canonical=True,
        )
        pending.append((vec, ctx))
        return None

    def _evaluate_pending(self, pending):
        """Batched network evaluation. Returns probs [N, 6] (encoded-turn perspective)."""
        if not pending:
            return None
        boards = torch.tensor([p[0] for p in pending], dtype=torch.long, device=self.device)
        ctxs = torch.tensor([p[1] for p in pending], dtype=torch.float32, device=self.device)
        outs = []
        with torch.inference_mode():
            for start in range(0, len(pending), self.batch_size):
                logits, _ = self.model(boards[start:start + self.batch_size],
                                       ctxs[start:start + self.batch_size])
                outs.append(torch.softmax(logits.float(), dim=-1))
        return torch.cat(outs, dim=0).cpu()

    @staticmethod
    def _resolve(values, pending_probs):
        """Replace None placeholders (in order) with rows from pending_probs."""
        k = 0
        out = []
        for v in values:
            if v is None:
                out.append(pending_probs[k])
                k += 1
            else:
                out.append(v)
        return out

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, game, my_score, opp_score, reset_tree=True, stochastic=False):
        """
        Evaluate every legal complete play for the current dice, prune losers,
        optionally 2-ply the survivors, then run PUCT among them.

        Returns a SearchResult. Empty if there is no legal play.
        """
        if reset_tree:
            self.root = None

        mover = game.turn
        opp = -mover
        root = game.fast_save()
        ctx_static = (game.cube, game.cube_owner, game.crawford_active)

        turns = game.get_legal_turns()
        if not turns:
            game.fast_restore(root)
            return SearchResult([])

        candidates = [Candidate(path, pos) for path, pos in turns]

        # ---------- 1-ply: afterstates evaluated with the opponent on roll ----------
        pending = []
        values = [
            self._leaf(c.position, opp, ctx_static, opp_score, my_score, pending)
            for c in candidates
        ]
        values = self._resolve(values, self._evaluate_pending(pending))
        probs = flip(torch.stack(values))               # -> mover's perspective
        eq = self.equity(probs, my_score, opp_score, game.cube)
        for i, c in enumerate(candidates):
            c.probs = probs[i]
            c.equity = float(eq[i])

        # ---------- prune losing moves (B1) ----------
        kept = prune_losing_moves(candidates, self.top_k, self.margin)

        # ---------- 2-ply expectimax for the survivors (B2) ----------
        if self.ply >= 2 and len(kept) > 1:
            self._two_ply(game, kept, mover, ctx_static, my_score, opp_score)

        # ---------- PUCT among survivors (same-player backup, A1) ----------
        self._puct(kept, stochastic=stochastic)
        self.root = kept

        game.fast_restore(root)
        return SearchResult(candidates)

    def _puct(self, kept, stochastic=False):
        if not kept:
            return

        equities = torch.tensor([c.equity for c in kept], dtype=torch.float32)
        # A3: priors from the network's win/equity judgment, not a separate policy head.
        logits = (equities - equities.max()) / max(Config.EXPLORE_TEMPERATURE, 1e-6)
        priors = torch.softmax(logits, dim=0)

        if stochastic and Config.DIRICHLET_EPS > 0:
            noise = torch.distributions.Dirichlet(
                torch.full((len(kept),), Config.DIRICHLET_ALPHA)
            ).sample()
            priors = (1.0 - Config.DIRICHLET_EPS) * priors + Config.DIRICHLET_EPS * noise

        priors = torch.clamp(priors, min=Config.MIN_PRIOR)
        priors = priors / priors.sum()
        for c, p in zip(kept, priors.tolist()):
            c.prior = p
            c.visits = 1
            c.value_sum = c.equity

        remaining = max(0, self.num_sims - len(kept))
        for _ in range(remaining):
            parent_n = sum(c.visits for c in kept)
            sqrt_n = math.sqrt(parent_n + Config.MIN_PRIOR)
            best_score = -math.inf
            best_child = kept[0]
            for child in kept:
                q = child.value()
                u = self.cpuct * child.prior * sqrt_n / (1 + child.visits)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_child = child
            # Same player at every child: add equity, do not negate (A1).
            best_child.visits += 1
            best_child.value_sum += best_child.equity

    def _two_ply(self, game, kept, mover, ctx_static, my_score, opp_score):
        opp = -mover
        idx_mover = 0 if mover == 1 else 1
        pending = []
        plan = []

        for ci, cand in enumerate(kept):
            board, bar, off = cand.position
            if off[idx_mover] >= Config.CHECKERS_PER_PLAYER:
                continue  # already won; 1-ply value is exact

            for roll, weight in ALL_ROLLS:
                game.board = list(board)
                game.bar = list(bar)
                game.off = list(off)
                game.turn = opp
                game.dice = list(roll)

                replies = game.get_legal_turns()
                if not replies:
                    reply_positions = [cand.position]
                else:
                    reply_positions = [pos for _, pos in replies]

                vals = [
                    self._leaf(pos, mover, ctx_static, my_score, opp_score, pending)
                    for pos in reply_positions
                ]
                plan.append((ci, weight, vals))

        pending_probs = self._evaluate_pending(pending)
        expected = [torch.zeros(Config.NUM_OUTCOMES) for _ in kept]
        touched = [False] * len(kept)

        k = 0
        for ci, weight, vals in plan:
            rows = []
            for v in vals:
                if v is None:
                    rows.append(pending_probs[k])
                    k += 1
                else:
                    rows.append(v)
            reply_probs = torch.stack(rows)                       # mover's perspective
            reply_eq = self.equity(reply_probs, my_score, opp_score, game.cube)
            best_reply = int(torch.argmin(reply_eq))               # opponent minimises our equity
            expected[ci] += weight * reply_probs[best_reply]
            touched[ci] = True

        for ci, cand in enumerate(kept):
            if touched[ci]:
                cand.probs = expected[ci]
                cand.equity = float(self.equity(expected[ci], my_score, opp_score, game.cube))


# Alias used by trainers / UI that historically constructed a Searcher.
Searcher = MCTS

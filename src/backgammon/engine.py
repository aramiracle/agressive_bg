import torch
import random
from src.backgammon.config import Config

class BackgammonGame:
    """
    Backgammon Engine with Strict Rule Adherence.
    Compatible with MCTS and Neural Network inputs.
    """

    def __init__(self):
        # Match-level state
        self.match_target = Config.MATCH_TARGET
        self.match_scores = {1: 0, -1: 0}

        # Crawford State
        self.crawford_active = False
        self.crawford_used = False

        self.reset()

    # =========================
    # CORE GAME STATE
    # =========================

    def reset(self):
        """Resets the board for a new game, preserving match state."""
        self.board = [0] * Config.NUM_POINTS
        for pos, count in Config.INITIAL_SETUP.items():
            self.board[pos] = count

        self.bar = [0, 0]  # Index 0: P1 (Positive), Index 1: P-1 (Negative)
        self.off = [0, 0]
        self.turn = 1

        self.cube = 1
        self.cube_owner = 0
        self.dice = []

        self._update_crawford_status()
        return self.get_state_key()

    def set_match_scores(self, p1_score, p2_score):
        self.match_scores = {1: int(p1_score), -1: int(p2_score)}
        self._update_crawford_status()

    def _update_crawford_status(self):
        p1_score = self.match_scores[1]
        p2_score = self.match_scores[-1]

        is_match_point = (
            p1_score == self.match_target - 1 or
            p2_score == self.match_target - 1
        )

        if is_match_point and not self.crawford_used:
            self.crawford_active = True
        else:
            self.crawford_active = False

    def get_state_key(self):
        return (
            tuple(self.board),
            tuple(self.bar),
            tuple(self.off),
            self.turn,
            self.cube,
            self.cube_owner,
            tuple(sorted(self.dice)),
            self.crawford_active
        )

    # =========================
    # SNAPSHOTS (Thread Safe)
    # =========================

    def fast_save(self):
        return (
            tuple(self.board),
            tuple(self.bar),
            tuple(self.off),
            self.turn,
            self.cube,
            self.cube_owner,
            tuple(sorted(self.dice)),
            self.crawford_active,
            self.crawford_used,
            tuple(sorted(self.match_scores.items()))
        )

    def fast_restore(self, snapshot):
        self.board = list(snapshot[0])
        self.bar = list(snapshot[1])
        self.off = list(snapshot[2])
        self.turn = snapshot[3]
        self.cube = snapshot[4]
        self.cube_owner = snapshot[5]
        self.dice = list(snapshot[6])
        self.crawford_active = snapshot[7]
        self.crawford_used = snapshot[8]
        self.match_scores = dict(snapshot[9])

    def copy(self):
        g = BackgammonGame()
        g.fast_restore(self.fast_save())
        return g

    # =========================
    # DICE & TURN
    # =========================

    def roll_dice(self):
        d1 = random.randint(1, Config.DICE_SIDES)
        d2 = random.randint(1, Config.DICE_SIDES)
        if d1 == d2:
            self.dice = [d1, d1, d1, d1]
        else:
            self.dice = [d1, d2]
        return list(self.dice)

    def switch_turn(self):
        self.turn *= -1
        self.dice = []

    # =========================
    # LEGAL MOVES (DFS STRICT)
    # =========================

    def get_legal_moves(self):
        """
        Generates strictly legal unique next atomic moves.
        Enforces Maximality Rule: Must play as many dice as possible.
        """
        if not self.dice:
            return []

        valid_paths = []
        self._find_move_paths(self.board, self.bar, self.off, self.dice, [], valid_paths)

        # Remove empty paths (no atomic moves possible)
        valid_paths = [p for p in valid_paths if len(p) > 0]
        if not valid_paths:
            return []


        # 1. Must use maximum number of dice
        max_len = max(len(p) for p in valid_paths)
        longest_paths = [p for p in valid_paths if len(p) == max_len]

        # 2. If still ambiguous and fewer dice used than available, maximize pips
        if max_len < len(self.dice) and len(longest_paths) > 1:
            best = []
            max_pips = -1
            for p in longest_paths:
                pips = sum(step[1] for step in p)
                if pips > max_pips:
                    best = [p]
                    max_pips = pips
                elif pips == max_pips:
                    best.append(p)
            longest_paths = best

        # Only return FIRST atomic action from each valid sequence
        unique_actions = set()
        for p in longest_paths:
            unique_actions.add(p[0][0])

        return sorted(unique_actions, key=lambda x: str(x))

    # =========================
    # DFS CORE (FIXED)
    # =========================

    def _find_move_paths(self, board, bar, off, dice, current_path, results):
        """
        DFS to find all legal chains of moves.
        IMPORTANT: Iterate by index, not set(dice), so duplicate dice are handled correctly.
        """
        can_move = False
        can_bear = self._can_bear_off(board, bar, self.turn)

        for i in range(len(dice)):
            die = dice[i]
            moves = self._get_single_moves(board, bar, self.turn, die, can_bear)

            for m in moves:
                can_move = True

                next_board = list(board)
                next_bar = list(bar)
                next_off = list(off)

                self._apply_single_move_logic(
                    next_board, next_bar, next_off,
                    self.turn, m[0], m[1]
                )

                next_dice = dice[:i] + dice[i + 1:]

                self._find_move_paths(
                    next_board,
                    next_bar,
                    next_off,
                    next_dice,
                    current_path + [(m, die)],
                    results
                )

        if not can_move:
            results.append(current_path)

    # =========================
    # SINGLE MOVE GENERATION
    # =========================

    def _get_single_moves(self, board, bar, player, die, can_bear_off_cached):
        moves = []
        p_idx = 0 if player == 1 else 1

        # 1. Must enter from bar
        if bar[p_idx] > 0:
            target = (Config.NUM_POINTS - die) if player == 1 else (die - 1)
            if 0 <= target < Config.NUM_POINTS and self._is_open(board, target, player):
                return [("bar", target)]
            return []

        # 2. Normal board moves
        iterator = (
            range(Config.NUM_POINTS - 1, -1, -1)
            if player == 1 else
            range(Config.NUM_POINTS)
        )

        for i in iterator:
            if (player == 1 and board[i] > 0) or (player == -1 and board[i] < 0):
                target = i - die if player == 1 else i + die

                # Bear off
                if (player == 1 and target < 0) or (player == -1 and target >= Config.NUM_POINTS):
                    if can_bear_off_cached:
                        dist_to_edge = (i + 1) if player == 1 else (Config.NUM_POINTS - i)
                        if dist_to_edge == die:
                            moves.append((i, "off"))
                        elif dist_to_edge < die:
                            is_furthest = True
                            check_range = (
                                range(i + 1, Config.NUM_POINTS)
                                if player == 1 else
                                range(0, i)
                            )
                            for k in check_range:
                                if (player == 1 and board[k] > 0) or (player == -1 and board[k] < 0):
                                    is_furthest = False
                                    break
                            if is_furthest:
                                moves.append((i, "off"))

                # Normal move
                elif 0 <= target < Config.NUM_POINTS:
                    if self._is_open(board, target, player):
                        moves.append((i, target))

        return moves

    def _is_open(self, board, target, player):
        cnt = board[target]
        if cnt == 0:
            return True
        if (player == 1 and cnt > 0) or (player == -1 and cnt < 0):
            return True
        if abs(cnt) == 1:
            return True
        return False

    def _can_bear_off(self, board, bar, player):
        p_idx = 0 if player == 1 else 1
        if bar[p_idx] > 0:
            return False

        if player == 1:
            for i in range(Config.HOME_SIZE, Config.NUM_POINTS):
                if board[i] > 0:
                    return False
        else:
            for i in range(0, Config.NUM_POINTS - Config.HOME_SIZE):
                if board[i] < 0:
                    return False

        return True

    # =========================
    # STRICT VALIDATION
    # =========================

    def _is_atomic_move_legal_for_die(self, action, die):
        start, end = action

        p_idx = 0 if self.turn == 1 else 1
        if self.bar[p_idx] > 0 and start != "bar":
            return False

        can_bear = self._can_bear_off(self.board, self.bar, self.turn)
        legal = self._get_single_moves(
            self.board,
            self.bar,
            self.turn,
            die,
            can_bear
        )
        return action in legal

    # =========================
    # ACTION APPLICATION
    # =========================

    def step_atomic(self, action):
        """
        Execute a single action and update state/dice.
        STRICT: Only allow actions from get_legal_moves().
        """
        legal_actions = set(self.get_legal_moves())
        if action not in legal_actions:
            raise ValueError(
                f"Illegal atomic action {action} "
                f"for dice {self.dice}. "
                f"Allowed: {sorted(list(legal_actions))}"
            )

        die_to_remove = self._identify_die_used(*action)
        if die_to_remove is None:
            raise ValueError(
                f"Illegal move {action} with dice {self.dice} - die mismatch"
            )

        self._apply_single_move_logic(
            self.board,
            self.bar,
            self.off,
            self.turn,
            action[0],
            action[1]
        )

        self.dice.remove(die_to_remove)
        return self.check_win()

    def _identify_die_used(self, start, end):
        if start == "bar":
            dist = (
                Config.NUM_POINTS - end
                if self.turn == 1 else
                end + 1
            )
        elif end == "off":
            dist = (
                start + 1
                if self.turn == 1 else
                Config.NUM_POINTS - start
            )
        else:
            dist = abs(start - end)

        # Exact die
        if dist in self.dice:
            if self._is_atomic_move_legal_for_die((start, end), dist):
                return dist

        # Overshoot bear-off
        candidates = sorted(d for d in self.dice if d > dist)
        for d in candidates:
            if self._is_atomic_move_legal_for_die((start, end), d):
                return d

        return None

    def _apply_single_move_logic(self, board, bar, off, player, start, end):
        p_idx = 0 if player == 1 else 1

        if start == "bar":
            bar[p_idx] -= 1
        else:
            board[start] -= 1 if player == 1 else -1

        if end == "off":
            off[p_idx] += 1
        else:
            # Hit logic
            if player == 1 and board[end] == -1:
                board[end] = 0
                bar[1] += 1
            elif player == -1 and board[end] == 1:
                board[end] = 0
                bar[0] += 1

            board[end] += 1 if player == 1 else -1

    # =========================
    # WINNING & SCORING
    # =========================

    def check_win(self):
        if self.off[0] == Config.CHECKERS_PER_PLAYER:
            return self._finalize_win(1)
        if self.off[1] == Config.CHECKERS_PER_PLAYER:
            return self._finalize_win(-1)
        return 0, 0

    def _finalize_win(self, winner):
        loser = -winner
        loser_idx = 0 if loser == 1 else 1

        mult = 1
        if self.off[loser_idx] == 0:
            mult = 2
            has_bar = self.bar[loser_idx] > 0
            has_home = False

            home_range = (
                range(0, 6)
                if winner == 1 else
                range(18, 24)
            )

            for i in home_range:
                if (loser == 1 and self.board[i] > 0) or \
                   (loser == -1 and self.board[i] < 0):
                    has_home = True
                    break

            if has_bar or has_home:
                mult = 3

        points = self.cube * mult
        self.match_scores[winner] += points

        if self.crawford_active:
            self.crawford_used = True

        return winner, points

    def handle_cube_refusal(self):
        winner = self.turn
        points = self.cube
        self.match_scores[winner] += points

        if self.crawford_active:
            self.crawford_used = True

        return winner, points

    # =========================
    # AI REPRESENTATION
    # =========================

    def get_vector(self, my_score=None, opp_score=None, device="cpu", canonical=True):
        vec_data = [0] * Config.BOARD_SEQ_LEN
        flip = canonical and self.turn == -1

        for i in range(Config.NUM_POINTS):
            val = self.board[i] if not flip else -self.board[Config.NUM_POINTS - 1 - i]
            vec_data[i] = val + Config.EMBED_OFFSET

        if not flip:
            vec_data[24] = self.bar[0] + Config.EMBED_OFFSET
            vec_data[25] = -self.bar[1] + Config.EMBED_OFFSET
            vec_data[26] = self.off[0] + Config.EMBED_OFFSET
            vec_data[27] = -self.off[1] + Config.EMBED_OFFSET
        else:
            vec_data[24] = self.bar[1] + Config.EMBED_OFFSET
            vec_data[25] = -self.bar[0] + Config.EMBED_OFFSET
            vec_data[26] = self.off[1] + Config.EMBED_OFFSET
            vec_data[27] = -self.off[0] + Config.EMBED_OFFSET

        if my_score is None or opp_score is None:
            if not flip:
                s_my, s_opp = self.match_scores[1], self.match_scores[-1]
            else:
                s_my, s_opp = self.match_scores[-1], self.match_scores[1]
        else:
            s_my, s_opp = my_score, opp_score

        turn_val = 1.0 if self.turn == 1 else -1.0

        ctx_data = [
            turn_val,
            float(self.cube),
            float(s_my) / float(self.match_target),
            float(s_opp) / float(self.match_target)
        ]

        t_board = torch.tensor(vec_data, dtype=torch.long, device=device)
        t_ctx = torch.tensor(ctx_data, dtype=torch.float, device=device)

        return t_board, t_ctx

    # =========================
    # CANONICAL ACTIONS
    # =========================

    def real_action_to_canonical(self, action):
        if self.turn == 1:
            return action
        start, end = action
        s_c = "bar" if start == "bar" else Config.NUM_POINTS - 1 - start
        e_c = "off" if end == "off" else Config.NUM_POINTS - 1 - end
        return s_c, e_c

    def canonical_action_to_real(self, action):
        return self.real_action_to_canonical(action)

    # =========================
    # CUBE
    # =========================

    def can_double(self):
        if self.crawford_active:
            return False
        if self.off[0] >= Config.CHECKERS_PER_PLAYER:
            return False
        if self.off[1] >= Config.CHECKERS_PER_PLAYER:
            return False
        return self.cube_owner == 0 or self.cube_owner == self.turn

    def apply_double(self):
        if not self.can_double():
            return False
        self.cube *= 2
        self.cube_owner = -self.turn
        return True

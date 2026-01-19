import torch
import random
from src.backgammon.config import Config


class BackgammonGame:
    """
    Full Backgammon engine with:
    - Legal move generation
    - Bar / bear-off logic
    - Doubling cube + Crawford rule
    - Match scoring
    - AI canonical board encoding
    - Fast snapshot system (thread-safe)
    - Torch vector output
    """

    def __init__(self):
        # Match-level state
        self.match_scores = {1: 0, -1: 0}
        self.crawford = False
        self.crawford_used = False
        self.reset()

    # =========================
    # CORE GAME STATE
    # =========================

    def reset(self):
        self.board = [0] * Config.NUM_POINTS
        for pos, count in Config.INITIAL_SETUP.items():
            self.board[pos] = count

        self.bar = [0, 0]
        self.off = [0, 0]
        self.turn = 1

        self.cube = 1
        self.cube_owner = 0
        self.dice = []

        # Crawford only applies to a single game
        self.crawford = getattr(self, "crawford", False)

        return self.get_state_key()

    def set_match_scores(self, score_player1, score_player_minus1):
        """
        Updates match scores and enables Crawford game if needed
        """
        self.match_scores = {
            1: int(score_player1),
            -1: int(score_player_minus1)
        }

        if not self.crawford_used and (
            self.match_scores[1] == Config.MATCH_TARGET - 1 or
            self.match_scores[-1] == Config.MATCH_TARGET - 1
        ):
            self.crawford = True
        else:
            self.crawford = False

    # =========================
    # FAST SNAPSHOT SYSTEM
    # =========================

    def fast_save(self):
        """
        Immutable snapshot for threading / AI search
        """
        return (
            tuple(self.board),
            tuple(self.bar),
            tuple(self.off),
            self.turn,
            self.cube,
            self.cube_owner,
            tuple(self.dice),
            self.crawford,
            tuple(sorted(self.match_scores.items()))
        )

    def fast_restore(self, snapshot):
        """
        Restore snapshot
        """
        self.board = list(snapshot[0])
        self.bar = list(snapshot[1])
        self.off = list(snapshot[2])
        self.turn = snapshot[3]
        self.cube = snapshot[4]
        self.cube_owner = snapshot[5]
        self.dice = list(snapshot[6])
        self.crawford = snapshot[7]

        ms = dict(snapshot[8])
        self.match_scores = {1: ms.get(1, 0), -1: ms.get(-1, 0)}

    # =========================
    # DICE
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

    # =========================
    # LEGAL MOVES
    # =========================

    def get_legal_moves(self, dice_rolls=None):
        rolls = dice_rolls if dice_rolls else self.dice
        if not rolls:
            return []

        unique_moves = set()
        for die in set(rolls):
            moves = self._get_single_moves(self.board, self.bar, self.turn, die)
            unique_moves.update(moves)

        return list(unique_moves)

    def _get_single_moves(self, board, bar, player, die):
        moves = []
        p_idx = 0 if player == 1 else 1
        max_idx = Config.NUM_POINTS - 1

        # Must enter from bar
        if bar[p_idx] > 0:
            target = (Config.NUM_POINTS - die) if player == 1 else (die - 1)
            if 0 <= target <= max_idx and self._is_open(board, target, player):
                return [("bar", target)]
            return []

        can_bear = self._can_bear_off(board, bar, player)

        iterator = (
            range(max_idx, -1, -1)
            if player == 1 else
            range(0, Config.NUM_POINTS)
        )

        for i in iterator:
            if (player == 1 and board[i] > 0) or (player == -1 and board[i] < 0):
                target = i - die if player == 1 else i + die

                # Bear off
                if target < 0 or target > max_idx:
                    if can_bear:
                        dist = (i + 1) if player == 1 else (Config.NUM_POINTS - i)

                        if dist == die:
                            moves.append((i, "off"))
                        elif dist < die:
                            # Check if furthest checker
                            is_furthest = True
                            behind_range = (
                                range(i + 1, Config.NUM_POINTS)
                                if player == 1 else
                                range(0, i)
                            )
                            for k in behind_range:
                                if (player == 1 and board[k] > 0) or \
                                   (player == -1 and board[k] < 0):
                                    is_furthest = False
                                    break

                            if is_furthest:
                                moves.append((i, "off"))

                # Normal move
                elif self._is_open(board, target, player):
                    moves.append((i, target))

        return moves

    def _is_open(self, board, target, player):
        if board[target] == 0:
            return True
        if (player == 1 and board[target] > 0) or \
           (player == -1 and board[target] < 0):
            return True
        if abs(board[target]) == 1:
            return True
        return False

    def _can_bear_off(self, board, bar, player):
        p_idx = 0 if player == 1 else 1
        if bar[p_idx] > 0:
            return False

        if player == 1:
            return all(board[i] <= 0 for i in range(Config.HOME_SIZE, Config.NUM_POINTS))
        else:
            return all(board[i] >= 0 for i in range(0, Config.NUM_POINTS - Config.HOME_SIZE))

    # =========================
    # APPLY MOVE
    # =========================

    def step_atomic(self, action):
        start, end = action

        self._apply_single_move_logic(
            self.board, self.bar, self.off, self.turn, start, end
        )

        # Determine die used
        if start == "bar":
            die_used = Config.NUM_POINTS - end if self.turn == 1 else end + 1
        elif end == "off":
            die_used = (
                start + 1
                if self.turn == 1 else
                Config.NUM_POINTS - start
            )
            if die_used not in self.dice and self.dice:
                die_used = max(self.dice)
        else:
            die_used = abs(start - end)

        # Remove die
        if die_used in self.dice:
            self.dice.remove(die_used)
        elif self.dice:
            self.dice.remove(max(self.dice))

        return self.check_win()

    def _apply_single_move_logic(self, board, bar, off, player, start, end):
        p_idx = 0 if player == 1 else 1

        # Remove checker
        if start == "bar":
            bar[p_idx] -= 1
        else:
            board[start] += -1 if player == 1 else 1

        # Place checker
        if end == "off":
            off[p_idx] += 1
        else:
            # Hit blot
            if (player == 1 and board[end] == -1) or \
               (player == -1 and board[end] == 1):
                board[end] = 0
                bar[1 if player == 1 else 0] += 1

            board[end] += 1 if player == 1 else -1

    # =========================
    # WIN / SCORING
    # =========================

    def check_win(self):
        if self.off[0] == Config.CHECKERS_PER_PLAYER:
            return self._score_win(1)
        if self.off[1] == Config.CHECKERS_PER_PLAYER:
            return self._score_win(-1)
        return 0, 0

    def _score_win(self, winner):
        loser_idx = 0 if winner == -1 else 1
        mult = 1

        # Gammon / Backgammon
        if self.off[loser_idx] == 0:
            mult = 2

            if self.bar[loser_idx] > 0:
                mult = 3
            else:
                loser = -winner
                home_range = (
                    range(0, Config.HOME_SIZE)
                    if winner == 1 else
                    range(Config.NUM_POINTS - Config.HOME_SIZE, Config.NUM_POINTS)
                )

                for i in home_range:
                    if (loser == 1 and self.board[i] > 0) or \
                       (loser == -1 and self.board[i] < 0):
                        mult = 3
                        break

        return winner, mult

    # =========================
    # STATE ENCODING
    # =========================

    def get_state_key(self):
        return (
            tuple(self.board),
            tuple(self.bar),
            tuple(self.off),
            self.turn,
            self.cube,
            tuple(self.dice)
        )

    def get_vector(self, my_score=0, opp_score=0, device="cpu", canonical=True):
        vec_data = [0] * Config.BOARD_SEQ_LEN

        if canonical and self.turn == -1:
            for i in range(Config.NUM_POINTS):
                vec_data[i] = -self.board[Config.NUM_POINTS - 1 - i] + Config.EMBED_OFFSET

            vec_data[24] = self.bar[1] + Config.EMBED_OFFSET
            vec_data[25] = -self.bar[0] + Config.EMBED_OFFSET
            vec_data[26] = self.off[1] + Config.EMBED_OFFSET
            vec_data[27] = -self.off[0] + Config.EMBED_OFFSET

            ctx_data = [1.0, float(self.cube), float(my_score), float(opp_score)]
        else:
            for i in range(Config.NUM_POINTS):
                vec_data[i] = self.board[i] + Config.EMBED_OFFSET

            vec_data[24] = self.bar[0] + Config.EMBED_OFFSET
            vec_data[25] = -self.bar[1] + Config.EMBED_OFFSET
            vec_data[26] = self.off[0] + Config.EMBED_OFFSET
            vec_data[27] = -self.off[1] + Config.EMBED_OFFSET

            ctx_data = [float(self.turn), float(self.cube), float(my_score), float(opp_score)]

        t_board = torch.tensor(vec_data, dtype=torch.long, device=device)
        t_ctx = torch.tensor(ctx_data, dtype=torch.float, device=device)

        return t_board, t_ctx

    # =========================
    # CANONICAL CONVERSION
    # =========================

    def canonical_action_to_real(self, action):
        if self.turn == 1:
            return action

        start, end = action
        new_start = "bar" if start == "bar" else (Config.NUM_POINTS - 1 - start)
        new_end = "off" if end == "off" else (Config.NUM_POINTS - 1 - end)

        return new_start, new_end

    def real_action_to_canonical(self, action):
        if self.turn == 1:
            return action

        start, end = action
        new_start = "bar" if start == "bar" else (Config.NUM_POINTS - 1 - start)
        new_end = "off" if end == "off" else (Config.NUM_POINTS - 1 - end)

        return new_start, new_end

    # =========================
    # DOUBLING CUBE
    # =========================

    def can_double(self):
        # Crawford forbids doubling
        if self.crawford:
            return False

        # Cannot double if opponent owns cube
        if self.cube_owner != 0 and self.cube_owner != self.turn:
            return False

        return True

    def apply_double(self):
        next_cube = self.cube * 2

        if next_cube > Config.MATCH_TARGET:
            self.cube = Config.MATCH_TARGET
        else:
            self.cube = next_cube

        self.cube_owner = -self.turn

    def handle_cube_refusal(self):
        winner = self.turn
        points = self.cube

        # Mark Crawford as used if applicable
        if self.crawford:
            self.crawford_used = True

        return winner, points
    
    def copy(self):
        """
        Deep copy of the game for AI / MCTS threads
        """
        g = BackgammonGame()

        g.board = self.board.copy()
        g.bar = self.bar.copy()
        g.off = self.off.copy()

        g.turn = self.turn
        g.cube = self.cube
        g.cube_owner = self.cube_owner

        g.dice = self.dice.copy()

        g.crawford = self.crawford
        g.crawford_used = self.crawford_used
        g.match_scores = self.match_scores.copy()

        return g


import torch
import random
from src.config import Config

class BackgammonGame:
    """
    Backgammon Engine with Strict Rule Adherence.
    Compatible with MCTS and Neural Network inputs.
    """

    def __init__(self, train_mode=None):
        """
        Initialize BackgammonGame.
        
        Args:
            train_mode: If True, use training values from Config (R_WIN, R_GAMMON, R_BACKGAMMON).
                       If False, use real backgammon values (1, 2, 3).
                       If None, check Config.TRAIN_MODE if it exists, else default to True (training).
        """
        # Match-level state
        self.match_target = Config.MATCH_TARGET
        self.match_scores = {1: 0, -1: 0}

        # Crawford State
        self.crawford_active = False
        self.crawford_used = False
        
        # Determine train_mode
        if train_mode is None:
            # Try to get from Config, default to True (training mode)
            self.train_mode = getattr(Config, 'TRAIN_MODE', True)
        else:
            self.train_mode = train_mode

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
        g = BackgammonGame(train_mode=self.train_mode)
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
        
        Returns:
            List of ((start, end), die_used) tuples.
            Including 'die_used' is CRITICAL to resolve ambiguous bear-offs
            where the same move can be done by different dice, affecting future moves.
        """
        if not self.dice:
            return []

        valid_paths = []
        self._find_move_paths(self.board, self.bar, self.off, self.dice, [], valid_paths)

        # Remove empty paths (no atomic moves possible)
        valid_paths = [p for p in valid_paths if len(p) > 0]
        if not valid_paths:
            return []

        # 1. Must use maximum number of dice (Maximality Rule)
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

        # Return unique FIRST atomic actions with the specific die used
        # Structure: ((start, end), die)
        unique_actions = set()
        for p in longest_paths:
            # p[0] is ((start, end), die)
            unique_actions.add(p[0])

        return sorted(list(unique_actions), key=lambda x: str(x))

    # =========================
    # DFS CORE
    # =========================

    def _find_move_paths(self, board, bar, off, dice, current_path, results):
        """
        DFS to find all legal chains of moves.
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
            # Fix: Must check if target is valid AND strictly if open
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
                        
                        # Exact bear off
                        if dist_to_edge == die:
                            moves.append((i, "off"))
                        
                        # Overshoot (Furthest checker rule)
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
            # Check range 6 to 23 (indices where checkers block bearoff)
            for i in range(Config.HOME_SIZE, Config.NUM_POINTS):
                if board[i] > 0:
                    return False
        else:
            # Check range 0 to 17
            for i in range(0, Config.NUM_POINTS - Config.HOME_SIZE):
                if board[i] < 0:
                    return False

        return True

    # =========================
    # ACTION APPLICATION
    # =========================

    def step_atomic(self, action_wrapper):
            """
            Execute a single action and update state/dice.
            Args:
                action_wrapper: Tuple of ((start, end), die_used)
            """
            # 1. Unpack
            try:
                (start, end), die_used = action_wrapper
            except (ValueError, TypeError):
                raise ValueError(f"step_atomic expects ((start, end), die), got {action_wrapper}")

            # 2. Deep Validation: Does the die used actually match the move?
            # This prevents the AI from 'cheating' by moving 5 pips while removing a 1-die.
            if start != "bar" and end != "off":
                dist = abs(start - end)
                if dist != die_used:
                    raise ValueError(f"Die mismatch: Action moves {dist} pips but claims to use die {die_used}")

            # 3. Membership Validation
            # We call get_legal_moves to ensure Maximality Rules (must use highest dice) are met
            legal_actions = self.get_legal_moves()
            if action_wrapper not in legal_actions:
                raise ValueError(
                    f"Illegal atomic action {action_wrapper}. "
                    f"Dice available: {self.dice}. "
                    f"Legal now: {legal_actions}"
                )

            # 4. Apply Logic
            self._apply_single_move_logic(self.board, self.bar, self.off, self.turn, start, end)

            # 5. Consume Die
            # Use remove() to ensure only ONE instance of the die is consumed (crucial for doubles)
            self.dice.remove(die_used)

            return self.check_win()

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
            """
            Calculate win points based on game type (single/gammon/backgammon).
            
            If train_mode=True: Use Config.R_WIN, Config.R_GAMMON, Config.R_BACKGAMMON (e.g., 1, 3, 5 for aggressive training)
            If train_mode=False: Use real backgammon values (1, 2, 3)
            """
            loser = -winner
            loser_idx = 0 if loser == 1 else 1

            # Determine multiplier based on train_mode
            if self.train_mode:
                # Training mode: use Config values (could be aggressive like 1, 3, 5)
                mult = Config.R_WIN 
                
                if self.off[loser_idx] == 0:
                    # Gammon condition met
                    mult = Config.R_GAMMON
                    
                    has_bar = self.bar[loser_idx] > 0
                    has_home = False

                    home_range = (
                        range(0, 6)
                        if winner == 1 else
                        range(18, 24)
                    )

                    for i in home_range:
                        if (loser == 1 and self.board[i] > 0) or (loser == -1 and self.board[i] < 0):
                            has_home = True
                            break

                    if has_bar or has_home:
                        # Backgammon condition met
                        mult = Config.R_BACKGAMMON
            else:
                # Real backgammon mode: use standard values (1, 2, 3)
                mult = 1  # Single game
                
                if self.off[loser_idx] == 0:
                    # Gammon condition met
                    mult = 2
                    
                    has_bar = self.bar[loser_idx] > 0
                    has_home = False

                    home_range = (
                        range(0, 6)
                        if winner == 1 else
                        range(18, 24)
                    )

                    for i in home_range:
                        if (loser == 1 and self.board[i] > 0) or (loser == -1 and self.board[i] < 0):
                            has_home = True
                            break

                    if has_bar or has_home:
                        # Backgammon condition met
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
        """
        Translates a move from the current player's perspective to 
        a normalized board representation.
        """
        # UNPACK: Check if action is the new atomic wrapper ((s, e), d) 
        # or the old (s, e)
        if isinstance(action[0], tuple):
            (start, end), die = action
        else:
            start, end = action

        if self.turn == 1:
            return (start, end), die if isinstance(action[0], tuple) else (start, end)
            
        # Transformation for Player -1 (Negative)
        s_c = "bar" if start == "bar" else Config.NUM_POINTS - 1 - start
        e_c = "off" if end == "off" else Config.NUM_POINTS - 1 - end
        
        # Return in the same format it arrived in
        if isinstance(action[0], tuple):
            return (s_c, e_c), die
        return s_c, e_c

    def canonical_action_to_real(self, action):
        return self.real_action_to_canonical(action)

    # =========================
    # CUBE
    # =========================

    def can_double(self):
        # 1. Crawford Rule
        if self.crawford_active:
            return False
            
        # 2. Checkers on bar/off restrictions
        if self.off[0] >= Config.CHECKERS_PER_PLAYER:
            return False
        if self.off[1] >= Config.CHECKERS_PER_PLAYER:
            return False
        
        # 3. Dynamic Cube Cap (Your Fix)
        # Calculate the maximum effective points needed to end the match
        p1_score = self.match_scores[1]
        p2_score = self.match_scores[-1]
        
        # How many points does the player furthest from winning need?
        # e.g. Target 7, Scores (6, 0). Min is 0. Limit is 7.
        # e.g. Target 7, Scores (4, 4). Min is 4. Limit is 3.
        limit = self.match_target - min(p1_score, p2_score)
        
        # If the current cube is already greater than or equal to the limit,
        # doubling adds no value to the match result, only variance/instability.
        if self.cube >= limit:
            return False

        # 4. Owner Check
        return self.cube_owner == 0 or self.cube_owner == self.turn

    def apply_double(self):
        if not self.can_double():
            return False
        self.cube *= 2
        self.cube_owner = -self.turn
        return True
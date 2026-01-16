import torch
import random
from config import Config


class BackgammonGame:
    def __init__(self):
        self.reset()

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
        return self.get_state_key()

    def fast_save(self):
        """Thread-safe snapshot using tuple (immutable)."""
        return (
            tuple(self.board),
            tuple(self.bar),
            tuple(self.off),
            self.turn,
            self.cube,
            tuple(self.dice)
        )

    def fast_restore(self, snapshot):
        """Restore from immutable snapshot."""
        self.board = list(snapshot[0])
        self.bar = list(snapshot[1])
        self.off = list(snapshot[2])
        self.turn = snapshot[3]
        self.cube = snapshot[4]
        self.dice = list(snapshot[5])

    def roll_dice(self):
        d1 = random.randint(1, Config.DICE_SIDES)
        d2 = random.randint(1, Config.DICE_SIDES)
        self.dice = [d1, d1, d1, d1] if d1 == d2 else [d1, d2]
        return self.dice

    def switch_turn(self):
        self.turn *= -1

    def get_legal_moves(self, dice_rolls=None):
        rolls = dice_rolls if dice_rolls is not None else self.dice
        if not rolls:
            return []
        
        unique_atomic_moves = set()
        for die in set(rolls):
            single_moves = self._get_single_moves(self.board, self.bar, self.turn, die)
            unique_atomic_moves.update(single_moves)
        
        return list(unique_atomic_moves)

    def _get_single_moves(self, board, bar, player, die):
        moves = []
        p_idx = 0 if player == 1 else 1
        max_idx = Config.NUM_POINTS - 1
        
        if bar[p_idx] > 0:
            target = (Config.NUM_POINTS - die) if player == 1 else (die - 1)
            if 0 <= target <= max_idx and self._is_open(board, target, player):
                return [('bar', target)]
            return []

        can_bear_off = self._can_bear_off(board, bar, player)
        iterator = range(max_idx, -1, -1) if player == 1 else range(0, Config.NUM_POINTS)
        
        for i in iterator:
            if (player == 1 and board[i] > 0) or (player == -1 and board[i] < 0):
                target = i - die if player == 1 else i + die
                
                if (player == 1 and target < 0) or (player == -1 and target > max_idx):
                    if can_bear_off:
                        dist = (i + 1) if player == 1 else (Config.NUM_POINTS - i)
                        if dist == die:
                            moves.append((i, 'off'))
                        elif dist < die:
                            is_furthest = True
                            behind_range = range(i + 1, Config.NUM_POINTS) if player == 1 else range(0, i)
                            for k in behind_range:
                                if (player == 1 and board[k] > 0) or (player == -1 and board[k] < 0):
                                    is_furthest = False
                                    break
                            if is_furthest:
                                moves.append((i, 'off'))
                elif 0 <= target <= max_idx:
                    if self._is_open(board, target, player):
                        moves.append((i, target))
        return moves

    def _is_open(self, board, target, player):
        if board[target] == 0:
            return True
        if (player == 1 and board[target] > 0) or (player == -1 and board[target] < 0):
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

    def step_atomic(self, action):
        start, end = action
        self._apply_single_move_logic(self.board, self.bar, self.off, self.turn, start, end)
        
        die_used = 0
        if start == 'bar':
            die_used = Config.NUM_POINTS - end if self.turn == 1 else end + 1
        elif end == 'off':
            die_used = start + 1 if self.turn == 1 else Config.NUM_POINTS - start
            if die_used not in self.dice and self.dice:
                die_used = max(self.dice)
        else:
            die_used = abs(start - end)
        
        if die_used in self.dice:
            self.dice.remove(die_used)
        elif self.dice:
            self.dice.remove(max(self.dice))
        
        return self.check_win()

    def _apply_single_move_logic(self, board, bar, off, player, start, end):
        p_idx = 0 if player == 1 else 1
        
        if start == 'bar':
            bar[p_idx] -= 1
        else:
            board[start] += -1 if player == 1 else 1
        
        if end == 'off':
            off[p_idx] += 1
        else:
            if (player == 1 and board[end] == -1) or (player == -1 and board[end] == 1):
                board[end] = 0
                bar[1 if player == 1 else 0] += 1
            board[end] += 1 if player == 1 else -1

    def check_win(self):
        if self.off[0] == Config.CHECKERS_PER_PLAYER:
            return self._score_win(1)
        if self.off[1] == Config.CHECKERS_PER_PLAYER:
            return self._score_win(-1)
        return 0, 0

    def _score_win(self, winner):
        loser_idx = 0 if winner == -1 else 1
        mult = 1
        
        if self.off[loser_idx] == 0:
            mult = 2
            if self.bar[loser_idx] > 0:
                mult = 3
            else:
                loser = -winner
                home_range = range(0, Config.HOME_SIZE) if winner == 1 else range(Config.NUM_POINTS - Config.HOME_SIZE, Config.NUM_POINTS)
                for i in home_range:
                    if (loser == 1 and self.board[i] > 0) or (loser == -1 and self.board[i] < 0):
                        mult = 3
                        break
        return winner, mult

    def get_state_key(self):
        return (tuple(self.board), tuple(self.bar), tuple(self.off), self.turn, self.cube, tuple(self.dice))

    def get_vector(self, my_score=0, opp_score=0, device='cpu', canonical=True):
        """
        Generate observation tensor on target device.
        
        Args:
            my_score: Current player's match score
            opp_score: Opponent's match score
            device: Target device for tensors
            canonical: If True, always represent from current player's perspective
        
        Returns:
            Tuple of (board_tensor, context_tensor)
        """
        vec_data = [0] * Config.BOARD_SEQ_LEN
        
        if canonical and self.turn == -1:
            # Flip board for player -1 so they see it from their perspective
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

    def canonical_action_to_real(self, action):
        """Convert canonical action to real board coordinates."""
        if self.turn == 1:
            return action
        start, end = action
        new_start = 'bar' if start == 'bar' else (Config.NUM_POINTS - 1 - start)
        new_end = 'off' if end == 'off' else (Config.NUM_POINTS - 1 - end)
        return (new_start, new_end)

    def real_action_to_canonical(self, action):
        """Convert real board action to canonical coordinates."""
        if self.turn == 1:
            return action
        start, end = action
        new_start = 'bar' if start == 'bar' else (Config.NUM_POINTS - 1 - start)
        new_end = 'off' if end == 'off' else (Config.NUM_POINTS - 1 - end)
        return (new_start, new_end)
    
    def can_double(self):
        """Check if the current player is allowed to double."""
        # Cannot double if the game is over or if the opponent owns the cube
        if self.cube_owner != 0 and self.cube_owner != self.turn:
            return False
        return True

    def apply_double(self):
        """Increments cube and flips ownership, capped by MATCH_TARGET."""
        next_cube_value = self.cube * 2
        
        # Cap the cube at the match target to prevent runaway point values
        if next_cube_value > Config.MATCH_TARGET:
            self.cube = Config.MATCH_TARGET
        else:
            self.cube = next_cube_value
            
        self.cube_owner = -self.turn

    def handle_cube_refusal(self):
        """If a double is refused, the game ends immediately."""
        # Current player wins because the other person refused
        winner = self.turn
        points = self.cube # Current cube value before doubling
        return winner, points
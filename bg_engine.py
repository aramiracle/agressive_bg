import torch
import random
from config import Config

class BackgammonGame:
    def __init__(self):
        self.reset()

    def reset(self):
        # Native lists are faster than numpy for frequent index access in Python
        self.board = [0] * Config.NUM_POINTS
        for pos, count in Config.INITIAL_SETUP.items():
            self.board[pos] = count

        self.bar = [0, 0] 
        self.off = [0, 0]
        
        self.turn = 1
        self.cube = 1
        self.cube_owner = 0
        self.dice = []
        self.history = []
        
        return self.get_state_key()

    # --- OPTIMIZATION START: FAST SAVE/RESTORE ---
    def fast_save(self):
        """Returns a lightweight snapshot of the game state."""
        # List slicing [:] is highly optimized in CPython
        return (
            self.board[:],
            self.bar[:],
            self.off[:],
            self.turn,
            self.cube,
            self.dice[:]
        )

    def fast_restore(self, snapshot):
        """Restores state from snapshot without object overhead."""
        self.board[:] = snapshot[0]
        self.bar[:] = snapshot[1]
        self.off[:] = snapshot[2]
        self.turn = snapshot[3]
        self.cube = snapshot[4]
        self.dice[:] = snapshot[5]
    # --- OPTIMIZATION END ---

    def roll_dice(self):
        # Use native random for speed
        d1 = random.randint(1, Config.DICE_SIDES)
        d2 = random.randint(1, Config.DICE_SIDES)
        if d1 == d2:
            self.dice = [d1, d1, d1, d1]
        else:
            self.dice = [d1, d2]
        return self.dice

    def switch_turn(self):
        self.turn *= -1

    def get_legal_moves(self, dice_rolls=None):
        rolls = dice_rolls if dice_rolls is not None else self.dice
        if not rolls:
            return []

        moves = set()
        
        # Helper to find moves recursively
        def find_moves(current_board, current_bar, current_off, remaining_dice, path):
            if not remaining_dice:
                moves.add(tuple(sorted(path)))
                return

            die = remaining_dice[0]
            possible_moves = self._get_single_moves(current_board, current_bar, self.turn, die)
            
            if not possible_moves:
                moves.add(tuple(sorted(path)))
                return

            for start, end in possible_moves:
                # Shallow copy lists for recursion (faster than deepcopy)
                next_board = current_board[:]
                next_bar = current_bar[:]
                next_off = current_off[:]
                
                self._apply_single_move_logic(next_board, next_bar, next_off, self.turn, start, end)
                find_moves(next_board, next_bar, next_off, remaining_dice[1:], path + [(start, end)])

        unique_atomic_moves = set()
        unique_dice = set(rolls)
        
        # Optimization: Only calculate atomic moves (depth 1) for the policy head
        for die in unique_dice:
            single_moves = self._get_single_moves(self.board, self.bar, self.turn, die)
            for move in single_moves:
                unique_atomic_moves.add(move)
        
        return list(unique_atomic_moves)

    def _get_single_moves(self, board, bar, player, die):
        moves = []
        p_idx = 0 if player == 1 else 1
        max_idx = Config.NUM_POINTS - 1
        
        if bar[p_idx] > 0:
            target = (Config.NUM_POINTS - die) if player == 1 else (die - 1)
            if self._is_open(board, target, player):
                return [('bar', target)]
            return []

        iterator = range(max_idx, -1, -1) if player == 1 else range(0, Config.NUM_POINTS)
        can_bear_off = self._can_bear_off(board, bar, player)
        
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
        if board[target] == 0: return True
        if (player == 1 and board[target] > 0) or (player == -1 and board[target] < 0): return True
        if abs(board[target]) == 1: return True
        return False

    def _can_bear_off(self, board, bar, player):
        p_idx = 0 if player == 1 else 1
        if bar[p_idx] > 0: return False
        
        home_boundary = Config.HOME_SIZE
        opp_home_start = Config.NUM_POINTS - Config.HOME_SIZE
        
        if player == 1:
            for i in range(home_boundary, Config.NUM_POINTS):
                if board[i] > 0: return False
        else:
            for i in range(0, opp_home_start):
                if board[i] < 0: return False
        return True

    def step_atomic(self, action):
        start, end = action
        self._apply_single_move_logic(self.board, self.bar, self.off, self.turn, start, end)
        
        die_used = 0
        if start == 'bar':
            if self.turn == 1: die_used = Config.NUM_POINTS - end
            else: die_used = end + 1
        elif end == 'off':
            if self.turn == 1: die_used = start + 1
            else: die_used = Config.NUM_POINTS - start
            if die_used not in self.dice:
                die_used = max(self.dice) if self.dice else 0
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
            if player == 1: board[start] -= 1
            else: board[start] += 1
            
        if end == 'off':
            off[p_idx] += 1
        else:
            if (player == 1 and board[end] == -1):
                board[end] = 0
                bar[1] += 1
            elif (player == -1 and board[end] == 1):
                board[end] = 0
                bar[0] += 1
            
            if player == 1: board[end] += 1
            else: board[end] -= 1

    def check_win(self):
        if self.off[0] == Config.CHECKERS_PER_PLAYER:
            return self._score_win(1)
        if self.off[1] == Config.CHECKERS_PER_PLAYER:
            return self._score_win(-1)
        return 0, 0

    def _score_win(self, winner):
        loser = -1 * winner
        l_idx = 0 if loser == 1 else 1
        mult = 1
        if self.off[l_idx] == 0:
            mult = 2
            if self.bar[l_idx] > 0:
                mult = 3
            else:
                if winner == 1: home_range = range(0, Config.HOME_SIZE)
                else: home_range = range(Config.NUM_POINTS - Config.HOME_SIZE, Config.NUM_POINTS)
                for i in home_range:
                    if (loser == 1 and self.board[i] > 0) or (loser == -1 and self.board[i] < 0):
                        mult = 3
                        break
        return winner, mult

    def get_state_key(self):
        return (tuple(self.board), tuple(self.bar), tuple(self.off), self.turn, self.cube, tuple(self.dice))
    
    # --- MAJOR OPTIMIZATION: Direct Tensor Generation ---
    def get_vector(self, my_score, opp_score, device='cpu', canonical=True):
        """
        Generates the observation tensor directly on the target device.
        Eliminates numpy conversion overhead.
        """
        offset = Config.EMBED_OFFSET
        
        # Pre-allocate lists for vector construction (faster than appending to Tensor incrementally)
        # We need 28 board positions: 24 points + bar_self, bar_opp, off_self, off_opp
        vec_data = [0] * Config.BOARD_SEQ_LEN
        
        if canonical and self.turn == -1:
            # Flip perspective for P-1
            for i in range(Config.NUM_POINTS):
                vec_data[i] = -self.board[Config.NUM_POINTS - 1 - i] + offset
            
            vec_data[24] = self.bar[1] + offset
            vec_data[25] = -self.bar[0] + offset
            vec_data[26] = self.off[1] + offset
            vec_data[27] = -self.off[0] + offset
            
            ctx_data = [1.0, float(self.cube), float(my_score), float(opp_score)]
        else:
            # P1 perspective
            for i in range(Config.NUM_POINTS):
                vec_data[i] = self.board[i] + offset
                
            vec_data[24] = self.bar[0] + offset
            vec_data[25] = -self.bar[1] + offset
            vec_data[26] = self.off[0] + offset
            vec_data[27] = -self.off[1] + offset
            
            ctx_data = [float(self.turn), float(self.cube), float(my_score), float(opp_score)]
        
        # Convert to Tensor directly
        # 'long' for embedding lookup, 'float' for context
        t_board = torch.tensor(vec_data, dtype=torch.long, device=device)
        t_ctx = torch.tensor(ctx_data, dtype=torch.float, device=device)
        
        return t_board, t_ctx
    
    def canonical_action_to_real(self, action):
        if self.turn == 1: return action
        start, end = action
        new_start = 'bar' if start == 'bar' else (Config.NUM_POINTS - 1 - start)
        new_end = 'off' if end == 'off' else (Config.NUM_POINTS - 1 - end)
        return (new_start, new_end)
    
    def real_action_to_canonical(self, action):
        if self.turn == 1: return action
        start, end = action
        new_start = 'bar' if start == 'bar' else (Config.NUM_POINTS - 1 - start)
        new_end = 'off' if end == 'off' else (Config.NUM_POINTS - 1 - end)
        return (new_start, new_end)
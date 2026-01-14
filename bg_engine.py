import numpy as np
import copy
from config import Config

class BackgammonGame:
    def __init__(self):
        self.reset()

    def reset(self):
        # Board: 0 to NUM_POINTS-1
        # Player 1 moves high -> low. Player -1 moves low -> high.
        # Positive values = Player 1 checkers. Negative = Player -1.
        self.board = np.zeros(Config.NUM_POINTS, dtype=int)
        
        # Initial Setup (Standard Backgammon) from Config
        for pos, count in Config.INITIAL_SETUP.items():
            self.board[pos] = count

        # Bar and Off: [P1, P-1]
        self.bar = [0, 0] 
        self.off = [0, 0]
        
        self.turn = 1  # 1 or -1
        self.cube = 1
        self.cube_owner = 0  # 0=Middle, 1=P1, -1=P-1
        self.dice = []
        self.history = []
        
        return self.get_state_key()

    def roll_dice(self):
        d1, d2 = np.random.randint(1, Config.DICE_SIDES + 1, 2)
        if d1 == d2:
            self.dice = [d1, d1, d1, d1]
        else:
            self.dice = [d1, d2]
        return self.dice

    def switch_turn(self):
        self.turn *= -1

    def get_legal_moves(self, dice_rolls=None):
        """
        Returns a list of all valid board transitions.
        Each move is a list of tuples: [(from, to), (from, to)]
        """
        rolls = dice_rolls if dice_rolls is not None else self.dice
        if not rolls:
            return []

        moves = set()
        
        # Helper to find moves recursively
        def find_moves(current_board, current_bar, current_off, remaining_dice, path):
            if not remaining_dice:
                moves.add(tuple(sorted(path))) # Store as tuple to be hashable
                return

            die = remaining_dice[0]
            possible_moves = self._get_single_moves(current_board, current_bar, self.turn, die)
            
            if not possible_moves:
                # If can't move, path ends here
                moves.add(tuple(sorted(path)))
                return

            for start, end in possible_moves:
                # Apply move virtually
                next_board = current_board.copy()
                next_bar = current_bar.copy()
                next_off = current_off.copy()
                
                # Apply logic
                self._apply_single_move_logic(next_board, next_bar, next_off, self.turn, start, end)
                
                # Recurse
                find_moves(next_board, next_bar, next_off, remaining_dice[1:], path + [(start, end)])

        # For this RL implementation, we treat ONE checker move as one step.
        # We collect moves for ALL available dice values (deduped).
        unique_atomic_moves = set()
        
        # Get unique dice values to avoid duplicate moves for doubles
        unique_dice = set(rolls)
        
        for die in unique_dice:
            single_moves = self._get_single_moves(self.board, self.bar, self.turn, die)
            for move in single_moves:
                unique_atomic_moves.add(move)
        
        return list(unique_atomic_moves)

    def _get_single_moves(self, board, bar, player, die):
        """Get valid (start, end) pairs for a specific die."""
        moves = []
        p_idx = 0 if player == 1 else 1
        max_idx = Config.NUM_POINTS - 1  # 23
        
        # 1. Must move from bar if checkers exist
        if bar[p_idx] > 0:
            # P1 enters at high end (NUM_POINTS - die), P-1 enters at low end (die - 1)
            target = (Config.NUM_POINTS - die) if player == 1 else (die - 1)
            if self._is_open(board, target, player):
                return [('bar', target)]
            return []

        # 2. Regular board moves
        iterator = range(max_idx, -1, -1) if player == 1 else range(0, Config.NUM_POINTS)
        can_bear_off = self._can_bear_off(board, bar, player)
        
        for i in iterator:
            if (player == 1 and board[i] > 0) or (player == -1 and board[i] < 0):
                target = i - die if player == 1 else i + die
                
                # Bear off logic
                if (player == 1 and target < 0) or (player == -1 and target > max_idx):
                    if can_bear_off:
                        # Distance to bear off
                        dist = (i + 1) if player == 1 else (Config.NUM_POINTS - i)
                        if dist == die:
                            moves.append((i, 'off'))
                        elif dist < die:
                            # Can use higher die only if this is the furthest checker
                            is_furthest = True
                            behind_range = range(i + 1, Config.NUM_POINTS) if player == 1 else range(0, i)
                            for k in behind_range:
                                if (player == 1 and board[k] > 0) or (player == -1 and board[k] < 0):
                                    is_furthest = False
                                    break
                            if is_furthest:
                                moves.append((i, 'off'))
                
                # Normal move
                elif 0 <= target <= max_idx:
                    if self._is_open(board, target, player):
                        moves.append((i, target))
                        
        return moves

    def _is_open(self, board, target, player):
        # Open if empty, own color, or only 1 opponent (hit)
        if board[target] == 0: return True
        if (player == 1 and board[target] > 0) or (player == -1 and board[target] < 0): return True
        if abs(board[target]) == 1: return True # Hit
        return False

    def _can_bear_off(self, board, bar, player):
        """Check if player can bear off (all checkers in home board, none on bar)"""
        p_idx = 0 if player == 1 else 1
        
        # Cannot bear off if player has checkers on bar
        if bar[p_idx] > 0:
            return False
        
        # Check if all checkers are in home board
        # P1 home: indices 0 to HOME_SIZE-1. P-1 home: indices (NUM_POINTS-HOME_SIZE) to NUM_POINTS-1
        home_boundary = Config.HOME_SIZE  # 6
        opp_home_start = Config.NUM_POINTS - Config.HOME_SIZE  # 18
        
        if player == 1:
            for i in range(home_boundary, Config.NUM_POINTS):
                if board[i] > 0:
                    return False
        else:
            for i in range(0, opp_home_start):
                if board[i] < 0:
                    return False
        return True

    def step_atomic(self, action):
        """
        Executes one checker move.
        action: tuple (start, end) where start/end are indices, 'bar', or 'off'
        """
        start, end = action
        self._apply_single_move_logic(self.board, self.bar, self.off, self.turn, start, end)
        
        # Calculate which die was used
        die_used = 0
        if start == 'bar':
            # Bar entry: P1 enters at (NUM_POINTS - die), P-1 enters at (die - 1)
            if self.turn == 1:
                die_used = Config.NUM_POINTS - end
            else:
                die_used = end + 1
        elif end == 'off':
            # Bear off: distance from edge
            if self.turn == 1:
                die_used = start + 1
            else:
                die_used = Config.NUM_POINTS - start
            # Handle using larger die than distance
            if die_used not in self.dice:
                die_used = max(self.dice)
        else:
            die_used = abs(start - end)
        
        # Remove used die
        if die_used in self.dice:
            self.dice.remove(die_used)
        else:
            # Fallback for bear off with larger die
            self.dice.remove(max(self.dice))

        return self.check_win()

    def _apply_single_move_logic(self, board, bar, off, player, start, end):
        p_idx = 0 if player == 1 else 1
        
        # Remove from source
        if start == 'bar':
            bar[p_idx] -= 1
        else:
            if player == 1: board[start] -= 1
            else: board[start] += 1
            
        # Add to dest
        if end == 'off':
            off[p_idx] += 1
        else:
            # Check Hit
            if (player == 1 and board[end] == -1):
                board[end] = 0
                bar[1] += 1 # Send opponent to bar
            elif (player == -1 and board[end] == 1):
                board[end] = 0
                bar[0] += 1
            
            if player == 1: board[end] += 1
            else: board[end] -= 1

    def check_win(self):
        """Check if game is over. Returns (winner, win_type) or (0, 0) if ongoing."""
        # 1=Normal, 2=Gammon, 3=Backgammon
        if self.off[0] == Config.CHECKERS_PER_PLAYER:
            return self._score_win(1)
        if self.off[1] == Config.CHECKERS_PER_PLAYER:
            return self._score_win(-1)
        return 0, 0

    def _score_win(self, winner):
        """Calculate win type: 1=Normal, 2=Gammon, 3=Backgammon"""
        loser = -1 * winner
        l_idx = 0 if loser == 1 else 1
        
        mult = 1
        if self.off[l_idx] == 0:
            mult = 2  # Gammon
            # Backgammon check (loser has checker on winner's home or bar)
            if self.bar[l_idx] > 0:
                mult = 3
            else:
                # Check winner's home board for loser checkers
                if winner == 1:
                    home_range = range(0, Config.HOME_SIZE)
                else:
                    home_range = range(Config.NUM_POINTS - Config.HOME_SIZE, Config.NUM_POINTS)
                    
                for i in home_range:
                    if (loser == 1 and self.board[i] > 0) or (loser == -1 and self.board[i] < 0):
                        mult = 3
                        break
        
        return winner, mult

    def get_state_key(self):
        return (tuple(self.board), tuple(self.bar), tuple(self.off), self.turn, self.cube, tuple(self.dice))
    
    def get_vector(self, my_score, opp_score, canonical=True):
        """
        Convert state to Tensor for Neural Net.
        
        Args:
            my_score: Current player's match score
            opp_score: Opponent's match score
            canonical: If True, always return state from current player's perspective
                      (current player's checkers are positive, moving high→low)
        
        Returns:
            (board_vector, context_vector) as numpy arrays
        """
        offset = Config.EMBED_OFFSET
        
        if canonical and self.turn == -1:
            # Flip perspective: P-1 sees board as if they were P1
            # 1. Reverse board indices (pos 0 ↔ pos 23)
            # 2. Negate values (P-1's checkers become positive)
            vec = []
            for i in range(Config.NUM_POINTS - 1, -1, -1):
                vec.append(-self.board[i] + offset)
            
            # Bar: current player first (P-1), then opponent (P1)
            vec.append(self.bar[1] + offset)   # P-1's bar (now positive)
            vec.append(-self.bar[0] + offset)  # P1's bar (now negative)
            
            # Off: current player first
            vec.append(self.off[1] + offset)   # P-1's off (now positive)
            vec.append(-self.off[0] + offset)  # P1's off (now negative)
            
            # Context: Turn is always 1 from canonical view
            ctx = [1, self.cube, my_score, opp_score]
        else:
            # Normal view (P1's perspective or non-canonical)
            vec = []
            for x in self.board:
                vec.append(x + offset)
            
            vec.append(self.bar[0] + offset)
            vec.append(-self.bar[1] + offset)
            vec.append(self.off[0] + offset)
            vec.append(-self.off[1] + offset)
            
            ctx = [self.turn, self.cube, my_score, opp_score]
        
        return np.array(vec, dtype=int), np.array(ctx, dtype=float)
    
    def canonical_action_to_real(self, action):
        """
        Convert a canonical action back to real board coordinates.
        Only needed when current player is P-1.
        
        Args:
            action: (start, end) in canonical coordinates
            
        Returns:
            (start, end) in real board coordinates
        """
        if self.turn == 1:
            return action  # No conversion needed
        
        start, end = action
        
        # Flip coordinates for P-1
        if start == 'bar':
            new_start = 'bar'
        else:
            new_start = Config.NUM_POINTS - 1 - start
        
        if end == 'off':
            new_end = 'off'
        else:
            new_end = Config.NUM_POINTS - 1 - end
        
        return (new_start, new_end)
    
    def real_action_to_canonical(self, action):
        """
        Convert a real action to canonical coordinates.
        Only needed when current player is P-1.
        
        Args:
            action: (start, end) in real board coordinates
            
        Returns:
            (start, end) in canonical coordinates
        """
        if self.turn == 1:
            return action  # No conversion needed
        
        start, end = action
        
        # Flip coordinates for P-1
        if start == 'bar':
            new_start = 'bar'
        else:
            new_start = Config.NUM_POINTS - 1 - start
        
        if end == 'off':
            new_end = 'off'
        else:
            new_end = Config.NUM_POINTS - 1 - end
        
        return (new_start, new_end)
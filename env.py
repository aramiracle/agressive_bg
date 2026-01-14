import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import Config
from bg_engine import BackgammonGame

class BackgammonEnv(gym.Env):
    """
    Backgammon environment compatible with Gymnasium API.
    
    Observation: Tuple of (board_vector, context_vector)
    Action: Tuple (start, end) where start/end are board indices, 'bar', or 'off'
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, match_play=True):
        super().__init__()
        self.engine = BackgammonGame()
        self.match_play = match_play
        self.scores = [0, 0]  # Match scores for P1 and P-1
        
        # Observation space: BOARD_SEQ_LEN positions (0 to EMBED_VOCAB_SIZE-1) + CONTEXT_SIZE values
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=Config.EMBED_VOCAB_SIZE - 1, 
                               shape=(Config.BOARD_SEQ_LEN,), dtype=np.int32),
            "context": spaces.Box(low=-Config.CHECKERS_PER_PLAYER, high=Config.CHECKERS_PER_PLAYER, 
                                 shape=(Config.CONTEXT_SIZE,), dtype=np.float32)
        })
        
        # Action space: (from, to) pairs using NUM_ACTIONS indices
        self.action_space = spaces.MultiDiscrete([Config.NUM_ACTIONS, Config.NUM_ACTIONS])
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.engine.reset()
        if not self.match_play:
            self.scores = [0, 0]
        
        obs = self._get_obs()
        info = {"legal_moves": self.engine.get_legal_moves()}
        return obs, info
    
    def _get_obs(self):
        """Get current observation."""
        board_vec, ctx_vec = self.engine.get_vector(self.scores[0], self.scores[1])
        return {
            "board": board_vec.astype(np.int32),
            "context": ctx_vec.astype(np.float32)
        }
        
    def step(self, action):
        """
        Execute one atomic move.
        
        Args:
            action: tuple (start, end) where:
                - start: 0-23 (board index) or 'bar'
                - end: 0-23 (board index) or 'off'
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action indices back to game format if needed
        if isinstance(action, (list, np.ndarray)):
            start = 'bar' if action[0] == Config.BAR_IDX else int(action[0])
            end = 'off' if action[1] == Config.OFF_IDX else int(action[1])
            action = (start, end)
        
        # Check if action is legal
        legal_moves = self.engine.get_legal_moves()
        if action not in legal_moves:
            # Invalid action - return negative reward, don't change state
            obs = self._get_obs()
            return obs, -1.0, False, False, {"error": "illegal_move", "legal_moves": legal_moves}
        
        # Execute move
        winner, win_type = self.engine.step_atomic(action)
        
        terminated = winner != 0
        truncated = False
        reward = 0.0
        info = {"legal_moves": self.engine.get_legal_moves(), "dice": self.engine.dice.copy()}
        
        if terminated:
            # Calculate reward based on win type
            base_points = Config.R_WIN
            if win_type == 2:
                base_points = Config.R_GAMMON
            elif win_type == 3:
                base_points = Config.R_BACKGAMMON
            
            # Apply cube multiplier
            total_points = base_points * self.engine.cube
            
            if self.match_play:
                # Match Play: Cap points at what's needed to win
                player_idx = 0 if winner == 1 else 1
                points_needed = Config.MATCH_TARGET - self.scores[player_idx]
                final_points = min(total_points, points_needed)
                self.scores[player_idx] += final_points
                
                # Check if match is over
                info["match_over"] = max(self.scores) >= Config.MATCH_TARGET
                info["scores"] = self.scores.copy()
            else:
                final_points = total_points
            
            # Reward from P1's perspective
            reward = final_points if winner == 1 else -final_points
            info["winner"] = winner
            info["win_type"] = win_type
                
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
    def get_legal_moves(self):
        """Get list of legal moves in current state."""
        return self.engine.get_legal_moves()
    
    def roll_dice(self):
        """Roll dice for current turn."""
        return self.engine.roll_dice()
    
    def switch_turn(self):
        """Switch to other player's turn."""
        self.engine.switch_turn()
    
    def render(self):
        """Render the current board state."""
        board = self.engine.board
        bar = self.engine.bar
        off = self.engine.off
        
        print("\n" + "=" * 50)
        print(f"Turn: {'P1 (White)' if self.engine.turn == 1 else 'P-1 (Black)'}")
        print(f"Dice: {self.engine.dice}")
        print(f"Cube: {self.engine.cube}")
        print(f"Scores: P1={self.scores[0]}, P-1={self.scores[1]}")
        print("-" * 50)
        
        # Top row (12-23)
        print("  12 13 14 15 16 17 | 18 19 20 21 22 23")
        top = [f"{board[i]:+3d}" for i in range(12, 24)]
        print(" ".join(top[:6]) + " |" + " ".join(top[6:]))
        
        print("\n" + " " * 20 + f"BAR: P1={bar[0]}, P-1={bar[1]}")
        print(" " * 20 + f"OFF: P1={off[0]}, P-1={off[1]}\n")
        
        # Bottom row (11-0)
        print("  11 10  9  8  7  6 |  5  4  3  2  1  0")
        bot = [f"{board[i]:+3d}" for i in range(11, -1, -1)]
        print(" ".join(bot[:6]) + " |" + " ".join(bot[6:]))
        print("=" * 50)
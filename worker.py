import torch
import random
from collections import deque
from config import Config
from bg_engine import BackgammonGame
from model import get_model
from mcts import MCTS
from utils import move_to_indices

class SelfPlayWorker:
    """CPU-optimized self-play worker using batched MCTS."""
    
    def __init__(self, model_state_dict, worker_id=0):
        self.worker_id = worker_id
        # Use CPU for workers to leave GPU for trainer, or as configured
        self.device = torch.device(Config.WORKER_DEVICE)
        
        # Initialize and optimize model
        self.model = get_model().to(self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        
        # Optimization: Disable gradient tracking for all worker operations
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Initialize the optimized MCTS
        self.mcts = MCTS(self.model, self.device)
        
        # Game instance
        self.game = BackgammonGame()
        
    def update_model(self, state_dict):
        """Update worker's model weights dynamically."""
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def play_match(self):
        """Plays a full match up to MATCH_TARGET points."""
        match_data = []
        scores = [0, 0] # [Player 1, Player 2]
        
        while max(scores) < Config.MATCH_TARGET:
            game_result_data, winner_id, points = self._play_single_game(scores)
            
            # Update match scores
            if winner_id == 1:
                scores[0] += points
            else:
                scores[1] += points
            
            # Collect data from this game
            match_data.extend(game_result_data)
            
            # Safety break for infinite loops in early training
            if len(match_data) > Config.MAX_TURNS: 
                break
                
        return self._process_match_results(match_data, scores), scores

    def _play_single_game(self, match_scores):
        """Plays a single game and returns raw step data."""
        self.game.reset()
        game_steps = []
        
        while True:
            winner, win_type = self.game.check_win()
            if winner != 0:
                return game_steps, winner, self._calculate_points(win_type)
            
            self.game.roll_dice()
            
            # A turn can consist of multiple atomic moves
            while self.game.dice:
                legal_moves = self.game.get_legal_moves()
                if not legal_moves:
                    break
                
                # MCTS Search
                # Fixed: passing self.game as the engine (not None)
                root = self.mcts.search(self.game, match_scores[0], match_scores[1])
                
                # Get visit counts for policy
                if not root.children:
                    # Fallback if MCTS fails to expand (should not happen with legal moves)
                    chosen_act = random.choice(legal_moves)
                else:
                    # Select action based on visit count distribution (Stochastic Play)
                    acts, visits = zip(*[(a, c.visits) for a, c in root.children.items()])
                    chosen_act = random.choices(acts, weights=visits, k=1)[0]
                
                # Record state BEFORE moving (Canonical perspective)
                b_t, c_t = self.game.get_vector(
                    match_scores[0], match_scores[1], 
                    device=self.device, 
                    canonical=True
                )
                
                # Move to CPU immediately to save worker memory
                b_t = b_t.squeeze(0).cpu()
                c_t = c_t.squeeze(0).cpu()
                
                # Convert action to indices for the policy head
                canonical_act = self.game.real_action_to_canonical(chosen_act)
                s_idx, e_idx = move_to_indices(canonical_act[0], canonical_act[1])
                
                # Store: (Board, Context, (Start, End), TurnID)
                game_steps.append([b_t, c_t, (s_idx, e_idx), self.game.turn])
                
                # Execute move
                self.game.step_atomic(chosen_act)
                
                # Check if this atomic move ended the game
                if self.game.check_win()[0] != 0:
                    break
            
            # Turn over
            if self.game.check_win()[0] == 0:
                self.game.switch_turn()

    def _calculate_points(self, win_type):
        """Calculates score based on gammon/backgammon multipliers."""
        multiplier = 1
        if win_type == 2: multiplier = Config.R_GAMMON
        if win_type == 3: multiplier = Config.R_BACKGAMMON
        return int(multiplier * self.game.cube)

    def _process_match_results(self, match_data, final_scores):
        """
        Assigns rewards based on the final match winner.
        Uses match-level reinforcement: Was the move helpful to win the MATCH?
        """
        match_winner = 1 if final_scores[0] > final_scores[1] else -1
        processed = []
        
        for board, context, action, turn_id in match_data:
            # Reward is 1.0 if the player who made the move eventually won the match
            reward = 1.0 if turn_id == match_winner else -1.0
            processed.append((board, context, action, reward))
            
        return processed
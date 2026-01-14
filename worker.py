"""Ray self-play worker for generating training data."""

import ray
import numpy as np
from config import Config
from bg_engine import BackgammonGame
from model import BackgammonTransformer
from mcts import MCTS
from utils import move_to_indices


@ray.remote
class SelfPlayWorker:
    """
    Ray actor for parallel self-play data generation.
    
    Each worker maintains its own copy of the game engine, model, and MCTS.
    Workers play full matches and return training data.
    """
    
    def __init__(self, model_state):
        """
        Initialize the self-play worker.
        
        Args:
            model_state: Initial model state dict
        """
        self.game = BackgammonGame()
        self.model = BackgammonTransformer().to(Config.DEVICE)
        self.model.load_state_dict(model_state)
        self.model.eval()
        self.mcts = MCTS(self.model)
        
    def update_model(self, state_dict):
        """
        Update the worker's model with new weights.
        
        Args:
            state_dict: New model state dict
        """
        self.model.load_state_dict(state_dict)

    def play_match(self):
        """
        Play a full match and collect training data.
        
        Returns:
            Tuple of (training_data, final_scores)
            - training_data: List of (board_vec, context_vec, action, reward)
            - final_scores: [P1_score, P-1_score]
        """
        data = []
        scores = [0, 0]  # P1, P-1
        
        while max(scores) < Config.MATCH_TARGET:
            game_data = self._play_single_game(scores)
            data.extend(game_data)
            
            # Check for winner after game ends
            winner, win_type = self.game.check_win()
            if winner != 0:
                pts = self._calculate_points(win_type)
                if winner == 1:
                    scores[0] += pts
                else:
                    scores[1] += pts
        
        # Process data with final rewards
        processed_data = self._process_game_data(data, scores)
        return processed_data, scores
    
    def _play_single_game(self, scores):
        """Play a single game within a match."""
        self.game.reset()
        game_data = []
        
        while True:
            winner, _ = self.game.check_win()
            if winner != 0:
                return game_data
            
            self.game.roll_dice()
            
            while self.game.dice:
                legal_moves = self.game.get_legal_moves()
                if not legal_moves:
                    self.game.dice = []
                    break
                
                # MCTS search
                root = self.mcts.search(None, self.game, scores[0], scores[1])
                
                # Select action based on visit counts
                visits = {act: child.visits for act, child in root.children.items()}
                total = sum(visits.values())
                if total == 0:
                    break
                
                # Probabilistic selection for exploration
                acts = list(visits.keys())
                probs = [v / total for v in visits.values()]
                act_idx = np.random.choice(len(acts), p=probs)
                chosen_act = acts[act_idx]
                
                # Store training data with canonical representation
                # The model always sees the board from current player's perspective
                b_vec, c_vec = self.game.get_vector(scores[0], scores[1], canonical=True)
                
                # Convert action to canonical coordinates for training
                canonical_act = self.game.real_action_to_canonical(chosen_act)
                s, e = move_to_indices(canonical_act[0], canonical_act[1])
                
                # Store canonical state and action (turn is always 1 in canonical view)
                game_data.append((b_vec, c_vec, (s, e), self.game.turn))
                
                # Execute move
                w, _ = self.game.step_atomic(chosen_act)
                if w != 0:
                    break
            
            if self.game.check_win()[0] != 0:
                continue
            self.game.switch_turn()
        
        return game_data
    
    def _calculate_points(self, win_type):
        """Calculate points based on win type."""
        pts = 1
        if win_type == 2:
            pts = int(Config.R_GAMMON)
        if win_type == 3:
            pts = int(Config.R_BACKGAMMON)
        return int(pts * self.game.cube)
    
    def _process_game_data(self, data, scores):
        """Process raw game data into training format with rewards."""
        winner = 1 if scores[0] > scores[1] else -1
        pts = abs(scores[0] - scores[1])
        final_reward = pts if winner == 1 else -pts
        
        processed_data = []
        for state, context, action, turn in data:
            # Reward from perspective of the player who made this move
            reward = final_reward if turn == 1 else -final_reward
            processed_data.append((state, context, action, reward))
        
        return processed_data


import ray
import torch
import random
from config import Config
from bg_engine import BackgammonGame
from model import get_model
from mcts import MCTS
from utils import move_to_indices

@ray.remote
class SelfPlayWorker:
    def __init__(self, model_state):
        self.game = BackgammonGame()
        
        # Set device for this worker (Likely CPU to save GPU for training)
        self.device = torch.device(Config.WORKER_DEVICE)
        
        self.model = get_model().to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()
        
        # MCTS now takes device arg
        self.mcts = MCTS(self.model, self.device)
        
    def update_model(self, state_dict):
        # Load state dict (CPU->CPU copy is fast)
        self.model.load_state_dict(state_dict)

    def play_match(self):
        data = []
        scores = [0, 0]
        total_turns = 0
        
        while max(scores) < Config.MATCH_TARGET and total_turns < Config.MAX_TURNS:
            game_data = self._play_single_game(scores)
            data.extend(game_data)
            
            winner, win_type = self.game.check_win()
            if winner != 0:
                pts = self._calculate_points(win_type)
                if winner == 1: scores[0] += pts
                else: scores[1] += pts
            total_turns += len(game_data)
            
        return self._process_game_data(data, scores), scores
    
    def _play_single_game(self, scores):
        self.game.reset()
        game_data = []
        
        while True:
            winner, _ = self.game.check_win()
            if winner != 0: return game_data
            
            self.game.roll_dice()
            
            while self.game.dice:
                legal_moves = self.game.get_legal_moves()
                if not legal_moves:
                    self.game.dice = []
                    break
                
                root = self.mcts.search(None, self.game, scores[0], scores[1])
                
                visits = [(act, child.visits) for act, child in root.children.items()]
                if not visits:
                    chosen_act = legal_moves[0]
                else:
                    acts, weights = zip(*visits)
                    # Python random.choices is fast for small lists
                    chosen_act = random.choices(acts, weights=weights, k=1)[0]
                
                # Get state tensors directly (On CPU)
                b_t, c_t = self.game.get_vector(scores[0], scores[1], device=self.device, canonical=True)
                
                canonical_act = self.game.real_action_to_canonical(chosen_act)
                s, e = move_to_indices(canonical_act[0], canonical_act[1])
                
                # Store tensors and integers. No numpy.
                game_data.append((b_t, c_t, (s, e), self.game.turn))
                
                w, _ = self.game.step_atomic(chosen_act)
                if w != 0: break
            
            if self.game.check_win()[0] != 0: continue
            self.game.switch_turn()
        
        return game_data
    
    def _calculate_points(self, win_type):
        pts = 1
        if win_type == 2: pts = int(Config.R_GAMMON)
        if win_type == 3: pts = int(Config.R_BACKGAMMON)
        return int(pts * self.game.cube)
    
    def _process_game_data(self, data, scores):
        winner = 1 if scores[0] > scores[1] else -1
        pts = float(abs(scores[0] - scores[1]))
        final_reward = pts if winner == 1 else -pts
        
        processed = []
        for state, context, action, turn in data:
            perspective_reward = final_reward * turn
            # Returns: (Tensor, Tensor, Tuple, float)
            processed.append((state, context, action, perspective_reward))
        return processed
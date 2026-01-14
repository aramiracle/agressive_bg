"""ELO rating system and model evaluation."""

from tqdm import tqdm
from config import Config
from mcts import MCTS


def calculate_expected_score(player_elo, opponent_elo):
    """
    Calculate expected score using ELO formula.
    
    Args:
        player_elo: Player's current ELO rating
        opponent_elo: Opponent's current ELO rating
        
    Returns:
        Expected score (0-1)
    """
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / Config.ELO_SCALE))


def update_elo(current_elo, opponent_elo, actual_score):
    """
    Update ELO rating based on match result.
    
    Args:
        current_elo: Player's current ELO rating
        opponent_elo: Opponent's current ELO rating
        actual_score: Actual result (1=win, 0.5=draw, 0=loss)
        
    Returns:
        New ELO rating
    """
    expected = calculate_expected_score(current_elo, opponent_elo)
    return current_elo + Config.ELO_K * (actual_score - expected)


def play_single_game(game, current_mcts, opponent_mcts, current_plays_white, current_score, opponent_score):
    """
    Play a single game between two models.
    
    Args:
        game: BackgammonGame instance
        current_mcts: MCTS for current model
        opponent_mcts: MCTS for opponent model
        current_plays_white: True if current model plays as P1 (white)
        current_score: Current model's match score
        opponent_score: Opponent model's match score
        
    Returns:
        Points scored (positive for current model win, negative for opponent win)
    """
    game.reset()
    
    while True:
        winner, win_type = game.check_win()
        if winner != 0:
            pts = 1
            if win_type == 2:
                pts = int(Config.R_GAMMON)
            if win_type == 3:
                pts = int(Config.R_BACKGAMMON)
            pts = int(pts * game.cube)
            
            # Determine who won
            if current_plays_white:
                return pts if winner == 1 else -pts
            else:
                return pts if winner == -1 else -pts
        
        game.roll_dice()
        
        while game.dice:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                game.dice = []
                break
            
            # Determine which model plays
            current_model_turn = (game.turn == 1 and current_plays_white) or \
                                (game.turn == -1 and not current_plays_white)
            
            if current_model_turn:
                root = current_mcts.search(None, game, current_score, opponent_score)
            else:
                root = opponent_mcts.search(None, game, opponent_score, current_score)
            
            visits = {act: child.visits for act, child in root.children.items()}
            total = sum(visits.values())
            if total == 0:
                chosen_act = legal_moves[0]
            else:
                chosen_act = max(visits, key=visits.get)
            
            w, _ = game.step_atomic(chosen_act)
            if w != 0:
                break
        
        if game.check_win()[0] != 0:
            continue
        game.switch_turn()


def play_match(game, current_mcts, opponent_mcts):
    """
    Play a full match (to MATCH_TARGET points) between two models.
    Players alternate colors each game within the match.
    
    Args:
        game: BackgammonGame instance
        current_mcts: MCTS for current model
        opponent_mcts: MCTS for opponent model
        
    Returns:
        True if current model won the match
    """
    current_score = 0
    opponent_score = 0
    game_count = 0
    
    while current_score < Config.MATCH_TARGET and opponent_score < Config.MATCH_TARGET:
        # Alternate colors each game
        current_plays_white = (game_count % 2 == 0)
        game_count += 1
        
        result = play_single_game(
            game, current_mcts, opponent_mcts,
            current_plays_white, current_score, opponent_score
        )
        
        if result > 0:
            current_score += result
        else:
            opponent_score += abs(result)
    
    return current_score >= Config.MATCH_TARGET


def evaluate_vs_opponent(game, current_model, opponent_model, num_games=None, show_progress=True):
    """
    Evaluate current model against opponent model.
    
    Args:
        game: BackgammonGame instance
        current_model: Current neural network model
        opponent_model: Opponent neural network model
        num_games: Number of matches to play (default: Config.ELO_EVAL_GAMES)
        show_progress: Whether to show progress bar
        
    Returns:
        Win rate of current model (0-1)
    """
    if num_games is None:
        num_games = Config.ELO_EVAL_GAMES
    
    current_mcts = MCTS(current_model)
    opponent_mcts = MCTS(opponent_model)
    
    wins = 0
    game_iter = tqdm(range(num_games), desc="  ⚔️  ELO Eval", leave=False) if show_progress else range(num_games)
    
    for _ in game_iter:
        if play_match(game, current_mcts, opponent_mcts):
            wins += 1
    
    return wins / num_games


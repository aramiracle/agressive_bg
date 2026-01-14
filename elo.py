"""ELO rating system and model evaluation with proper game-level tracking."""

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


def update_elo(current_elo, opponent_elo, actual_score, num_games=1):
    """
    Update ELO rating based on game results.
    
    Args:
        current_elo: Player's current ELO rating
        opponent_elo: Opponent's current ELO rating
        actual_score: Actual win rate (0-1) across all games
        num_games: Number of games played
        
    Returns:
        New ELO rating
    """
    expected = calculate_expected_score(current_elo, opponent_elo)
    # Scale K-factor by number of games for more stable updates
    effective_k = Config.ELO_K * num_games
    return current_elo + effective_k * (actual_score - expected)


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
        Tuple of (match_won, games_won, total_games)
    """
    current_score = 0
    opponent_score = 0
    game_count = 0
    games_won = 0
    
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
            games_won += 1
        else:
            opponent_score += abs(result)
    
    match_won = current_score >= Config.MATCH_TARGET
    return match_won, games_won, game_count


def evaluate_vs_opponent(game, current_model, opponent_model, num_matches=None, show_progress=True):
    """
    Evaluate current model against opponent model using game-level statistics.
    
    Args:
        game: BackgammonGame instance
        current_model: Current neural network model
        opponent_model: Opponent neural network model
        num_matches: Number of matches to play (default: Config.ELO_EVAL_GAMES)
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (win_rate, total_games_played)
    """
    if num_matches is None:
        num_matches = Config.ELO_EVAL_GAMES
    
    current_mcts = MCTS(current_model, Config.DEVICE)
    opponent_mcts = MCTS(opponent_model, Config.DEVICE)
    
    total_games_won = 0
    total_games_played = 0
    matches_won = 0
    
    game_iter = tqdm(range(num_matches), desc="  ⚔️  ELO Eval", leave=False) if show_progress else range(num_matches)
    
    for _ in game_iter:
        match_won, games_won, games_played = play_match(game, current_mcts, opponent_mcts)
        
        if match_won:
            matches_won += 1
        
        total_games_won += games_won
        total_games_played += games_played
        
        # Update progress bar with current stats
        if show_progress:
            game_iter.set_postfix({
                'Matches': f'{matches_won}/{_+1}',
                'Games': f'{total_games_won}/{total_games_played}'
            })
    
    game_win_rate = total_games_won / total_games_played if total_games_played > 0 else 0.5
    
    return game_win_rate, total_games_played
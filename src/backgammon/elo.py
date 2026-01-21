"""ELO rating system for model evaluation."""

from tqdm import tqdm
import torch
from src.backgammon.config import Config
from src.backgammon.mcts import MCTS

def calculate_expected_score(player_elo, opponent_elo):
    """Calculate expected score using standard ELO formula."""
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / Config.ELO_SCALE))

def update_elo(current_elo, opponent_elo, wins, total_games):
    """Update ELO based on match results."""
    if total_games == 0:
        return current_elo

    actual = wins / total_games
    expected = calculate_expected_score(current_elo, opponent_elo)
    delta = Config.ELO_K * (actual - expected)

    delta = max(-Config.ELO_SCALE, min(Config.ELO_SCALE, delta))
    return current_elo + delta * total_games

def play_single_game(game, mcts_a, mcts_b, a_is_white, max_moves=500):
    """
    Play a single game between two MCTS agents.
    """
    game.reset()
    mcts_a.reset()
    mcts_b.reset()
    
    move_count = 0

    while move_count < max_moves:
        winner, _ = game.check_win()
        if winner != 0:
            if a_is_white:
                return 1.0 if winner == 1 else 0.0
            else:
                return 1.0 if winner == -1 else 0.0

        game.roll_dice()

        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                break

            a_turn = (game.turn == 1 and a_is_white) or (game.turn == -1 and not a_is_white)
            mcts = mcts_a if a_turn else mcts_b
            
            # Search using current game state (MCTS internally copies)
            root = mcts.search(game, 0, 0)

            if root.children:
                action = max(root.children.items(), key=lambda x: x[1].visits)[0]
            else:
                action = legal[0]

            game.step_atomic(action)
            mcts.advance_to_child(action)
            move_count += 1
            
            if game.check_win()[0] != 0:
                break

        if game.check_win()[0] == 0:
            game.switch_turn()
            # Dice roll changes randomness, so old trees are invalid for new turn
            mcts_a.reset()
            mcts_b.reset()

    return 0.5 # Draw/Timeout

def evaluate_vs_opponent(game, model_a, model_b, num_games, device, show_progress=False):
    """Evaluate model_a against model_b."""
    mcts_a = MCTS(model_a, device=device)
    mcts_b = MCTS(model_b, device=device)

    wins = 0.0
    games_played = 0
    
    iterable = range(num_games)
    if show_progress:
        iterable = tqdm(iterable, desc="ELO Evaluation", dynamic_ncols=True, leave=False)

    for i in iterable:
        a_is_white = (i % 2 == 0)
        result = play_single_game(game, mcts_a, mcts_b, a_is_white)
        wins += result
        games_played += 1
        
        if show_progress:
            iterable.set_postfix({'Wins': f'{wins:.1f}/{games_played}'})

    return wins, games_played
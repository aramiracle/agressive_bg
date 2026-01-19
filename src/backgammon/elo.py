"""ELO rating system for model evaluation."""

from tqdm import tqdm
from src.backgammon.config import Config
from src.backgammon.mcts import MCTS


def calculate_expected_score(player_elo, opponent_elo):
    """Calculate expected score using standard ELO formula."""
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / Config.ELO_SCALE))


def update_elo(current_elo, opponent_elo, wins, total_games):
    """
    Standard ELO update for the evaluated player using a batch average.
    K is interpreted as per-batch scaling (typical default: 16 or 24).
    """
    if total_games == 0:
        return current_elo

    actual = wins / total_games
    expected = calculate_expected_score(current_elo, opponent_elo)
    delta = Config.ELO_K * (actual - expected)

    # clamp delta to prevent extreme swings
    delta = max(-Config.ELO_SCALE, min(Config.ELO_SCALE, delta))
    return current_elo + delta * total_games

def play_single_game(game, mcts_a, mcts_b, a_is_white, max_moves=500):
    """
    Play a single game between two MCTS agents.
    
    Args:
        game: BackgammonGame instance
        mcts_a: MCTS for model A
        mcts_b: MCTS for model B
        a_is_white: True if model A plays as white (turn=1)
        max_moves: Maximum moves before timeout
    
    Returns:
        1.0 if model_a wins, 0.0 if model_b wins, 0.5 for timeout/draw
    """
    game.reset()
    move_count = 0

    while move_count < max_moves:
        # Check for winner first
        winner, mult = game.check_win()
        if winner != 0:
            if a_is_white:
                return 1.0 if winner == 1 else 0.0
            else:
                return 1.0 if winner == -1 else 0.0

        # Roll dice for current turn
        game.roll_dice()

        # Execute all moves for this roll
        while game.dice:
            legal = game.get_legal_moves()
            if not legal:
                break

            # Determine whose turn it is
            a_turn = (game.turn == 1 and a_is_white) or (game.turn == -1 and not a_is_white)
            mcts = mcts_a if a_turn else mcts_b
            
            # Run MCTS search
            root = mcts.search(game, 0, 0)

            # Select best action by visit count
            if root.children:
                action = max(root.children.items(), key=lambda x: x[1].visits)[0]
            else:
                action = legal[0]

            game.step_atomic(action)
            move_count += 1

        # Switch turn if game not over
        if game.check_win()[0] == 0:
            game.switch_turn()

    # Timeout reached - treat as draw
    return 0.5


def evaluate_vs_opponent(game, model_a, model_b, num_games, device, show_progress=False):
    """
    Evaluate model_a against model_b over multiple games.
    
    Alternates colors to ensure fairness.
    
    Args:
        game: BackgammonGame instance (will be reset each game)
        model_a: First model to evaluate
        model_b: Second model (typically best known model)
        num_games: Number of games to play
        device: Device for inference
        show_progress: Whether to show tqdm progress bar
    
    Returns:
        Tuple of (wins_for_model_a, total_games_played)
    """
    mcts_a = MCTS(model_a, device)
    mcts_b = MCTS(model_b, device)

    wins = 0.0
    games_played = 0
    
    iterable = range(num_games)
    if show_progress:
        iterable = tqdm(iterable, desc="ELO Evaluation", dynamic_ncols=True, leave=False)

    for i in iterable:
        # Alternate who plays white for fairness
        a_is_white = (i % 2 == 0)
        result = play_single_game(game, mcts_a, mcts_b, a_is_white)
        wins += result
        games_played += 1
        
        if show_progress:
            iterable.set_postfix({'Wins': f'{wins:.1f}/{games_played}'})

    return wins, games_played


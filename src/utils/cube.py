import torch
import random
from src.config import Config


def compute_cube_features(game_equity, game, my_score, opp_score, equity_table, is_take=False):
    """
    Compute match-equity-aware features for the cube decision.
    """
    cube_level = getattr(game, 'cube_value', 1)
    target     = Config.MATCH_TARGET

    pts_win  = cube_level * 2
    pts_lose = cube_level * 2
    pts_drop = cube_level

    # 1. Back out the raw game winning probability from the value head's match equity.
    # The value head predicts E[match_equity] = p_win * me_win(current_cube) + (1-p_win) * me_lose(current_cube)
    me_win_current  = equity_table.get_equity(min(my_score + cube_level, target), opp_score)
    me_lose_current = equity_table.get_equity(my_score, min(opp_score + cube_level, target))
    
    denom = max(me_win_current - me_lose_current, 1e-6)
    p_win_game = (game_equity - me_lose_current) / denom
    p_win_game = max(0.0, min(1.0, p_win_game))

    # 2. Evaluate outcomes at the doubled stakes
    me_win_double  = equity_table.get_equity(min(my_score + pts_win, target), opp_score)
    me_lose_double = equity_table.get_equity(my_score, min(opp_score + pts_lose, target))
    
    ev_take = p_win_game * me_win_double + (1.0 - p_win_game) * me_lose_double
    game_equity_at_stake = abs(me_win_double - me_lose_double)

    if is_take:
        # TAKER PERSPECTIVE: choice is between TAKING (playing for double) or DROPPING (losing pts_drop)
        me_drop = equity_table.get_equity(my_score, min(opp_score + pts_drop, target))
        ev_gain = ev_take - me_drop
        
        features = torch.tensor([
            game_equity, me_drop, me_win_double, me_lose_double, me_drop,
            ev_take, ev_gain, float(cube_level) / 64.0,
        ], dtype=torch.float32)

        return features, ev_gain, game_equity_at_stake
    else:
        # DOUBLER PERSPECTIVE: choice is between DOUBLING or NOT DOUBLING
        # If we double, the opponent will choose the outcome that minimizes our equity
        me_drop = equity_table.get_equity(min(my_score + pts_drop, target), opp_score)
        ev_double = min(ev_take, me_drop)
        
        # Net gain vs doing nothing (which is exactly our current game_equity)
        ev_gain = ev_double - game_equity
        
        features = torch.tensor([
            game_equity, game_equity, me_win_double, me_lose_double, me_drop,
            ev_take, ev_gain, float(cube_level) / 64.0,
        ], dtype=torch.float32)

        return features, ev_gain, game_equity_at_stake


def compute_me_soft_target(ev_gain, game_equity_at_stake):
    """
    Convert ME net gain into a soft probability target for the cube head.
    """
    temperature = getattr(Config, 'CUBE_ME_TEMPERATURE', 4.0)
    eps = 1e-6

    normalised = ev_gain / max(game_equity_at_stake, eps)
    normalised = max(-2.0, min(2.0, normalised))

    p_positive  = torch.sigmoid(torch.tensor(normalised * temperature, dtype=torch.float32))
    soft_target = torch.stack([1.0 - p_positive, p_positive])
    return soft_target


def get_learned_cube_decision(model, game, device, my_score, opp_score,
                               equity_table=None,
                               stochastic=True, epsilon=0.0, is_take=False):
    """
    Get cube decision from the model.
    """
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.no_grad():
        _, _, v, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        value_est   = v.item()
        cube_logits = cube_logits.squeeze(0).clone()  # [2]

    me_soft_target = None
    if equity_table is not None:
        game_equity = (value_est + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        _, ev_gain, game_equity_at_stake = compute_cube_features(
            game_equity, game, my_score, opp_score, equity_table, is_take
        )
        me_soft_target = compute_me_soft_target(ev_gain, game_equity_at_stake)

    log_probs = torch.log_softmax(cube_logits, dim=0)
    probs     = log_probs.exp()

    if epsilon > 0 and random.random() < epsilon:
        action = random.randint(0, 1)
    elif stochastic:
        action = torch.multinomial(probs, 1).item()
    else:
        action = torch.argmax(probs).item()

    return action, me_soft_target, value_est
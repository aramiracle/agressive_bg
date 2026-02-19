import torch
import random
from src.config import Config


def compute_cube_features(game_equity, game, my_score, opp_score, equity_table):
    """
    Compute match-equity-aware features for the cube decision.

    Takes game_equity (already from the model's value head) and the equity
    table.  Returns structured features and the net ME gain from doubling.

    Features layout [8]:
        [0] game_equity         : model P(win game), in [0, 1]
        [1] me_current          : ME at current score
        [2] me_win              : ME if I win (cube * 2 pts)
        [3] me_lose             : ME if I lose (cube * 2 pts)
        [4] me_drop             : ME if opponent drops (I gain cube pts)
        [5] ev_double_take      : EV of doubling, assuming opponent takes
        [6] ev_double_vs_no     : net ME gain from doubling vs not doubling
        [7] cube_level_norm     : current cube / 64
    """
    cube_level = getattr(game, 'cube_value', 1)
    target     = Config.MATCH_TARGET

    pts_win  = cube_level * 2
    pts_lose = cube_level * 2
    pts_drop = cube_level

    me_current = equity_table.get_equity(my_score, opp_score)
    me_win     = equity_table.get_equity(min(my_score + pts_win,  target), opp_score)
    me_lose    = equity_table.get_equity(my_score, min(opp_score + pts_lose, target))
    me_drop    = equity_table.get_equity(min(my_score + pts_drop, target), opp_score)

    ev_double_take  = game_equity * me_win + (1.0 - game_equity) * me_lose
    ev_double_vs_no = ev_double_take - me_current

    features = torch.tensor([
        game_equity,
        me_current,
        me_win,
        me_lose,
        me_drop,
        ev_double_take,
        ev_double_vs_no,
        float(cube_level) / 64.0,
    ], dtype=torch.float32)

    return features, ev_double_vs_no


def get_learned_cube_decision(model, game, device, my_score, opp_score,
                               equity_table=None,
                               stochastic=True, epsilon=0.0):
    """
    Get cube decision from the model.

    ME reasoning is applied as a logit adjustment AFTER the normal forward
    pass — model architecture and ctx_proj are completely untouched.

    The adjustment:
        logit[1] (double/take) += alpha * ev_double_vs_no

    where ev_double_vs_no is the net ME gain from doubling:
        > 0  → ME says doubling gains match equity  → boost logit[1]
        < 0  → ME says doubling loses match equity  → suppress logit[1]
        = 0  → ME is neutral                        → no adjustment

    IMPORTANT: The adjustment is clamped to [-alpha, +alpha] to prevent the
    ME signal from completely overriding the model's board-texture assessment.
    The model must retain meaningful gradient signal — if alpha is too large
    the cube head degenerates to a threshold on ev_double_vs_no alone.

    Config keys:
        CUBE_ME_ALPHA : scale of ME logit adjustment (default 2.0).
                        Tune between 1.0–4.0.  Larger = more ME influence.

    Returns:
        action           : int, 0 (no-double / drop) or 1 (double / take)
        log_prob_chosen  : scalar tensor, log π(action|state) [for REINFORCE]
        value_est        : float, value head output in [-1, 1] [critic baseline]
    """
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.no_grad():
        _, _, v, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        value_est   = v.item()
        cube_logits = cube_logits.squeeze(0).clone()  # [2], detached

    if equity_table is not None:
        game_equity = (value_est + 1.0) / 2.0  # [-1,1] -> [0,1]
        _, ev_double_vs_no = compute_cube_features(
            game_equity, game, my_score, opp_score, equity_table
        )
        alpha = getattr(Config, 'CUBE_ME_ALPHA', 2.0)
        # Clamp so ME can nudge but not dictate
        adjustment = float(max(-alpha, min(alpha, alpha * ev_double_vs_no)))
        cube_logits[1] = cube_logits[1] + adjustment

    log_probs = torch.log_softmax(cube_logits, dim=0)
    probs     = log_probs.exp()

    if epsilon > 0 and random.random() < epsilon:
        action = random.randint(0, 1)
    elif stochastic:
        action = torch.multinomial(probs, 1).item()
    else:
        action = torch.argmax(probs).item()

    log_prob_chosen = log_probs[action]

    return action, log_prob_chosen, value_est
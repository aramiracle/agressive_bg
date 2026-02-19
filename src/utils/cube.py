import torch
import random
from src.config import Config


def compute_cube_features(game_equity, game, my_score, opp_score, equity_table):
    """
    Compute match-equity-aware features for the cube decision.

    Takes game_equity (already computed from the model's value head) plus
    the equity table to derive structured ME reasoning.

    Returns:
        features      : float32 tensor of shape [8]
        ev_double_vs_no : float, net gain from doubling (used for logit adjustment)

    Features layout:
        [0] game_equity         : model's P(win game), mapped to [0,1]
        [1] me_current          : ME at current score
        [2] me_win              : ME if I win (cube * 2 pts)
        [3] me_lose             : ME if I lose (cube * 2 pts)
        [4] me_drop             : ME if opponent drops (I gain cube pts)
        [5] ev_double_take      : EV of doubling assuming opponent takes
        [6] ev_double_vs_no     : Gain from doubling vs. not doubling
        [7] cube_level_norm     : Current cube level / 64
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
    pass — the model architecture and ctx_proj are completely untouched.

    The adjustment:
        logit[1] (double/take) += alpha * ev_double_vs_no

    where alpha = Config.CUBE_ME_ALPHA (suggested default: 1.0).
    ev_double_vs_no > 0 means ME says doubling is profitable → boost logit[1].
    ev_double_vs_no < 0 means ME says doubling is harmful   → suppress logit[1].

    The model still has full control: if the position is a clear double/pass
    from board texture, the model logit dominates. The ME adjustment only
    matters near the decision boundary, which is exactly where ME reasoning
    is most valuable.

    Returns:
        action           : int, 0 (no-double / drop) or 1 (double / take)
        log_prob_chosen  : scalar tensor, log π(action | state)  [for REINFORCE]
        value_est        : float, value head output in [-1, 1]   [critic baseline]
    """
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.no_grad():
        _, _, v, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        value_est   = v.item()
        cube_logits = cube_logits.squeeze(0).clone()  # [2]

    # Apply ME-aware logit adjustment without touching the model
    if equity_table is not None:
        game_equity = (value_est + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
        _, ev_double_vs_no = compute_cube_features(
            game_equity, game, my_score, opp_score, equity_table
        )
        alpha = getattr(Config, 'CUBE_ME_ALPHA', 1.0)
        # Shift logit[1] (double/take) relative to logit[0] (pass/no-double)
        # by the scaled ME gain. This preserves the model's base calibration
        # while injecting match equity signal at the decision boundary.
        cube_logits[1] = cube_logits[1] + alpha * ev_double_vs_no

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
import torch
import random
from src.config import Config


def compute_cube_features(game_equity, game, my_score, opp_score, equity_table):
    """
    Compute match-equity-aware features for the cube decision.

    Features layout [8]:
        [0] game_equity         : model P(win game), in [0, 1]
        [1] me_current          : ME at current score
        [2] me_win              : ME if I win (cube * 2 pts)
        [3] me_lose             : ME if I lose (cube * 2 pts)
        [4] me_drop             : ME if opponent drops (I gain cube pts)
        [5] ev_double_take      : EV of doubling, assuming opponent takes
        [6] ev_double_vs_no     : net ME gain from doubling vs not doubling
        [7] cube_level_norm     : current cube / 64

    Returns:
        features        : float32 tensor [8]
        ev_double_vs_no : float — core signal for soft target construction
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


def compute_me_soft_target(ev_double_vs_no, temperature=None):
    """
    Convert ME net gain into a soft probability target for the cube head.

    This is the teacher signal for the cube policy — analogous to how MCTS
    visit counts are the teacher signal for the move policy.

    Sigmoid maps ev_double_vs_no to P(double=correct):
        ev_double_vs_no >> 0  →  target[1] → 1.0  (strongly double)
        ev_double_vs_no ~  0  →  target[1] → 0.5  (uncertain, ME-neutral)
        ev_double_vs_no << 0  →  target[1] → 0.0  (strongly no-double)

    Temperature controls sharpness:
        lower  → sharper targets, faster learning, less exploration
        higher → softer targets, slower but more stable

    Returns float32 tensor [2]: [P(no-double), P(double)]
    """
    if temperature is None:
        temperature = getattr(Config, 'CUBE_ME_TEMPERATURE', 4.0)

    p_double   = torch.sigmoid(torch.tensor(ev_double_vs_no * temperature))
    soft_target = torch.stack([1.0 - p_double, p_double])
    return soft_target


def get_learned_cube_decision(model, game, device, my_score, opp_score,
                               equity_table=None,
                               stochastic=True, epsilon=0.0):
    """
    Get cube decision from the model.

    Returns:
        action          : int, 0 (no-double / drop) or 1 (double / take)
        me_soft_target  : float32 tensor [2] — ME-derived soft target for JS
                          training. None if equity_table not provided.
        value_est       : float, value head output in [-1, 1]
    """
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.no_grad():
        _, _, v, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        value_est   = v.item()
        cube_logits = cube_logits.squeeze(0).clone()  # [2]

    # Compute ME soft target for training (independent of action selection)
    me_soft_target = None
    if equity_table is not None:
        game_equity = (value_est + 1.0) / 2.0  # [-1,1] -> [0,1]
        _, ev_double_vs_no = compute_cube_features(
            game_equity, game, my_score, opp_score, equity_table
        )
        me_soft_target = compute_me_soft_target(ev_double_vs_no)

    log_probs = torch.log_softmax(cube_logits, dim=0)
    probs     = log_probs.exp()

    if epsilon > 0 and random.random() < epsilon:
        action = random.randint(0, 1)
    elif stochastic:
        action = torch.multinomial(probs, 1).item()
    else:
        action = torch.argmax(probs).item()

    return action, me_soft_target, value_est
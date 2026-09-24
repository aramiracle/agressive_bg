import torch
import random
from src.config import Config
from src.utils.outcome import flip, money_equity, match_equity_from_outcomes


def _stake_span(equity_table, my_score, opp_score, stake):
    """Match-equity gap between winning and losing a single at this stake."""
    target = equity_table.match_target
    win = equity_table.get_equity(min(my_score + stake, target), opp_score)
    lose = equity_table.get_equity(my_score, min(opp_score + stake, target))
    return abs(win - lose)


def cube_decision_gain(probs, my_score, opp_score, cube, equity_table, is_take=False):
    """
    Match-equity gain of the active cube action, priced from the full outcome
    distribution (single, gammon, and backgammon) at the current and doubled stake.

    probs are from `my_score`'s perspective. Positive gain means double (or take).
    A drop scores the current cube as a single, which is what a refusal is worth.
    The opponent takes when playing on at the doubled stake is at least as good
    for them as dropping.
    """
    target = equity_table.match_target
    stake = int(cube)
    me_now = match_equity_from_outcomes(probs, my_score, opp_score, stake, equity_table)
    me_doubled = match_equity_from_outcomes(probs, my_score, opp_score, stake * 2, equity_table)
    magnitude = _stake_span(equity_table, my_score, opp_score, stake)

    if is_take:
        me_drop = equity_table.get_equity(my_score, min(opp_score + stake, target))
        return me_doubled - me_drop, magnitude

    opp_take = match_equity_from_outcomes(
        flip(probs), opp_score, my_score, stake * 2, equity_table
    )
    opp_drop = equity_table.get_equity(opp_score, min(my_score + stake, target))
    my_drop = equity_table.get_equity(min(my_score + stake, target), opp_score)
    ev_if_double = me_doubled if opp_take >= opp_drop else my_drop
    return ev_if_double - me_now, magnitude



def compute_me_soft_target(ev_gain, equity_magnitude):
    """
    Convert ME net gain into a soft probability target for the cube head.

    We want the target to be:
      - ≈ 1.0  when doubling/taking is strongly positive EV
      - ≈ 0.5  near break-even
      - ≈ 0.0  when strongly negative EV

    The key insight: ev_gain is already in match-equity units (range roughly
    [-0.5, +0.5] for realistic positions).  equity_magnitude is the "size" of
    the game — how much match equity is actually at stake.  The ratio is the
    relevant signal, but we clamp it before applying temperature so large
    ratios don't saturate the sigmoid.

    Temperature is kept moderate (2.0) so that near-break-even positions
    produce soft targets rather than hard 0/1 — this preserves gradient
    information throughout training.
    """
    temperature = getattr(Config, 'CUBE_ME_TEMPERATURE', 2.0)

    # Normalise by equity magnitude, but guard against degenerate table entries.
    # Use a floor of 0.05 to prevent amplification when the table is early/flat.
    safe_magnitude = max(equity_magnitude, 0.05)
    normalised = ev_gain / safe_magnitude

    # Clamp to [-1.5, 1.5] — at temperature 2.0 this maps sigmoid to [0.05, 0.95]
    # giving the model room to learn without hard targets.
    normalised = max(-1.5, min(1.5, normalised))

    p_positive  = torch.sigmoid(torch.tensor(normalised * temperature, dtype=torch.float32))
    soft_target = torch.stack([1.0 - p_positive, p_positive])
    return soft_target


def get_learned_cube_decision(model, game, device, my_score, opp_score,
                               equity_table=None,
                               stochastic=True, epsilon=0.0, is_take=False,
                               equity_probs=None):
    """
    Get cube decision from the model.

    Returns: (action, me_soft_target, value_est, outcome_probs)
        action          : 0=no-double/drop  1=double/take
        me_soft_target  : training target for the cube head [2]
        value_est       : cubeless money equity in [-1, 1] of the priced distribution
        outcome_probs   : this net's 6-way outcome distribution

    equity_probs prices the soft target. A take passes the doubler's distribution
    flipped into the responder's perspective, because the responder will not roll.
    """
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.inference_mode():
        outcome_logits, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        outcome_probs = torch.softmax(outcome_logits.float().squeeze(0), dim=0).cpu()
        cube_logits = cube_logits.float().squeeze(0).clone().cpu()

    priced = outcome_probs if equity_probs is None else equity_probs
    value_est = float(money_equity(priced))

    me_soft_target = None
    if equity_table is not None:
        ev_gain, equity_magnitude = cube_decision_gain(
            priced, my_score, opp_score, game.cube, equity_table, is_take
        )
        me_soft_target = compute_me_soft_target(ev_gain, equity_magnitude)

    cube_probs = torch.softmax(cube_logits, dim=0)

    if epsilon > 0 and random.random() < epsilon:
        action = random.randint(0, 1)
    elif stochastic:
        action = torch.multinomial(cube_probs, 1).item()
    else:
        action = torch.argmax(cube_probs).item()

    return action, me_soft_target, value_est, outcome_probs
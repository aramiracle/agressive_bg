import torch
import random
from src.config import Config



def compute_cube_features(game_equity, game, my_score, opp_score, equity_table, is_take=False):
    """
    Compute match-equity-aware features for the cube decision.

    game_equity : float in [0,1] — the value-head output mapped to match-equity space.
                  At the moment this is called the cube is at `cube_level` (not yet doubled),
                  so the model sees equity under the CURRENT stake, which is:
                    E_current = p_win * ME(my+cube) + (1-p_win) * ME(opp+cube_from_opp)

    For the DOUBLER the choice is:
        double → opponent takes  → play at 2*cube  [EV = ev_take]
        double → opponent drops  → we gain cube pts [EV = me_drop_doubler]
        no-double                → continue at cube  [EV = game_equity (baseline)]

    For the TAKER the choice is:
        take  → play at 2*cube           [EV = ev_take  (from taker's perspective)]
        drop  → opponent gains cube pts  [EV = me_drop_taker]

    Returns: (features, ev_gain, equity_magnitude)
        ev_gain          : how much better the "active" action is vs the passive one
        equity_magnitude : absolute scale of the decision, used for normalisation

    Design principle: NO thresholds, NO heuristics.  The soft target is a smooth
    function of ev_gain that lets the RL signal guide the cube head.
    """
    cube_level = game.cube
    target     = Config.MATCH_TARGET

    # Points exchanged at current and doubled stake
    pts_current = cube_level           # winning/losing right now
    pts_doubled = cube_level * 2       # winning/losing after a double

    # ----------------------------------------------------------------
    # Step 1: Back-solve for game-winning probability p_win.
    #
    #   E_current = p * ME(my+pts_current) + (1-p) * ME(opp+pts_current from opp)
    #
    # The model was trained to predict exactly this quantity, so the
    # inversion is in-distribution.
    # ----------------------------------------------------------------
    me_win_current  = equity_table.get_equity(
        min(my_score + pts_current, target), opp_score)
    me_lose_current = equity_table.get_equity(
        my_score, min(opp_score + pts_current, target))

    denom = me_win_current - me_lose_current
    # Guard against a degenerate table (shouldn't happen after warmup)
    if abs(denom) < 1e-6:
        # Table not yet calibrated – fall back to game_equity as p_win
        p_win = float(game_equity)
    else:
        p_win = (game_equity - me_lose_current) / denom
        # Clip to [0,1] but keep a small margin so we don't saturate gradients
        p_win = max(1e-4, min(1.0 - 1e-4, p_win))

    # ----------------------------------------------------------------
    # Step 2: Evaluate TAKING at the doubled stake (both sides share this)
    #
    #   From the DOUBLER's perspective:
    #     ev_take = p_win * ME(my+pts_doubled) + (1-p_win) * ME(opp+pts_doubled)
    # ----------------------------------------------------------------
    me_win_doubled  = equity_table.get_equity(
        min(my_score + pts_doubled, target), opp_score)
    me_lose_doubled = equity_table.get_equity(
        my_score, min(opp_score + pts_doubled, target))

    ev_take = p_win * me_win_doubled + (1.0 - p_win) * me_lose_doubled

    # ----------------------------------------------------------------
    # Step 3: Compute ev_gain and equity_magnitude
    # ----------------------------------------------------------------
    if is_take:
        # TAKER: compare EV of taking vs EV of dropping
        #   Taking → play on for doubled stakes → ev_take (from taker's POV, already correct)
        #   Dropping → opponent wins pts_current (not doubled) immediately
        me_drop_taker = equity_table.get_equity(
            my_score, min(opp_score + pts_current, target))

        ev_gain = ev_take - me_drop_taker

        # Magnitude: span of outcomes at doubled stake from taker's view
        equity_magnitude = abs(me_win_doubled - me_lose_doubled)

        max_cube = 1 << target.bit_length()
        features = torch.tensor([
            game_equity,
            p_win,
            me_win_doubled, me_lose_doubled,
            me_drop_taker,
            ev_take, ev_gain,
            float(cube_level) / max_cube,
        ], dtype=torch.float32)

    else:
        # DOUBLER:
        #   If we offer the double, the rational opponent takes iff ev_take < me_drop_opp.
        #   (Taking is good for the opponent when our EV is LOW, i.e. ev_take is low.)
        #   Opponent minimises our EV → we receive min(ev_take, me_drop_doubler).
        #
        #   me_drop_doubler: equity if opponent drops (they give us pts_current immediately)
        me_drop_doubler = equity_table.get_equity(
            min(my_score + pts_current, target), opp_score)

        # What we actually get if we double (opponent plays optimally against us)
        ev_if_we_double = min(ev_take, me_drop_doubler)

        # Gain vs NOT doubling (continue at current stake)
        ev_gain = ev_if_we_double - game_equity

        # ----------------------------------------------------------------
        # Magnitude: use the equity span of the CURRENT (pre-double) game,
        # NOT the doubled game. This is the correct normalisation scale:
        # ev_gain is already expressed in match-equity units, and
        # ME(win@current) - ME(lose@current) is the maximum swing possible
        # from this cube decision — it is never tiny because it measures
        # the CURRENT game's worth.
        #
        # Normalising by the doubled span was the original bug:
        #   doubled span ≈ 0.08 at (0,0) cube=1 → amplified ev_gain ~12×
        #   → normalised saturated sigmoid → always double.
        # ----------------------------------------------------------------
        equity_magnitude = abs(me_win_current - me_lose_current)

        max_cube = 1 << target.bit_length()
        features = torch.tensor([
            game_equity,
            p_win,
            me_win_doubled, me_lose_doubled,
            me_drop_doubler,
            ev_take, ev_gain,
            float(cube_level) / max_cube,
        ], dtype=torch.float32)

    return features, ev_gain, equity_magnitude


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
                               stochastic=True, epsilon=0.0, is_take=False):
    """
    Get cube decision from the model.

    Returns: (action, me_soft_target, value_est)
        action          : 0=no-double/drop  1=double/take
        me_soft_target  : training target for the cube head [2]
        value_est       : raw value head output in [-1,1]
    """
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)

    with torch.no_grad():
        _, _, v, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        value_est   = v.item()
        cube_logits = cube_logits.squeeze(0).clone()  # [2]

    me_soft_target = None
    if equity_table is not None:
        # Map value head output [-1,1] to match-equity space [0,1]
        game_equity = (value_est + 1.0) / 2.0

        # Clip away from extremes: a value near ±1.0 means the model is already
        # very confident about the match outcome (e.g., 6-away 6-away behind by a
        # lot).  Clipping prevents p_win saturation when the ME table is still
        # being learned, while staying differentiable in practice.
        game_equity = max(0.02, min(0.98, game_equity))

        _, ev_gain, equity_magnitude = compute_cube_features(
            game_equity, game, my_score, opp_score, equity_table, is_take
        )
        me_soft_target = compute_me_soft_target(ev_gain, equity_magnitude)

    log_probs = torch.log_softmax(cube_logits, dim=0)
    probs     = log_probs.exp()

    if epsilon > 0 and random.random() < epsilon:
        action = random.randint(0, 1)
    elif stochastic:
        action = torch.multinomial(probs, 1).item()
    else:
        action = torch.argmax(probs).item()

    return action, me_soft_target, value_est
import os
import torch


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Config:
    # ========================================
    # TRAINING STAGE (read from environment so spawned workers see it)
    #   Stage 1: 1-point games, no cube  -> learn checker play
    #   Stage 2: 7-point matches + cube  -> learn match play and doubling
    # ========================================
    STAGE = _env_int("BG_STAGE", 1)

    # Board / Game
    NUM_POINTS = 24
    CHECKERS_PER_PLAYER = 15
    HOME_SIZE = 6
    DICE_SIDES = 6

    BAR_IDX = 24
    OFF_IDX = 25
    NUM_ACTIONS = 26

    BOARD_SEQ_LEN = 28
    EMBED_VOCAB_SIZE = 31
    EMBED_OFFSET = 15
    # ctx = [cube_owner, cube / MAX_CUBE, my_score / target, opp_score / target,
    #        crawford_active]
    # cube_owner is +1 mine, -1 opp, 0 centred. CUBE_OFFERED (2) is reserved
    # for the responder to a double: that side will not roll.
    CONTEXT_SIZE = 5
    CUBE_OFFERED = 2.0
    MAX_CUBE = 64.0

    INITIAL_SETUP = {
        0: -2, 5: 5, 7: 3, 11: -5,
        12: 5, 16: -3, 18: -5, 23: 2,
    }

    # Game
    MATCH_TARGET = _env_int("BG_MATCH_TARGET", 1 if STAGE == 1 else 7)
    CUBE_ENABLED = _env_bool("BG_CUBE_ENABLED", STAGE == 2)
    # Stage 1 scores games with the aggressive weights below (R_*), stage 2 uses
    # the real 1/2/3 point values so a 7-point match behaves like real match play.
    TRAIN_MODE = _env_bool("BG_TRAIN_MODE", STAGE == 1)

    # Rewards (used as game point values when TRAIN_MODE, and as equity weights)
    R_WIN = 1.0
    R_GAMMON = 3.0
    R_BACKGAMMON = 5.0

    # Outcome head: [win, win gammon, win backgammon, lose, lose gammon, lose backgammon]
    NUM_OUTCOMES = 6

    # Model
    MODEL_TYPE = "transformer"  # "transformer" or "cnn"
    D_MODEL = 128
    DROPOUT = 0.1
    VALUE_HIDDEN = 64
    MAX_SEQ_LEN = BOARD_SEQ_LEN + 1

    # Transformer: width kept, depth/FFN cut so a full AdamW checkpoint
    # (weights + optimizer) stays under 15MB for both training stages.
    N_HEAD = 16
    N_LAYERS = 10
    DIM_FEEDFORWARD = 256

    # CNN specific
    CNN_BLOCKS = 4
    CNN_KERNEL = 3

    # ========================================
    # MCTS (efficient complete-turn tree)
    #   Expand every legal full turn, score afterstates with the outcome head
    #   (win probability / equity), prune losing plays, then PUCT among the
    #   survivors. Same-player backup — values are never negated inside a turn.
    #   SEARCH_PLY=1 is the training default: every afterstate is scored once.
    #   SEARCH_PLY=2 adds expectimax over the 21 opponent rolls on survivors
    #   (~30-50x more network evals per turn). Use BG_SEARCH_PLY=2 for play.
    # ========================================
    NUM_SIMULATIONS = _env_int("BG_NUM_SIMULATIONS", 32)
    MCTS_BATCH = 4
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPS = 0.25
    MIN_PRIOR = 1e-5

    SEARCH_PLY = _env_int("BG_SEARCH_PLY", 1)
    SEARCH_PRUNE_TOP_K = 3
    SEARCH_PRUNE_MARGIN = 0.10      # equity units in [-1, 1]
    SEARCH_EVAL_BATCH = 512

    # Race handling (B6): when there is no contact and both sides have already
    # borne off a checker, the game is a pure race with no gammon possible; an
    # analytic pip-count formula is used as the leaf evaluation, and self-play
    # games are terminated early with that distribution as the target.
    RACE_EARLY_TERMINATION = True

    # Exploration (B5): sample among candidate turns for the first EXPLORE_TURNS
    # turns of each game with a softmax over equity, then play greedily.
    EXPLORE_TURNS = 8
    EXPLORE_TEMPERATURE = 0.05

    # TD(lambda) value targets (B4)
    TD_LAMBDA = 0.7

    # ========================================
    # CUBE LEARNING - CURRICULUM
    # ========================================
    # epsilon controls stochastic exploration of cube decisions.
    # cube_weight scales the cube head loss relative to value + policy loss.
    #
    # The soft target is derived from ME net-gain (ev_gain / equity_magnitude),
    # so the cube head receives a meaningful signal at every position.
    # We ramp epsilon down slowly to explore both take and drop positions,
    # then let the model converge.
    CUBE_CURRICULUM_ENABLED = True
    CUBE_CURRICULUM_STAGES = [
        {'steps': 0,      'epsilon': 0.2,  'cube_weight': 1.5},
        {'steps': 25000,  'epsilon': 0.1,  'cube_weight': 1.3},
        {'steps': 50000,  'epsilon': 0.05,  'cube_weight': 1.2},
        {'steps': 75000,  'epsilon': 0.02,  'cube_weight': 1.1},
        {'steps': 100000, 'epsilon': 0.01,  'cube_weight': 1.0},
        {'steps': 150000, 'epsilon': 0.005,  'cube_weight': 1.0},
        {'steps': 200000, 'epsilon': 0.002, 'cube_weight': 1.0},
    ]

    # CUBE_LOSS_WEIGHT: the cube JS loss is now properly scaled (ev_gain in ME units,
    # magnitude floored at 0.05), so 1.0 is the right default.
    CUBE_LOSS_WEIGHT    = 1.0

    # CUBE_ME_TEMPERATURE: sigmoid sharpness applied to normalised ev_gain.
    # normalised = ev_gain / equity_magnitude, clamped to [-1.5, 1.5].
    # At temperature 2.0, sigmoid(1.5*2) = 0.95 — never a hard target.
    # This was 4.0 before, which combined with small equity_magnitude
    # caused sigmoid saturation (always double).
    CUBE_ME_TEMPERATURE = 2.0

    # ELO
    INITIAL_ELO = 0
    ELO_K = 1
    ELO_SCALE = 400.0
    ELO_EVAL_INTERVAL = 1000
    ELO_EVAL_GAMES = 64

    # Best-model gating (E3): the candidate replaces best_model only if its win
    # rate against best_model over GATE_GAMES games exceeds GATE_WIN_RATE.
    GATE_WIN_RATE = _env_float("BG_GATE_WIN_RATE", 0.53)
    GATE_GAMES = _env_int("BG_GATE_GAMES", 40)

    # Training
    MATCHES_PER_ITERATION = 4
    TRAIN_UPDATES_PER_ITER = 200

    BATCH_SIZE = 512 if torch.cuda.is_available() else 256
    # D2: one sample per turn now (~50 per game). Stage 1 collects ~1k samples per
    # iteration, stage 2 (7-pt matches) ~10k, so these hold roughly 50 / 30
    # iterations of fresh data.
    BUFFER_SIZE = _env_int("BG_BUFFER_SIZE", 50000 if STAGE == 1 else 300000)
    KL_EPSILON = 1e-6
    LABEL_SMOOTHING = 0.02

    LR = 1e-5
    GRAD_CLIP = 1.0
    WEIGHT_DECAY = 1e-4
    TRAIN_STEPS = 1000000
    LOSS_AVG_WINDOW = 100
    MAX_GAME_MOVES = 400

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SELF_PLAY_DEVICE = "cpu"  # Often faster to run env on CPU if GPU is busy training

    # Checkpoints live under checkpoints/: baseline/ is frozen, stageN/ is live.
    CHECKPOINT_DIR = os.environ.get(
        "BG_CHECKPOINT_DIR", os.path.join("checkpoints", f"stage{STAGE}")
    )
    # Warm-start weights when no checkpoint exists in CHECKPOINT_DIR
    # (stage 2 starts from the stage 1 best model by default).
    INIT_FROM = os.environ.get(
        "BG_INIT_FROM",
        os.path.join("checkpoints", "stage1", "best_model.pt") if STAGE == 2 else ""
    )
    BASELINE_DIR = os.environ.get(
        "BG_BASELINE_DIR", os.path.join("checkpoints", "baseline")
    )
    BASELINE_MODEL_NAME = "best_model.pt"
    BASELINE_SWITCH_ON_SURPASS = True
    BASELINE_SELF_PLAY_RATIO = 0.5

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
    # Stage 1: 1-point games, no cube, aggressive scores. Learn checker play.
    # Stage 2: 7-point matches with the cube and real 1/2/3 scoring.
    STAGE = _env_int("BG_STAGE", 1)
    MATCH_TARGET = _env_int("BG_MATCH_TARGET", 1 if STAGE == 1 else 7)
    CUBE_ENABLED = _env_bool("BG_CUBE_ENABLED", STAGE == 2)
    TRAIN_MODE = _env_bool("BG_TRAIN_MODE", STAGE == 1)

    # Point values while TRAIN_MODE is on. Otherwise a win is 1/2/3.
    R_WIN = 1.0
    R_GAMMON = 3.0
    R_BACKGAMMON = 5.0

    # Board
    NUM_POINTS = 24
    CHECKERS_PER_PLAYER = 15
    HOME_SIZE = 6
    DICE_SIDES = 6
    BAR_IDX = 24
    OFF_IDX = 25
    NUM_ACTIONS = 26
    INITIAL_SETUP = {
        0: -2, 5: 5, 7: 3, 11: -5,
        12: 5, 16: -3, 18: -5, 23: 2,
    }

    # Encoding. ctx = [cube_owner, cube / MAX_CUBE, my_score / target,
    # opp_score / target, crawford]. cube_owner is +1 mine, -1 opp, 0 centred.
    # CUBE_OFFERED is the responder to a double: that side will not roll.
    BOARD_SEQ_LEN = 28
    EMBED_VOCAB_SIZE = 31
    EMBED_OFFSET = 15
    CONTEXT_SIZE = 5
    CUBE_OFFERED = 2.0
    MAX_CUBE = 64.0
    NUM_OUTCOMES = 6

    # Model. Transformer depth is capped so an AdamW checkpoint stays small.
    MODEL_TYPE = "transformer"  # "transformer" or "cnn"
    D_MODEL = 128
    DROPOUT = 0.1
    VALUE_HIDDEN = 64
    MAX_SEQ_LEN = BOARD_SEQ_LEN + 1
    N_HEAD = 16
    N_LAYERS = 10
    DIM_FEEDFORWARD = 256
    CNN_BLOCKS = 4
    CNN_KERNEL = 3

    # Search. Ply 1 scores every afterstate once. Ply 2 expectimaxes the
    # opponent's replies (BG_SEARCH_PLY=2 for play).
    NUM_SIMULATIONS = _env_int("BG_NUM_SIMULATIONS", 32)
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPS = 0.25
    MIN_PRIOR = 1e-5
    SEARCH_PLY = _env_int("BG_SEARCH_PLY", 1)
    SEARCH_PRUNE_TOP_K = 3
    SEARCH_PRUNE_MARGIN = 0.10
    SEARCH_EVAL_BATCH = 512
    RACE_EARLY_TERMINATION = True
    EXPLORE_TURNS = 8
    EXPLORE_TEMPERATURE = 0.05
    TD_LAMBDA = 0.7
    MAX_GAME_MOVES = 400

    # Cube. Epsilon is the random-action rate; cube_weight scales the cube loss.
    CUBE_CURRICULUM_STAGES = [
        {'steps': 0,      'epsilon': 0.2,  'cube_weight': 1.5},
        {'steps': 25000,  'epsilon': 0.1,  'cube_weight': 1.3},
        {'steps': 50000,  'epsilon': 0.05, 'cube_weight': 1.2},
        {'steps': 75000,  'epsilon': 0.02, 'cube_weight': 1.1},
        {'steps': 100000, 'epsilon': 0.01, 'cube_weight': 1.0},
        {'steps': 150000, 'epsilon': 0.005, 'cube_weight': 1.0},
        {'steps': 200000, 'epsilon': 0.002, 'cube_weight': 1.0},
    ]
    CUBE_LOSS_WEIGHT = 1.0
    CUBE_ME_TEMPERATURE = 2.0

    # Optimisation
    MATCHES_PER_ITERATION = 8
    TRAIN_UPDATES_PER_ITER = 200
    BATCH_SIZE = 512 if torch.cuda.is_available() else 256
    BUFFER_SIZE = _env_int("BG_BUFFER_SIZE", 50000 if STAGE == 1 else 300000)
    LABEL_SMOOTHING = 0.02
    LR = 1e-5
    GRAD_CLIP = 1.0
    WEIGHT_DECAY = 1e-4
    TRAIN_STEPS = 1000000

    # Rating. The champion is replaced only after GATE_GAMES wins above GATE_WIN_RATE.
    INITIAL_ELO = 0
    ELO_K = 1
    ELO_SCALE = 400.0
    ELO_EVAL_INTERVAL = 1000
    GATE_WIN_RATE = _env_float("BG_GATE_WIN_RATE", 0.53)
    GATE_GAMES = _env_int("BG_GATE_GAMES", 100)

    # Runtime
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SELF_PLAY_DEVICE = "cpu"
    CHECKPOINT_DIR = os.environ.get(
        "BG_CHECKPOINT_DIR", os.path.join("checkpoints", f"stage{STAGE}")
    )
    INIT_FROM = os.environ.get(
        "BG_INIT_FROM",
        os.path.join("checkpoints", "stage1", "best_model.pt") if STAGE == 2 else ""
    )
    BASELINE_DIR = os.environ.get(
        "BG_BASELINE_DIR", os.path.join("checkpoints", "baseline")
    )
    BASELINE_MODEL_NAME = "best_model.pt"
    BASELINE_SELF_PLAY_RATIO = 0.5

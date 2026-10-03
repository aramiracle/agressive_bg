# Backgammon AI

A neural-network Backgammon engine trained by self-play reinforcement learning.
The network predicts the six-way outcome distribution of a position (single /
gammon / backgammon, won or lost); a search over complete dice plays ranks the
legal turns with it, and a second head learns doubling-cube decisions priced
from match equity. See [`GUIDE.md`](GUIDE.md) for the full design reference.

## Features

- **Outcome-distribution network**: transformer (default) or CNN with a
  six-way outcome head and a cube head (`src/model.py`)
- **Turn-level search**: scores every legal complete play, prunes near-best,
  optional two-ply expectimax over all 21 rolls, PUCT bandit (`src/mcts.py`,
  `BG_SEARCH_PLY=2` for the deeper mode)
- **Self-play training**: TD(lambda) on distributions, prioritized replay,
  two-stage curriculum (stage 1: 1-point games, no cube; stage 2: 7-point
  matches with cube and Crawford)
- **ELO gate**: the live network must beat the champion to be promoted
  (`checkpoints/*/best_model.pt`)
- **Web UI**: play against the AI in the browser (human vs AI, AI vs human,
  AI vs AI with autoplay)

## Project Structure

```
agressive_bg/
├── src/
│   ├── config.py              # all settings; BG_* env overrides
│   ├── engine.py              # strict rules engine (moves, maximality, cube, Crawford)
│   ├── model.py               # transformer, CNN, legacy value adapter
│   ├── mcts.py                # turn search: expectimax + PUCT bandit
│   ├── search.py              # Searcher API and candidate selection
│   ├── replay_buffer.py       # prioritized replay
│   ├── trainer.py             # self-play training loop
│   ├── trainer_vs_baseline.py # same loop vs frozen baseline / champion
│   └── utils/                 # checkpoint, cube, elo, game, train, match_equity, outcome, ...
├── scripts/
│   └── play_web.py            # web server: HTML UI over HTTP + WebSocket
├── ui/
│   ├── html_ui.html           # browser board (served by play_web.py)
│   └── ws_server.py           # standalone WebSocket server (legacy; its AI never cubes)
├── checkpoints/
│   ├── baseline/              # frozen stage-1 champion (opponent for vs-baseline training)
│   ├── stage1/                # live stage-1 run
│   └── stage2/                # live stage-2 run
├── tests/                     # regression + rules/protocol suites (see Testing)
├── GUIDE.md                   # detailed design and behavior reference
├── requirements.txt
└── pyproject.toml
```

Each checkpoint directory holds `best_model.pt`, `latest_model.pt`, a
`config.py` describing the architecture, and a learned `match_equity.pt`
table. A saved `.pt` carries `model_state_dict`, `optimizer_state_dict`,
`step`, `elo`, `loss`, and a small `config` record.

## Installation

```bash
pip install -r requirements.txt   # torch, tqdm, websockets
```

Everything imports from the repository root (`from src.…`), so run all entry
points in module form from the repo root. `pip install -e .` works but is not
required, and its `bg-train` console script is stale (points at a removed
`backgammon` package — use the module commands below).

## Training

```bash
# Pure self-play (stage selected by BG_STAGE, default 1)
python -m src.trainer

# Self-play + frozen baseline evaluation, explicit stage
python -m src.trainer_vs_baseline --stage 1
python -m src.trainer_vs_baseline --stage 2 --matches 80 --steps 50000
```

Training plays `MATCHES_PER_ITERATION` matches per iteration (CPU workers for
`src.trainer`), appends labeled transitions to the replay buffer, runs
optimizer steps, and every `ELO_EVAL_INTERVAL` steps plays a gate of
`GATE_GAMES` matches, promoting `best_model.pt` only above the win-rate gate.

## Playing (Web UI)

```bash
python -m scripts.play_web      # from the repo root
```

Then open **http://localhost:8080/html_ui.html** in a browser. The same
process serves the WebSocket the client talks to on **ws://localhost:8765**
(hosts `0.0.0.0`). It auto-loads the first checkpoint found in
`checkpoints/stage2/`, then `stage1/`, then `baseline/`; the Load Model button
can upload a `.pt` instead.

Run it under `BG_STAGE=2` for the match/cube configuration the UI's match
controls expect:

```bash
BG_STAGE=2 python -m scripts.play_web
```

Game modes (dropdown): Human vs AI (you play White), AI vs Human (you play
Black), AI vs AI (watch, with Step / Autoplay controls).

## Configuration

Edit `src/config.py`, or use these environment variables (read once at import):

| Variable | Meaning (default) |
|----------|-------------------|
| `BG_STAGE` | `1` = 1-point games, no cube, aggressive `1/3/5` training scores; `2` = 7-point matches, cube, real `1/2/3` |
| `BG_MATCH_TARGET` / `BG_CUBE_ENABLED` / `BG_TRAIN_MODE` | override the stage defaults individually |
| `BG_NUM_SIMULATIONS` | bandit visit budget (32) |
| `BG_SEARCH_PLY` | `1` afterstate scoring (default), `2` = expectimax over opponent replies |
| `BG_BUFFER_SIZE` | replay capacity (50,000 / 300,000 by stage) |
| `BG_GATE_WIN_RATE` / `BG_GATE_GAMES` | promotion gate (0.54 / 100) |
| `BG_CHECKPOINT_DIR` / `BG_INIT_FROM` / `BG_BASELINE_DIR` | checkpoint locations |

Other notable settings in the file: `D_MODEL = 128`, `N_LAYERS = 10`,
`N_HEAD = 16`, `DIM_FEEDFORWARD = 256`, `C_PUCT = 1.5`, `TD_LAMBDA = 0.7`,
`LR = 1e-5`, `BATCH_SIZE = 512` on CUDA (`256` otherwise),
`TRAIN_STEPS = 1_000_000`. `GUIDE.md` section 15 explains which knobs change
the objective versus the speed.

## Testing

```bash
python -m pytest tests/ -q                  # everything
python -m tests.test                        # regression suite (search, TD, gate, Elo)
python -m tests.test_rules_edge_cases       # engine rules + AI legality fuzz
python -m tests.test_ui_protocol            # every web-UI action and edge case
```

Known state: the checkpoint-size guard in `tests.test` currently reports a
failure (the shipped 10-layer model saves to ~15.6 MB against a 15 MB budget);
all behavioral checks pass. See `GUIDE.md` section 13.

## License

MIT License (see `pyproject.toml`).

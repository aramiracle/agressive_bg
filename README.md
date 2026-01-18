# Backgammon AI

A neural network-based Backgammon AI using Monte Carlo Tree Search (MCTS) with self-play reinforcement learning.

## Features

- **Transformer/CNN Neural Network**: Configurable model architecture for board evaluation
- **MCTS Search**: Monte Carlo Tree Search for move selection
- **Self-Play Training**: Reinforcement learning through self-play games
- **ELO Rating System**: Track model improvement over training
- **Doubling Cube**: Full support for doubling cube strategy
- **Crawford Rule**: Proper match play with Crawford rule
- **Multiple UIs**: Web-based and desktop (Kivy) interfaces

## Project Structure

```
agressive_bg/
├── src/
│   └── backgammon/
│       ├── __init__.py
│       ├── config.py          # Configuration and hyperparameters
│       ├── engine.py          # Game engine (rules, moves, scoring)
│       ├── model.py           # Neural network models (Transformer/CNN)
│       ├── mcts.py            # Monte Carlo Tree Search
│       ├── trainer.py         # Self-play training loop
│       ├── checkpoint.py      # Model checkpointing utilities
│       ├── elo.py             # ELO rating system
│       ├── utils.py           # Utility functions
│       ├── ui_desktop.py      # Kivy desktop UI
│       └── ui_web/
│           └── html_ui.html   # Web UI (HTML/CSS/JS)
├── scripts/
│   ├── train.py               # Training entry point
│   ├── play_web.py            # WebSocket server for web UI
│   └── play_desktop.py        # Desktop UI entry point
├── checkpoints/               # Saved model checkpoints
├── tests/                     # Unit tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/agressive_bg.git
cd agressive_bg

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### With Optional Dependencies

```bash
# For web UI
pip install -e ".[web]"

# For desktop UI
pip install -e ".[desktop]"

# For all features
pip install -e ".[all]"

# For development
pip install -e ".[dev]"
```

## Usage

### Training

```bash
# Start training
python scripts/train.py

# Or if installed as package
bg-train
```

Training will:
- Generate self-play games
- Train the neural network
- Periodically evaluate against the best model
- Save checkpoints to `checkpoints/`

### Playing (Web UI)

```bash
# Start the WebSocket server
python scripts/play_web.py

# Open the HTML UI in your browser
# File: src/backgammon/ui_web/html_ui.html
```

Features:
- Human vs AI, AI vs Human, AI vs AI modes
- Autoplay for AI vs AI with speed control
- Model loading from file
- Match scoring with Crawford rule
- Doubling cube support

### Playing (Desktop UI)

```bash
# Requires Kivy
pip install kivy

# Start the desktop app
python scripts/play_desktop.py
```

## Configuration

Edit `src/backgammon/config.py` to customize:

```python
class Config:
    # Model architecture
    MODEL_TYPE = "transformer"  # "transformer" or "cnn"
    D_MODEL = 64
    N_LAYERS = 3
    
    # MCTS
    NUM_SIMULATIONS = 64
    C_PUCT = 1.5
    
    # Training
    BATCH_SIZE = 1024
    LR = 5e-4
    TRAIN_STEPS = 300000
    
    # Match
    MATCH_TARGET = 7
```

## Game Modes

- **Human vs AI**: You play White, AI plays Black
- **AI vs Human**: AI plays White, you play Black
- **AI vs AI**: Watch two AI agents play (with autoplay)

## Model Checkpoints

Checkpoints are saved to `checkpoints/`:
- `best_model.pt`: Best performing model by ELO
- `latest_model.pt`: Most recent checkpoint

Checkpoint structure:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'step': int,
    'elo': float,
    'loss': float,
    'config': {
        'model_type': str,
        'd_model': int,
        'n_layers': int,
    }
}
```

## License

MIT License


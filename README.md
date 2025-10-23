# Pass the Pigs (Dice Game)

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.26.2+-green.svg)](https://gymnasium.farama.org/)
[![Pygame](https://img.shields.io/badge/pygame-2.5.2+-orange.svg)](https://www.pygame.org/news)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![TensorBoard](https://img.shields.io/badge/tensorboard-2.0+-ff6f00.svg)](https://www.tensorflow.org/tensorboard)


Quick introduction: Pass the Pigs is a version of the dice game Pig (created by David Moffatt in 1977) and uses asymmetrical throwing dice.

--> Each turn, players throw 2 model pigs (each has a dot on its side) and the player will gain/lose points or be eliminated from the game.
--> Players can choose to "bank" points or risk losing them on a bad roll
--> Winner is first player to reach a predetermined score (at the moment, we say 100).

### Rules
![alt text](https://github.com/apayne185/pass-the-pig-RL/blob/main/assets/game/game-rules.png?raw=true)


### Scoring System

| Roll Result | Points | Description |
|-------------|---------|-------------|
| Double Trotter | 40 | Both pigs on side with legs showing |
| Double Snouter | 30 | Both pigs on snout |
| Double Razorback | 20 | Both pigs on back |
| Double Leaning Jowler | 20 | Both pigs leaning on side |
| Leaning Jowler | 15 | Single pig leaning on side |
| Snouter | 10 | Single pig on snout |
| Razorback | 5 | Single pig on back |
| Trotter | 5 | Single pig with legs showing |
| Sider | 1 | Single pig on side | roll again |
| Piggyback | 0 | One pig on top of another | lose turn |
| Pig Out | 0 | One pig on side, one on back | lose turn |
| Oinker | 0 | Both pigs on different sides | lose turn |


## stuff to modify
- **try cournot variant**  --> anna
- **try sequential variant (stackelberg)** --> close to what we already have --> tomas
- progressive training (curriculum learning)
    1. start with dumb opponents - roller 50%
    2. then smarter opponents (baseline threshold 15, 20,...)
    3. then mirror match (agent v itself --> self play)

- progress further with hog calls --> we already have a withhogcall flag
    1. intermediate rewards integrated for agent to learn immediate rewards after correct/incorrect calls

- multiagent (we train both agents to see if they converge to stable strategies)
    --> maybe try collusion or coalitions if 2+ players
    --> add 2+ players? make it collaborative?


## Directory
```
pass-the-pig/
├── main.py            # Main runner script (renamed from run.py)
├── env.py             # Main environment and agent classes
├── environment.yml    # Conda environment setup
├── README.md          # This file
├── output/            # Model results saved here
├── runs/              # TensorBoard logs
└── assets/     # Game assets (images, sounds)
```

## Project Overview

### Environment Design
- **State**:
- **Actions**:
- **Rewards**:


### Agent Architecture

The Q-learning agent features:

- **Curriculum Training**: Progressive opponent difficulty
- **Efficient Q-tables**: defaultdict implementation for large state spaces
- **Epsilon-greedy Exploration**: Balanced exploration vs exploitation
- **Threshold-based Strategies**: Intelligent banking decisions




## Installation

### Prerequisites
- Python 3.9+
- Conda (Miniconda or Anaconda)

### Setup

1. **Create and activate the conda environment:**

```bash
conda env create -f environment.yml
conda activate ptp-env
```

2. **Verify installation:**

```bash
conda list
```

The environment includes:
- Python 3.9
- NumPy (numerical computing)
- Pygame (game rendering)
- PyTorch (deep learning framework)
- Gymnasium 0.26.2 (RL environment interface)
- TensorBoard (training visualization)

## Running the Game

The game can be run in multiple modes using the `main.py` script with command-line arguments.

### Basic Usage

```bash
python main.py [--mode MODE] [--games NUM] [--agent1 TYPE] [--agent2 TYPE] [options]
```

### Game Modes

#### 1. **Console Mode (Default)** - Text-Based Interactive Play
Play the game in your terminal without requiring pygame:

```bash
# Default mode - just run main.py
python main.py

# Or explicitly specify console mode
python main.py --mode console
```

**Controls:**
- `r` - Roll the dice
- `p` - Pass (bank your points and end turn)
- `1` - Hog Call 1 (bet opponent will roll exactly 5 points)
- `2` - Hog Call 2 (bet opponent will roll exactly 10 points)
- `q` - Quit game

#### 2. **Human Mode** - Watch AI Agents Play
Watch AI agents compete with visual rendering:

```bash
# 10 games, QTable vs Baseline
python main.py --mode human

# Customize agents and number of games
python main.py --mode human --games 20 --agent1 Baseline --agent2 Baseline --threshold1 25 --threshold2 15

# Watch with detailed output
python main.py --mode human --games 5 --verbose
```

#### 3. **Interactive Mode** - Play with Pygame
Control one player using pygame keyboard controls while AI controls the opponent:

```bash
python main.py --mode interactive
```

**Controls:**
- `SPACE` or `R` - Roll the dice
- `P` - Pass (bank your points and end turn)
- `H` or `1` - Hog Call 1 (bet on 5 points)
- `2` - Hog Call 2 (bet on 10 points)
- `Arrow Keys` - Move cursor (demo feature)
- Close window to exit

#### 4. **Training Mode** - Train Q-Learning Agent
Train a Q-learning agent against a baseline opponent:

```bash
# Default: 50 epochs × 50 games = 2,500 total games
python main.py --mode train

# Custom training duration
python main.py --mode train --epochs 100 --games-per-epoch 100

# Results saved to output/ folder and TensorBoard logs to runs/
```

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir=runs
```

#### 5. **Evaluation Mode** - Fast Testing
Evaluate agents quickly without rendering:

```bash
# Run 1000 games quickly
python main.py --mode eval --games 1000

# Compare different strategies
python main.py --mode eval --games 500 --agent1 Baseline --agent2 Roller --threshold1 20 --threshold2 75
```

#### 6. **Benchmark Mode** - Comprehensive Testing
Run extensive benchmark tests across different strategy configurations:

```bash
python main.py --mode benchmark

# Creates matrices testing 15×15 threshold combinations
# Results saved to output/benchmark_*.csv and output/benchmark_*.npy
```

### Agent Types

**QTable** - Q-Learning Agent
- Trainable reinforcement learning agent
- Uses epsilon-greedy exploration
- Learns optimal policy through experience

**Baseline** - Threshold-Based Strategy
- Uses heuristic rule: roll until turn score exceeds threshold
- Specify threshold with `--threshold1` or `--threshold2`
- Example: `--agent1 Baseline --threshold1 25`

**Roller** - Probabilistic Strategy
- Random action selection based on probability
- Threshold represents % chance of rolling
- Example: `--agent2 Roller --threshold2 75` (75% roll, 25% pass)

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | string | `console` | Game mode: `console`, `interactive`, `human`, `train`, `benchmark`, `eval` |
| `--games` | int | 10 | Number of games to play (human/eval modes) |
| `--epochs` | int | 50 | Training epochs (train mode) |
| `--games-per-epoch` | int | 50 | Games per epoch (train mode) |
| `--agent1` | string | `QTable` | Agent 1 strategy: `QTable`, `Baseline`, `Roller` |
| `--agent2` | string | `Baseline` | Agent 2 strategy: `QTable`, `Baseline`, `Roller` |
| `--threshold1` | int | 25 | Threshold for Agent 1 (Baseline/Roller) |
| `--threshold2` | int | 20 | Threshold for Agent 2 (Baseline/Roller) |
| `--load-model` | string | None | Load saved Q-table (path without extension) |
| `--verbose` | flag | False | Print detailed game information |

### Examples

```bash
# Play in console mode (default)
python main.py

# Watch a trained model play (if you have a saved model)
python main.py --mode human --games 5 --load-model output/QT_0

# Train for 20 epochs
python main.py --mode train --epochs 20 --games-per-epoch 30

# Quick evaluation: Baseline(20) vs Baseline(30)
python main.py --mode eval --games 100 --agent1 Baseline --agent2 Baseline --threshold1 20 --threshold2 30

# Interactive play with pygame and verbose output
python main.py --mode interactive --verbose

# Benchmark different threshold strategies
python main.py --mode benchmark
```

### Output Files

Training and benchmark modes save results to the `output/` folder:

- `QT_0.json` - Q-table configuration
- `QT_0.npy` - Trained Q-table weights
- `benchmark_matrix_*.csv/npy` - Benchmark results matrix
- `benchmark_list_*.csv/npy` - Flattened benchmark data

TensorBoard logs are saved to `runs/` and can be viewed with:
```bash
tensorboard --logdir=runs
```



## License

This project is available for educational and research purposes.

## Acknowledgments

- Original Pass the Pigs game by David Moffatt and Nigel Hirst
- Gymnasium framework by Farama Foundation
- Reinforcement learning algorithms from Sutton & Barto



*need to analyze the final results using the graph examples in the pdf slideshow*

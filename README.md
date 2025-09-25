# Pass the Pigs: Advanced Reinforcement Learning Environment

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29.1+-green.svg)](https://gymnasium.farama.org/)

A comprehensive reinforcement learning implementation of the classic "Pass the Pigs"z game, featuring modern Gymnasium environments, Q-learning with curriculum training, and advanced visualization tools.

## Overview

This project implements Pass the Pigs as a reinforcement learning environment where agents learn optimal strategies through Q-learning with curriculum-based training. The implementation includes sophisticated analysis tools for strategy visualization and performance evaluation.

### Key Features

- **Modern Gymnasium Environment**: Full compatibility with the latest RL frameworks
- **Q-Learning with Curriculum Training**: Progressive difficulty scaling for enhanced learning
- **Advanced Visualization**: 3D strategy surfaces, contour plots, and performance analytics
- **Multiple Execution Modes**: Demo, analysis, and interactive play options
- **Sparse Reward Structure**: Efficient learning with strategic reward design
- **Professional Analysis Tools**: Comprehensive strategy comparison and evaluation

## Game Rules

Pass the Pigs is a strategic dice game where players:

1. Roll two pig-shaped dice to score points
2. Continue rolling to accumulate points in their turn
3. Choose to "bank" points or risk losing them on a bad roll
4. First player to reach 100 points wins

### Scoring System

| Roll Result | Points | Description |
|-------------|---------|-------------|
| Double Trotter | 40 | Both pigs on side with legs showing |
| Double Snouter | 30 | Both pigs on snout |
| Double Razorback | 20 | Both pigs on back |
| Double Leaning Jowler | 20 | Both pigs leaning on side |
| Piggyback | 20 | One pig on top of another |
| Leaning Jowler | 15 | Single pig leaning on side |
| Snouter | 10 | Single pig on snout |
| Razorback | 5 | Single pig on back |
| Trotter | 5 | Single pig with legs showing |
| Sider | 1 | Single pig on side |
| Pig Out | 0 | One pig on side, one on back (lose turn) |
| Oinker | 0 | Both pigs on different sides (lose turn) |

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone or download the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python run.py
```

This launches the interactive menu with multiple execution modes.

### Available Modes

1. **Demo Mode**: Watch trained agents play against each other
2. **Analysis Mode**: Generate comprehensive strategy visualizations
3. **Interactive Mode**: Play against the AI yourself

### Programming Interface

```python
from env import PassThePigsEnv, PassThePigsAgent
from analysis import PassThePigsAnalyzer

# Create environment
env = PassThePigsEnv()

# Train an agent
agent = PassThePigsAgent(mode='q_learning')
env.train_agent(agent, episodes=10000)

# Analyze strategies
analyzer = PassThePigsAnalyzer(env)
analyzer.plot_strategy_surface(agent)
```



## Architecture

### Core Components

- **`env.py`**: Main Gymnasium environment implementation
- **`analysis.py`**: Visualization and strategy analysis tools
- **`run.py`**: Multi-mode execution interface
- **`example.py`**: Basic usage demonstrations

### Environment Design

The environment uses a clean state representation:

- **State**: `(own_score, opponent_score, turn_total, current_player)`
- **Actions**: `0` (bank points), `1` (continue rolling)
- **Rewards**: Sparse structure (+1 win, -1 loss, 0 otherwise)

### Agent Architecture

The Q-learning agent features:

- **Curriculum Training**: Progressive opponent difficulty
- **Efficient Q-tables**: defaultdict implementation for large state spaces
- **Epsilon-greedy Exploration**: Balanced exploration vs exploitation
- **Threshold-based Strategies**: Intelligent banking decisions

## Analysis Tools

### Strategy Visualization

Generate 3D strategy surfaces showing banking decisions:

```python
from analysis import PassThePigsAnalyzer
analyzer = PassThePigsAnalyzer(env)
analyzer.plot_strategy_surface(agent)
```

### Performance Analysis

Compare multiple agents with statistical analysis:

```python
analyzer.compare_agents([agent1, agent2], num_games=1000)
```

### Learning Curves

Visualize training progress and convergence:

```python
analyzer.plot_training_progress(agent)
```



## Training Configuration

### Default Hyperparameters

- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.99
- **Exploration Rate (ε)**: 0.1 → 0.01 (decay)
- **Training Episodes**: 10,000
- **Curriculum Stages**: 4 difficulty levels

### Curriculum Training

The agent trains against progressively challenging opponents:

1. **Stage 1**: Always-rolling opponent (easy warmup)
2. **Stage 2**: Conservative opponent (threshold=10)
3. **Stage 3**: Moderate opponent (threshold=15)
4. **Stage 4**: Aggressive opponent (threshold=20)

## File Structure

```
pass-the-pig/
├── env.py              # Main environment and agent classes
├── analysis.py         # Visualization and analysis tools
├── run.py              # Main entry point with different modes
├── example.py          # Usage examples and tutorials
├── requirements.txt    # Dependencies
├── README.md          # This file
└── pass_the_pigs/     # Game assets (images, sounds)
```
## Research Applications

This implementation is ideal for:

- **Multi-agent RL Research**: Self-play and opponent modeling
- **Curriculum Learning Studies**: Progressive difficulty training
- **Strategy Analysis**: Game-theoretic investigation
- **Educational Purposes**: RL concept demonstration

## Performance

- **Training Time**: ~2-3 minutes for 10,000 episodes
- **Memory Usage**: Efficient sparse Q-table representation
- **Convergence**: Typically achieves stable strategies within 5,000 episodes

## Technical Notes

### State Space
- **Size**: Theoretically ~2M states (100×100×100×2)
- **Practical**: ~50K states encountered during typical training
- **Representation**: Tuple format for efficient hashing

### Action Space
- **Binary**: Bank (0) vs Continue (1)
- **Context-dependent**: Continue only available with valid roll

### Reward Engineering
- **Terminal Rewards**: +1 (win), -1 (loss)
- **Step Rewards**: 0 (sparse structure)
- **Rationale**: Encourages long-term strategic thinking

## Advanced Features

### Grid Search
Find optimal threshold strategies:

```python
analyzer = PassThePigsAnalyzer()
win_matrix, thresholds, fig = analyzer.grid_search_thresholds(
    max_threshold=30, step=5, num_games=1000
)
```

### Strategy Comparison
Compare multiple agents:

```python
agents = {
    'Random': PassThePigsAgent(mode='random'),
    'Threshold-15': PassThePigsAgent(mode='threshold', threshold=15),
    'Threshold-20': PassThePigsAgent(mode='threshold', threshold=20),
    'Q-Learning': trained_q_agent
}
analyzer.compare_agents(agents)
```

## Contributing

Areas for contribution:
- Additional RL algorithms (DQN, PPO, A3C)
- Enhanced opponent modeling
- Advanced visualization features
- Performance optimizations

## License

This project is available for educational and research purposes.

## Acknowledgments

- Original Pass the Pigs game by David Moffatt and Nigel Hirst
- Gymnasium framework by Farama Foundation
- Reinforcement learning algorithms from Sutton & Barto

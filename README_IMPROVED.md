# Pass the Pigs - Improved Implementation

This project contains an improved implementation of the Pass the Pigs reinforcement learning environment, incorporating insights from the Jupyter notebook analysis and modern best practices.

## What's New

### 🎯 **Improved Environment** (`improved_env.py`)

- **Cleaner state representation**: `(own_score, opp_score, turn_total, player_to_move)`
- **Simplified actions**: Just `0=roll`, `1=hold` (removed complex hog call system initially)
- **Better game mechanics**: More accurate probability distributions and cleaner scoring logic
- **Sparse rewards**: Uses +1 for win, -1 for loss, 0 otherwise (better for RL)
- **Modern Gymnasium interface**: Compatible with latest RL libraries

### 🧠 **Better Q-Learning** (`ImprovedPassThePigsAgent`)

- **Efficient Q-table**: Uses `defaultdict` for memory efficiency
- **Better discretization**: Smarter bucketing strategy for continuous states
- **Curriculum learning**: Progressive training against stronger opponents
- **Configurable parameters**: Easy to tune learning rate, epsilon, etc.

### 📊 **Advanced Analysis Tools** (`analysis.py`)

- **3D strategy surfaces**: Visualize hold thresholds across all game states
- **2D contour plots**: Clear view of strategy patterns
- **Strategy comparison**: Compare multiple agents side-by-side
- **Performance analysis**: Comprehensive evaluation against various opponents
- **Grid search**: Find optimal threshold strategies

### 🎮 **Multiple Agent Types**

- **Q-learning**: Trainable agent that learns optimal policy
- **Threshold**: Simple baseline that holds when turn_total >= threshold
- **Random**: Random policy for baseline comparisons

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_improved.txt
```

### 2. Run Basic Demo

```bash
python run.py --mode demo
```

This will:
- Train a Q-learning agent using curriculum learning
- Test it against a threshold baseline
- Show performance results

### 3. Generate Analysis Plots

```bash
python run.py --mode analysis
```

This creates comprehensive visualizations including:
- 3D strategy surface
- 2D contour plots
- Performance comparisons
- Hold threshold analysis

### 4. Watch Interactive Games

```bash
python run.py --mode interactive
```

This shows 5 games being played between agents with detailed turn-by-turn output.

## Examples

### Basic Usage

```python
from improved_env import ImprovedPassThePigsEnv, ImprovedPassThePigsAgent, play_match

# Create environment
env = ImprovedPassThePigsEnv()

# Create agents
agent1 = ImprovedPassThePigsAgent(mode='threshold', threshold=20)
agent2 = ImprovedPassThePigsAgent(mode='q_learning')

# Play games
win_rate = play_match(env, agent1, agent2, num_games=1000)
print(f"Agent1 win rate: {win_rate:.3f}")
```

### Training with Curriculum Learning

```python
from improved_env import curriculum_training

# Create and train Q-learning agent
q_agent = ImprovedPassThePigsAgent(mode='q_learning')
trained_agent = curriculum_training(env, q_agent, games_per_phase=5000)
```

### Analysis and Visualization

```python
from analysis import PassThePigsAnalyzer

# Create analyzer
analyzer = PassThePigsAnalyzer()

# Generate comprehensive analysis
fig = analyzer.create_comprehensive_analysis(trained_agent, "My Agent")

# Performance analysis
results, fig = analyzer.analyze_agent_performance(trained_agent)
```

## Key Improvements from Original

### 1. **Environment Design**
- **Original**: Complex state with hog calls, unclear reward structure
- **Improved**: Clean state representation, sparse rewards, focus on core game

### 2. **Q-Learning Implementation**
- **Original**: Fixed arrays, complex bucketing, hard-coded parameters
- **Improved**: Dynamic Q-table, smart discretization, curriculum learning

### 3. **Analysis Capabilities**
- **Original**: Basic win/loss tracking
- **Improved**: Comprehensive strategy analysis, 3D visualizations, performance metrics

### 4. **Code Structure**
- **Original**: Monolithic file, pygame dependencies
- **Improved**: Modular design, optional rendering, clean interfaces

## File Structure

```
improved_env.py      # Main environment and agent classes
analysis.py          # Visualization and analysis tools
run.py              # Main entry point with different modes
example.py          # Detailed examples and tutorials
requirements_improved.txt  # Dependencies
README_IMPROVED.md   # This file

# Original files (still available)
env.py              # Original implementation
pass_the_pigs.ipynb # Jupyter notebook analysis
```

## Advanced Features

### Curriculum Learning

The improved implementation includes curriculum learning that progressively trains against stronger opponents:

1. **Phase 1**: Always-rolling opponent (easy)
2. **Phase 2**: Threshold=10 opponent
3. **Phase 3**: Threshold=15 opponent
4. **Phase 4**: Threshold=20 opponent (challenging)

This approach leads to much better final performance than training against a fixed opponent.

### Strategy Analysis

The analysis tools can help you understand:

- **When to hold**: Optimal hold thresholds for any game state
- **Strategy patterns**: How decisions change based on scores
- **Performance**: Win rates against different opponent types
- **Comparison**: How different strategies compare visually

### Grid Search

Find optimal threshold strategies:

```python
analyzer = PassThePigsAnalyzer()
win_matrix, thresholds, fig = analyzer.grid_search_thresholds(
    max_threshold=30, step=5, num_games=1000
)
```

## Performance Tips

1. **Training**: Use curriculum learning for better results
2. **Evaluation**: Test against multiple opponent types
3. **Analysis**: Use visualizations to understand learned strategies
4. **Tuning**: Adjust learning rate, epsilon decay, and discretization

## Next Steps

This improved implementation provides a solid foundation for:

1. **Advanced RL algorithms**: Try DQN, Policy Gradient, Actor-Critic
2. **Multi-agent RL**: Implement self-play and population-based training
3. **Strategy analysis**: Deeper investigation of optimal play
4. **Game variants**: Add back hog calls or other rule variations

## Comparison with Jupyter Notebook

The improvements integrate the best insights from the Jupyter notebook:

- ✅ **Cleaner environment design**
- ✅ **Better probability distributions**
- ✅ **Sparse reward structure**
- ✅ **Curriculum learning approach**
- ✅ **Advanced visualization tools**
- ✅ **Strategy surface analysis**

The result is a more robust, analyzable, and extensible implementation that maintains the analytical depth of the notebook while providing a clean, reusable codebase.

---

*This improved implementation was created by integrating the best features from both the original environment and the Jupyter notebook analysis, resulting in a more powerful and user-friendly system for studying Pass the Pigs with reinforcement learning.*
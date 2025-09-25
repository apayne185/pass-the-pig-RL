"""
Simple example showing how to use the Pass the Pigs environment
"""

from env import PassThePigsEnv, PassThePigsAgent, play_match, curriculum_training
from analysis import PassThePigsAnalyzer
import matplotlib.pyplot as plt

def basic_example():
    """Basic usage example"""
    print("=== Basic Pass the Pigs Example ===")

    # Create environment
    env = PassThePigsEnv()

    # Create simple agents
    agent1 = PassThePigsAgent(mode='threshold', threshold=15)
    agent2 = PassThePigsAgent(mode='threshold', threshold=20)

    # Play some games
    win_rate = play_match(env, agent1, agent2, num_games=1000)
    print(f"Agent1 (threshold=15) vs Agent2 (threshold=20): {win_rate:.3f}")

    # Try different thresholds
    print("\nTesting different thresholds against threshold=20:")
    for threshold in [5, 10, 15, 20, 25, 30]:
        test_agent = PassThePigsAgent(mode='threshold', threshold=threshold)
        wr = play_match(env, test_agent, agent2, num_games=500)
        print(f"  Threshold {threshold:2d}: {wr:.3f}")

def training_example():
    """Q-learning training example"""
    print("\n=== Q-Learning Training Example ===")

    # Create environment and Q-learning agent
    env = PassThePigsEnv()
    q_agent = PassThePigsAgent(mode='q_learning')

    print("Training Q-learning agent...")
    print("  Initial epsilon:", q_agent.epsilon)

    # Train using curriculum learning
    q_agent = curriculum_training(env, q_agent, games_per_phase=3000)

    print(f"  Final epsilon: {q_agent.epsilon:.4f}")
    print(f"  Episodes trained: {q_agent.episode_count}")
    print(f"  Training win rate: {q_agent.get_win_rate():.3f}")

    # Test against various opponents
    print("\nTesting trained agent:")
    opponents = {
        'Random': PassThePigsAgent(mode='random'),
        'Threshold-15': PassThePigsAgent(mode='threshold', threshold=15),
        'Threshold-20': PassThePigsAgent(mode='threshold', threshold=20),
        'Threshold-25': PassThePigsAgent(mode='threshold', threshold=25),
    }

    for name, opponent in opponents.items():
        wr = play_match(env, q_agent, opponent, num_games=1000, training=False)
        print(f"  vs {name}: {wr:.3f}")

    return q_agent

def visualization_example():
    """Visualization example"""
    print("\n=== Visualization Example ===")

    # Train a Q-learning agent
    env = PassThePigsEnv()
    q_agent = PassThePigsAgent(mode='q_learning')
    q_agent = curriculum_training(env, q_agent, games_per_phase=2000)

    # Create analyzer
    analyzer = PassThePigsAnalyzer()

    print("Generating visualizations...")

    # 1. Strategy surface plot
    fig1 = analyzer.plot_3d_strategy_surface(q_agent, "Q-Learning Strategy (3D)")
    plt.show()

    # 2. Strategy contour plot
    fig2 = analyzer.plot_2d_strategy_contour(q_agent, "Q-Learning Strategy (2D)")
    plt.show()

    # 3. Strategy lines for different opponent scores
    fig3 = analyzer.plot_strategy_lines(q_agent, title="Q-Learning Hold Thresholds")
    plt.show()

    # 4. Compare with baseline strategies
    baseline_agents = [
        q_agent,
        PassThePigsAgent(mode='threshold', threshold=15),
        PassThePigsAgent(mode='threshold', threshold=20),
        PassThePigsAgent(mode='threshold', threshold=25)
    ]
    names = ['Q-Learning', 'Threshold-15', 'Threshold-20', 'Threshold-25']

    fig4 = analyzer.compare_strategies(baseline_agents, names)
    plt.show()

    # 5. Performance analysis
    results, fig5 = analyzer.analyze_agent_performance(q_agent, num_games=500)
    plt.show()

    print("Visualization complete!")

def grid_search_example():
    """Grid search example"""
    print("\n=== Grid Search Example ===")

    analyzer = PassThePigsAnalyzer()

    print("Running grid search for optimal thresholds...")
    win_matrix, thresholds, fig = analyzer.grid_search_thresholds(
        max_threshold=25, step=5, num_games=200
    )

    # Find best threshold
    best_idx = win_matrix.mean(axis=1).argmax()
    best_threshold = thresholds[best_idx]
    best_avg_win_rate = win_matrix.mean(axis=1)[best_idx]

    print(f"Best threshold: {best_threshold} (avg win rate: {best_avg_win_rate:.3f})")

    plt.show()

def comprehensive_example():
    """Comprehensive analysis example"""
    print("\n=== Comprehensive Analysis Example ===")

    # Train agent
    env = PassThePigsEnv()
    q_agent = PassThePigsAgent(mode='q_learning')
    q_agent = curriculum_training(env, q_agent, games_per_phase=2500)

    # Create analyzer and generate comprehensive analysis
    analyzer = PassThePigsAnalyzer()
    fig = analyzer.create_comprehensive_analysis(q_agent, "Final Q-Learning Agent")

    plt.show()

    print("Comprehensive analysis complete!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pass the Pigs Examples')
    parser.add_argument('--example', choices=['basic', 'training', 'visualization', 'grid_search', 'comprehensive', 'all'],
                       default='basic', help='Which example to run')

    args = parser.parse_args()

    if args.example == 'basic' or args.example == 'all':
        basic_example()

    if args.example == 'training' or args.example == 'all':
        training_example()

    if args.example == 'visualization' or args.example == 'all':
        visualization_example()

    if args.example == 'grid_search' or args.example == 'all':
        grid_search_example()

    if args.example == 'comprehensive' or args.example == 'all':
        comprehensive_example()

    print("\nAll examples complete!")
    print("\nYou can run specific examples with:")
    print("  python example.py --example basic")
    print("  python example.py --example training")
    print("  python example.py --example visualization")
    print("  python example.py --example grid_search")
    print("  python example.py --example comprehensive")

from improved_env import ImprovedPassThePigsEnv, ImprovedPassThePigsAgent, play_match, curriculum_training
from analysis import PassThePigsAnalyzer
import matplotlib.pyplot as plt

def demo_improved_environment():
    """Demo of the improved environment with better agents"""
    print("=== Pass the Pigs - Improved Environment Demo ===")

    # Create environment
    env = ImprovedPassThePigsEnv(render_mode="human")

    # Create agents
    q_agent = ImprovedPassThePigsAgent(mode='q_learning')
    baseline_agent = ImprovedPassThePigsAgent(mode='threshold', threshold=20)

    print("\n1. Training Q-learning agent with curriculum learning...")
    trained_agent = curriculum_training(env, q_agent, games_per_phase=2000)

    print(f"\nQ-learning agent trained! Episodes: {trained_agent.episode_count}")
    print(f"Training win rate: {trained_agent.get_win_rate():.3f}")

    print("\n2. Testing trained agent vs baseline...")
    win_rate = play_match(env, trained_agent, baseline_agent, num_games=1000, verbose=True)
    print(f"Trained agent win rate vs threshold-20: {win_rate:.3f}")

    return env, trained_agent, baseline_agent

def demo_analysis():
    """Demo of the analysis tools"""
    print("\n=== Strategy Analysis Demo ===")

    # Get trained agents
    env, trained_agent, baseline_agent = demo_improved_environment()

    # Create analyzer
    analyzer = PassThePigsAnalyzer()

    print("\n3. Generating comprehensive analysis...")
    fig = analyzer.create_comprehensive_analysis(trained_agent, "Trained Q-Learning Agent")
    plt.show()

    print("\n4. Comparing different strategies...")
    agents = [
        trained_agent,
        ImprovedPassThePigsAgent(mode='threshold', threshold=15),
        ImprovedPassThePigsAgent(mode='threshold', threshold=20),
        ImprovedPassThePigsAgent(mode='threshold', threshold=25),
    ]

    agent_names = ['Q-Learning', 'Threshold-15', 'Threshold-20', 'Threshold-25']
    fig2 = analyzer.compare_strategies(agents, agent_names)
    plt.show()

    print("\n5. Performance analysis...")
    results, fig3 = analyzer.analyze_agent_performance(trained_agent)
    plt.show()

    return trained_agent

def interactive_demo():
    """Interactive demo similar to original"""
    print("\n=== Interactive Demo ===")

    # Create environment (you can add pygame support later)
    env = ImprovedPassThePigsEnv(render_mode="human")

    # Get a trained agent
    q_agent = ImprovedPassThePigsAgent(mode='q_learning')
    q_agent = curriculum_training(env, q_agent, games_per_phase=1000)

    print("\nPlaying 5 games between trained Q-agent and threshold-20 agent:")

    baseline_agent = ImprovedPassThePigsAgent(mode='threshold', threshold=20)

    for game in range(5):
        print(f"\n--- Game {game + 1} ---")
        obs, _ = env.reset()
        done = False
        turn = 0

        while not done:
            current_player = obs[3]
            agent_name = "Q-Learning" if current_player == 0 else "Threshold-20"

            if current_player == 0:
                action = q_agent.predict(obs, training=False)
            else:
                action = baseline_agent.predict(obs, training=False)

            action_name = "ROLL" if action == 0 else "HOLD"
            print(f"Turn {turn}: Player {current_player} ({agent_name}) - {action_name}")

            obs, reward, done, _, info = env.step(action)
            env.render()

            turn += 1

            if done:
                winner = info.get("winner", -1)
                reason = info.get("reason", "unknown")
                print(f"Game over! Winner: Player {winner} (Reason: {reason})")
                break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pass the Pigs - Improved Version')
    parser.add_argument('--mode', choices=['demo', 'analysis', 'interactive'],
                       default='demo', help='Run mode')

    args = parser.parse_args()

    if args.mode == 'demo':
        demo_improved_environment()
    elif args.mode == 'analysis':
        demo_analysis()
    elif args.mode == 'interactive':
        interactive_demo()

    print("\nDemo complete! Try running with different modes:")
    print("  python run.py --mode demo       # Basic training and testing")
    print("  python run.py --mode analysis   # Full analysis with plots")
    print("  python run.py --mode interactive # Watch games being played")

"""
Visualization and Analysis Tools for Pass the Pigs
Based on the analysis from the Jupyter notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from improved_env import ImprovedPassThePigsEnv, ImprovedPassThePigsAgent, play_match
import seaborn as sns

class PassThePigsAnalyzer:
    """
    Analysis and visualization tools for Pass the Pigs strategies
    """

    def __init__(self, target_score=100):
        self.target_score = target_score
        self.env = ImprovedPassThePigsEnv(target_score=target_score)

    def extract_hold_threshold(self, agent, own_score, opp_score, max_turn=30):
        """
        Extract the hold threshold from an agent's policy for given scores
        Returns the minimum turn score where holding becomes preferred
        """
        if not hasattr(agent, 'q_table'):
            # For threshold agents, return their threshold
            return getattr(agent, 'threshold', 20)

        for turn_score in range(1, max_turn + 1):
            obs = np.array([own_score, opp_score, turn_score, 0])
            state = agent.discretize_state(obs)

            # Check if hold (action 1) is preferred over roll (action 0)
            if agent.q_table[state][1] > agent.q_table[state][0]:
                return turn_score

        return max_turn  # Never prefer holding

    def compute_strategy_surface(self, agent, score_step=2):
        """
        Compute hold threshold surface for visualization
        """
        score_range = np.arange(0, self.target_score, score_step)
        surface = np.zeros((len(score_range), len(score_range)))

        for i, own_score in enumerate(score_range):
            for j, opp_score in enumerate(score_range):
                if own_score < self.target_score and opp_score < self.target_score:
                    threshold = self.extract_hold_threshold(agent, own_score, opp_score)
                    surface[i, j] = threshold
                else:
                    surface[i, j] = np.nan

        return score_range, surface

    def plot_3d_strategy_surface(self, agent, title="Hold Strategy: 3D Surface"):
        """
        Create 3D surface plot of hold strategy
        """
        score_range, surface = self.compute_strategy_surface(agent)
        Own, Opp = np.meshgrid(score_range, score_range)

        # Mask invalid states
        valid_mask = (Own < self.target_score) & (Opp < self.target_score)
        surface_masked = np.where(valid_mask.T, surface, np.nan)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(Own, Opp, surface_masked.T, cmap='viridis', alpha=0.8)

        ax.set_xlabel('My Score')
        ax.set_ylabel('Opponent Score')
        ax.set_zlabel('Hold Threshold')
        ax.set_title(title)

        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=30)
        cbar.set_label('Hold Threshold')

        plt.tight_layout()
        return fig

    def plot_2d_strategy_contour(self, agent, title="Hold Strategy: 2D Contour"):
        """
        Create 2D contour plot of hold strategy
        """
        score_range, surface = self.compute_strategy_surface(agent)
        Own, Opp = np.meshgrid(score_range, score_range)

        valid_mask = (Own < self.target_score) & (Opp < self.target_score)
        surface_masked = np.where(valid_mask.T, surface, np.nan)

        fig, ax = plt.subplots(figsize=(10, 8))

        contour = ax.contourf(Own, Opp, surface_masked.T, levels=20, cmap='viridis')
        ax.contour(Own, Opp, surface_masked.T, levels=20, colors='black', alpha=0.3, linewidths=0.5)

        ax.set_xlabel('My Score')
        ax.set_ylabel('Opponent Score')
        ax.set_title(title)

        cbar = plt.colorbar(contour)
        cbar.set_label('Hold Threshold')

        plt.tight_layout()
        return fig

    def plot_strategy_lines(self, agent, opp_scores=[0, 25, 50, 75, 90, 95],
                           title="Hold Threshold vs My Score"):
        """
        Plot hold threshold as lines for different opponent scores
        """
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

        plt.figure(figsize=(12, 8))

        for opp_score, color in zip(opp_scores, colors):
            own_scores = []
            thresholds = []

            for own_score in range(0, self.target_score, 2):
                if own_score < self.target_score and opp_score < self.target_score:
                    threshold = self.extract_hold_threshold(agent, own_score, opp_score)
                    own_scores.append(own_score)
                    thresholds.append(threshold)

            plt.plot(own_scores, thresholds,
                    label=f'Opponent Score = {opp_score}',
                    color=color, linewidth=2)

        plt.xlabel('My Score')
        plt.ylabel('Hold Threshold')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def compare_strategies(self, agents, agent_names, score_step=10):
        """
        Create comparison plot showing multiple strategies
        """
        n_agents = len(agents)
        fig, axes = plt.subplots(2, (n_agents + 1) // 2, figsize=(16, 10))
        axes = axes.flatten()

        for i, (agent, name) in enumerate(zip(agents, agent_names)):
            score_range, surface = self.compute_strategy_surface(agent, score_step)
            Own, Opp = np.meshgrid(score_range, score_range)

            valid_mask = (Own < self.target_score) & (Opp < self.target_score)
            surface_masked = np.where(valid_mask.T, surface, np.nan)

            im = axes[i].imshow(surface_masked.T, aspect='auto', cmap='viridis',
                               origin='lower', extent=[0, self.target_score, 0, self.target_score])
            axes[i].set_xlabel('My Score')
            axes[i].set_ylabel('Opponent Score')
            axes[i].set_title(name)

            plt.colorbar(im, ax=axes[i], label='Hold Threshold')

        # Remove extra subplots
        for i in range(len(agents), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        return fig

    def analyze_agent_performance(self, agent, num_games=1000):
        """
        Comprehensive performance analysis of an agent
        """
        # Test against various opponents
        opponents = {
            'Random': ImprovedPassThePigsAgent(mode='random'),
            'Threshold-10': ImprovedPassThePigsAgent(mode='threshold', threshold=10),
            'Threshold-15': ImprovedPassThePigsAgent(mode='threshold', threshold=15),
            'Threshold-20': ImprovedPassThePigsAgent(mode='threshold', threshold=20),
            'Threshold-25': ImprovedPassThePigsAgent(mode='threshold', threshold=25),
        }

        results = {}

        print("Performance Analysis:")
        print("-" * 40)

        for opp_name, opponent in opponents.items():
            win_rate = play_match(self.env, agent, opponent, num_games, training=False)
            results[opp_name] = win_rate
            print(f"vs {opp_name:12}: {win_rate:.3f}")

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))

        opponents_list = list(results.keys())
        win_rates = list(results.values())

        bars = ax.bar(opponents_list, win_rates, color='skyblue', edgecolor='navy')
        ax.set_ylabel('Win Rate')
        ax.set_title('Agent Performance vs Different Opponents')
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, rate in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.3f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()

        return results, fig

    def grid_search_thresholds(self, max_threshold=30, step=5, num_games=500):
        """
        Grid search to find optimal threshold strategies
        """
        thresholds = list(range(step, max_threshold + 1, step))
        n = len(thresholds)
        win_matrix = np.zeros((n, n))

        print("Grid search for optimal thresholds...")

        for i, thresh1 in enumerate(thresholds):
            for j, thresh2 in enumerate(thresholds):
                agent1 = ImprovedPassThePigsAgent(mode='threshold', threshold=thresh1)
                agent2 = ImprovedPassThePigsAgent(mode='threshold', threshold=thresh2)

                win_rate = play_match(self.env, agent1, agent2, num_games, training=False)
                win_matrix[i, j] = win_rate

                print(f"Threshold {thresh1:2d} vs {thresh2:2d}: {win_rate:.3f}")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(win_matrix,
                   xticklabels=thresholds,
                   yticklabels=thresholds,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0.5,
                   ax=ax)

        ax.set_xlabel('Opponent Threshold')
        ax.set_ylabel('My Threshold')
        ax.set_title('Win Rate Matrix: Threshold vs Threshold')

        plt.tight_layout()

        return win_matrix, thresholds, fig

    def create_comprehensive_analysis(self, agent, agent_name="Trained Agent"):
        """
        Create a comprehensive 2x2 analysis plot combining multiple views
        """
        fig = plt.figure(figsize=(16, 12))

        # 3D Surface (subplot 1)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        score_range, surface = self.compute_strategy_surface(agent)
        Own, Opp = np.meshgrid(score_range, score_range)
        valid_mask = (Own < self.target_score) & (Opp < self.target_score)
        surface_masked = np.where(valid_mask.T, surface, np.nan)

        surf = ax1.plot_surface(Own, Opp, surface_masked.T, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('My Score')
        ax1.set_ylabel('Opponent Score')
        ax1.set_zlabel('Hold Threshold')
        ax1.set_title('3D Strategy Surface')
        ax1.view_init(elev=20, azim=45)

        # 2D Contour (subplot 2)
        ax2 = fig.add_subplot(2, 2, 2)
        contour = ax2.contourf(Own, Opp, surface_masked.T, levels=20, cmap='viridis')
        ax2.contour(Own, Opp, surface_masked.T, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        ax2.set_xlabel('My Score')
        ax2.set_ylabel('Opponent Score')
        ax2.set_title('2D Contour View')

        # Strategy Lines (subplot 3)
        ax3 = fig.add_subplot(2, 2, 3)
        opp_scores = [0, 25, 50, 75, 90, 95]
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

        for opp_score, color in zip(opp_scores, colors):
            own_scores = []
            thresholds = []

            for own_score in range(0, self.target_score, 2):
                if own_score < self.target_score and opp_score < self.target_score:
                    threshold = self.extract_hold_threshold(agent, own_score, opp_score)
                    own_scores.append(own_score)
                    thresholds.append(threshold)

            ax3.plot(own_scores, thresholds, label=f'Opp={opp_score}',
                    color=color, linewidth=1.5)

        ax3.set_xlabel('My Score')
        ax3.set_ylabel('Hold Threshold')
        ax3.set_title('Strategy by Opponent Score')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Performance Analysis (subplot 4)
        ax4 = fig.add_subplot(2, 2, 4)

        # Quick performance test
        opponents = ['Random', 'Thresh-10', 'Thresh-15', 'Thresh-20', 'Thresh-25']
        win_rates = []

        for i, thresh in enumerate([None, 10, 15, 20, 25]):
            if thresh is None:
                opp = ImprovedPassThePigsAgent(mode='random')
            else:
                opp = ImprovedPassThePigsAgent(mode='threshold', threshold=thresh)

            wr = play_match(self.env, agent, opp, 200, training=False)
            win_rates.append(wr)

        bars = ax4.bar(opponents, win_rates, color='skyblue', edgecolor='navy')
        ax4.set_ylabel('Win Rate')
        ax4.set_title('Performance vs Opponents')
        ax4.set_ylim(0, 1)

        # Add value labels
        for bar, rate in zip(bars, win_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')

        plt.xticks(rotation=45)

        # Add colorbars
        plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)

        cbar2 = fig.colorbar(contour, ax=ax2, shrink=0.8, aspect=20)
        cbar2.set_label('Hold Threshold', fontsize=10)

        plt.suptitle(f'Complete Strategy Analysis: {agent_name}', fontsize=16, y=0.95)

        return fig


def demo_analysis():
    """
    Demonstration of the analysis tools
    """
    print("Running Pass the Pigs Analysis Demo...")

    # Create analyzer
    analyzer = PassThePigsAnalyzer()

    # Create and train a Q-learning agent
    from improved_env import curriculum_training
    env = ImprovedPassThePigsEnv()
    q_agent = ImprovedPassThePigsAgent(mode='q_learning')

    print("Training Q-learning agent (quick demo)...")
    q_agent = curriculum_training(env, q_agent, games_per_phase=1000)

    # Create baseline agents for comparison
    agents = [
        q_agent,
        ImprovedPassThePigsAgent(mode='threshold', threshold=15),
        ImprovedPassThePigsAgent(mode='threshold', threshold=20),
        ImprovedPassThePigsAgent(mode='threshold', threshold=25),
    ]

    agent_names = ['Q-Learning', 'Threshold-15', 'Threshold-20', 'Threshold-25']

    # Generate comprehensive analysis
    fig = analyzer.create_comprehensive_analysis(q_agent, "Q-Learning Agent")
    plt.show()

    # Compare strategies
    fig2 = analyzer.compare_strategies(agents, agent_names)
    plt.show()

    # Performance analysis
    results, fig3 = analyzer.analyze_agent_performance(q_agent)
    plt.show()

    print("Analysis complete!")


if __name__ == "__main__":
    demo_analysis()
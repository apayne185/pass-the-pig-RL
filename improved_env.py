"""
Improved Pass the Pigs Environment
Combines the best features from the original env.py and the Jupyter notebook implementation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from typing import Optional, Dict, Tuple
import random
import copy

# Game constants
TARGET_SCORE = 100
NUM_PLAYERS = 2

# Actions
ROLL = 0
HOLD = 1
ACTIONS = ['ROLL', 'HOLD']

# Single-pig outcomes and scores (from notebook - more accurate probabilities)
SINGLE_PIG_OUTCOMES = {
    "side_dot": {"p": 0.33, "score": 0},
    "side_pink": {"p": 0.33, "score": 0},
    "razorback": {"p": 0.18, "score": 5},
    "trotter": {"p": 0.09, "score": 5},
    "snouter": {"p": 0.05, "score": 10},
    "leaning_jowler": {"p": 0.02, "score": 15},
}

# Special two-pig events probabilities
PIGGYBACK_P = 0.005   # Player eliminated
OINKER_P = 0.02       # Score wiped to zero

class ImprovedPassThePigsEnv(gym.Env):
    """
    Improved Pass the Pigs Environment with cleaner implementation
    State: (own_score, opp_score, turn_total, player_to_move)
    Actions: 0=roll, 1=hold
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, target_score=TARGET_SCORE, render_mode=None):
        super().__init__()

        self.target_score = target_score
        self.render_mode = render_mode

        # Action space: 0=roll, 1=hold
        self.action_space = spaces.Discrete(2)

        # Observation space: (own_score, opp_score, turn_total, player)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([target_score, target_score, target_score, 1]),
            dtype=np.int32
        )

        # Initialize random number generator
        self.rng = np.random.default_rng()

        # Game state
        self.reset()

    def _sample_one_pig(self):
        """Sample outcome for a single pig"""
        names = list(SINGLE_PIG_OUTCOMES.keys())
        probs = [SINGLE_PIG_OUTCOMES[k]["p"] for k in names]
        # Normalize probabilities
        probs = np.array(probs) / sum(probs)

        outcome = self.rng.choice(names, p=probs)
        return outcome, SINGLE_PIG_OUTCOMES[outcome]["score"]

    def _roll_two_pigs(self):
        """
        Roll two pigs and return points earned and any special events
        Returns: (points, flags_dict)
        """
        # Check for special events first
        if self.rng.random() < PIGGYBACK_P:
            return 0, {"piggyback": True}

        if self.rng.random() < OINKER_P:
            return 0, {"oinker": True}

        # Roll two individual pigs
        pig1, score1 = self._sample_one_pig()
        pig2, score2 = self._sample_one_pig()

        # Check for Pig Out (opposite sides)
        if (("side_dot" in (pig1, pig2)) and ("side_pink" in (pig1, pig2)) and
            pig1 != pig2):
            return 0, {"pig_out": True}

        # Sider: same side → 1 point
        if pig1 == pig2 and pig1 in ("side_dot", "side_pink"):
            return 1, {}

        # Neither is side → add scores, double if same non-side
        if (pig1 not in ("side_dot", "side_pink") and
            pig2 not in ("side_dot", "side_pink")):
            base_score = score1 + score2
            if pig1 == pig2:
                base_score *= 2  # Double for matching non-side outcomes
            return base_score, {}

        # One side + one scoring outcome
        if ((pig1 in ("side_dot", "side_pink")) !=
            (pig2 in ("side_dot", "side_pink"))):
            return score1 + score2, {}

        return 0, {}

    def _get_obs(self):
        """Get observation for current player"""
        own_score = self.scores[self.current_player]
        opp_score = self.scores[1 - self.current_player]
        return np.array([own_score, opp_score, self.turn_total, self.current_player],
                       dtype=np.int32)

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.scores = [0, 0]  # Scores for both players
        self.turn_total = 0   # Points accumulated in current turn
        self.current_player = 0  # Which player's turn it is
        self.last_roll_info = ""

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        """Execute one step in the environment"""
        if action not in [ROLL, HOLD]:
            raise ValueError(f"Invalid action: {action}")

        done = False
        reward = 0
        info = {}

        if action == ROLL:
            # Roll the pigs
            points, flags = self._roll_two_pigs()
            self.last_roll_info = f"Rolled: {points} points"

            if "piggyback" in flags:
                # Current player eliminated (loses immediately)
                reward = -1
                done = True
                info = {"reason": "piggyback", "winner": 1 - self.current_player}

            elif "oinker" in flags:
                # Current player's total score wiped out
                self.scores[self.current_player] = 0
                self.turn_total = 0
                self.current_player = 1 - self.current_player
                self.last_roll_info += " - OINKER! Score wiped to 0!"

            elif "pig_out" in flags:
                # Turn ends, no points added
                self.turn_total = 0
                self.current_player = 1 - self.current_player
                self.last_roll_info += " - PIG OUT! Turn ends."

            else:
                # Normal roll - add points to turn total
                self.turn_total += points

        else:  # HOLD
            # Add turn total to player's score and switch players
            self.scores[self.current_player] += self.turn_total
            self.turn_total = 0

            # Check if current player won
            if self.scores[self.current_player] >= self.target_score:
                reward = 1
                done = True
                info = {"reason": "target_reached", "winner": self.current_player}
            else:
                # Switch to other player
                self.current_player = 1 - self.current_player

        observation = self._get_obs()

        return observation, reward, done, False, info

    def render(self):
        """Render the current state"""
        if self.render_mode == "human":
            print(f"\n--- Pass the Pigs ---")
            print(f"Player 0 Score: {self.scores[0]}")
            print(f"Player 1 Score: {self.scores[1]}")
            print(f"Current Player: {self.current_player}")
            print(f"Turn Total: {self.turn_total}")
            if self.last_roll_info:
                print(f"Last Roll: {self.last_roll_info}")
            print("-" * 20)

    def close(self):
        """Clean up resources"""
        pass


class ImprovedPassThePigsAgent:
    """
    Improved agent implementation with better Q-learning
    """

    def __init__(self, mode='q_learning', threshold=20, bucket_size=5):
        self.mode = mode
        self.threshold = threshold
        self.bucket_size = bucket_size

        if mode == 'q_learning':
            # Use defaultdict for more efficient Q-table
            self.q_table = defaultdict(lambda: np.zeros(2))

            # Learning parameters
            self.learning_rate = 0.1
            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995

            # Statistics
            self.episode_count = 0
            self.wins = 0

    def discretize_state(self, obs):
        """Discretize continuous state into buckets"""
        own_score, opp_score, turn_total, player = obs

        # Bucket the scores
        own_bucket = min(own_score // self.bucket_size, 19)  # Max 19 buckets
        opp_bucket = min(opp_score // self.bucket_size, 19)
        turn_bucket = min(turn_total // self.bucket_size, 19)

        return (own_bucket, opp_bucket, turn_bucket, player)

    def predict(self, obs, training=True):
        """Predict action given observation"""
        if self.mode == 'q_learning':
            state = self.discretize_state(obs)

            # Epsilon-greedy action selection
            if training and np.random.random() < self.epsilon:
                return np.random.randint(0, 2)
            else:
                return np.argmax(self.q_table[state])

        elif self.mode == 'threshold':
            # Simple threshold policy: hold if turn_total >= threshold
            _, _, turn_total, _ = obs
            return HOLD if turn_total >= self.threshold else ROLL

        elif self.mode == 'random':
            return np.random.randint(0, 2)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def learn(self, old_obs, action, reward, new_obs, done):
        """Update Q-table based on experience"""
        if self.mode != 'q_learning':
            return

        old_state = self.discretize_state(old_obs)
        new_state = self.discretize_state(new_obs)

        # Q-learning update
        old_q_value = self.q_table[old_state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[new_state])

        # Update Q-value
        self.q_table[old_state][action] = old_q_value + self.learning_rate * (target - old_q_value)

        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_count += 1
            if reward > 0:
                self.wins += 1

    def get_win_rate(self):
        """Get current win rate"""
        if self.episode_count == 0:
            return 0.0
        return self.wins / self.episode_count


def play_match(env, agent1, agent2, num_games=1000, training=False, verbose=False):
    """
    Play multiple games between two agents
    Returns win rate for agent1
    """
    agent1_wins = 0

    for game in range(num_games):
        obs, _ = env.reset()
        done = False

        # Store old observations for learning
        old_obs = [None, None]
        actions = [None, None]

        while not done:
            current_player = obs[3]

            # Get action from current player's agent
            if current_player == 0:
                action = agent1.predict(obs, training)
            else:
                action = agent2.predict(obs, training)

            # Store for learning
            old_obs[current_player] = obs.copy()
            actions[current_player] = action

            # Take step
            new_obs, reward, done, _, info = env.step(action)

            # Learn from experience if training
            if training:
                if current_player == 0 and hasattr(agent1, 'learn'):
                    agent1.learn(old_obs[0], actions[0], reward, new_obs, done)
                elif current_player == 1 and hasattr(agent2, 'learn'):
                    # Flip reward for opponent
                    opp_reward = -reward if done else 0
                    agent2.learn(old_obs[1], actions[1], opp_reward, new_obs, done)

            obs = new_obs

        # Count wins for agent1
        if done and "winner" in info and info["winner"] == 0:
            agent1_wins += 1

        if verbose and (game + 1) % 100 == 0:
            win_rate = agent1_wins / (game + 1)
            print(f"Games {game + 1}: Agent1 win rate = {win_rate:.3f}")

    return agent1_wins / num_games


def curriculum_training(env, agent, num_phases=4, games_per_phase=5000):
    """
    Train agent using curriculum learning with progressively stronger opponents
    """
    print("Starting curriculum training...")

    # Phase 1: Train against always-rolling opponent
    print("Phase 1: Training against always-rolling opponent...")
    opponent = ImprovedPassThePigsAgent(mode='threshold', threshold=100)  # Always roll
    win_rate = play_match(env, agent, opponent, games_per_phase, training=True)
    print(f"Win rate vs always-rolling: {win_rate:.3f}")

    # Phase 2: Train against threshold=10 opponent
    print("Phase 2: Training against threshold=10 opponent...")
    opponent = ImprovedPassThePigsAgent(mode='threshold', threshold=10)
    win_rate = play_match(env, agent, opponent, games_per_phase, training=True)
    print(f"Win rate vs threshold=10: {win_rate:.3f}")

    # Phase 3: Train against threshold=15 opponent
    print("Phase 3: Training against threshold=15 opponent...")
    opponent = ImprovedPassThePigsAgent(mode='threshold', threshold=15)
    win_rate = play_match(env, agent, opponent, games_per_phase, training=True)
    print(f"Win rate vs threshold=15: {win_rate:.3f}")

    # Phase 4: Train against threshold=20 opponent
    print("Phase 4: Training against threshold=20 opponent...")
    opponent = ImprovedPassThePigsAgent(mode='threshold', threshold=20)
    win_rate = play_match(env, agent, opponent, int(games_per_phase * 1.5), training=True)
    print(f"Win rate vs threshold=20: {win_rate:.3f}")

    print("Curriculum training complete!")
    return agent


if __name__ == "__main__":
    # Test the improved environment
    env = ImprovedPassThePigsEnv(render_mode="human")

    # Create agents
    q_agent = ImprovedPassThePigsAgent(mode='q_learning')
    baseline_agent = ImprovedPassThePigsAgent(mode='threshold', threshold=20)

    # Train Q-learning agent using curriculum
    q_agent = curriculum_training(env, q_agent)

    # Evaluate final performance
    print("\nFinal evaluation:")
    final_win_rate = play_match(env, q_agent, baseline_agent, 1000, training=False, verbose=True)
    print(f"Final win rate vs baseline: {final_win_rate:.3f}")

    env.close()
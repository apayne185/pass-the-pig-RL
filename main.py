"""
Pass the Pigs - SEQUENTIAL VARIANT Main Runner Script
This script sets up and runs the Sequential variant of Pass the Pigs.
The Sequential variant allows agents to observe opponent's last action and points scored.

Usage:
    python main.py [--mode MODE] [--games NUM_GAMES] [--render] [--training]

Modes:
    - console: Play the game in text-based console mode (no pygame)
    - interactive: Play the game manually using pygame keyboard controls
    - human: Watch AI agents play with visual rendering
    - benchmark: Run benchmark tests between different agent strategies
    - train: Train a Q-learning agent
"""

import argparse
import time
import numpy as np
from env import (
    PassThePigs_2Players_Sequential_Env,
    PassThePigsAgent_Sequential,
    RENDER_MODE,
    TRAINING,
    RENDER_DELAY,
    NUM_PLAYERS,
    OWN_SCORE,
    OPP_SCORE,
    TURN_SCORE,
    HOG_CALL,
    OPP_LAST_ACTION,
    OPP_LAST_POINTS,
    WITH_HOG_CALLS,
    save_QT_model,
    load_QT_model,
    writer,
    model_name
)


def play_single_game(env, render=True, training=False, verbose=True):
    """
    Play a single game between two agents.

    Args:
        env: The Pass the Pigs Sequential environment
        render: Whether to render the game visually
        training: Whether agents should learn from the game
        verbose: Whether to print game information

    Returns:
        winner: The index of the winning player (0 or 1)
        game_info: Dictionary containing game statistics
    """
    obs, info = env.reset()
    done = False
    step_count = 0

    if render and env.render_mode:
        time.sleep(RENDER_DELAY)
        env.render()

    while not done:
        # Process pygame events to prevent freezing
        if env.render_mode and env.render_mode in ['human', 'interactive']:
            import pygame as pg
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    raise KeyboardInterrupt("Window closed by user")

        # Get action from current player's agent
        action = env.get_action(training)

        # Execute action in environment
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        # Training: update agent Q-tables based on experience
        if training:
            for k in range(NUM_PLAYERS):
                # Sparse rewards: only on terminal states
                rew = 0
                if done:
                    if env.winner == k:
                        rew = 1.0  # Win reward
                    else:
                        rew = -1.0  # Loss penalty

                # Update agent if there's a reward
                if rew != 0:
                    env.agents[k].learn(
                        env.old_obs[k],
                        env.last_actions[k],
                        env.new_obs[k],
                        rew,
                        done,
                        truncated,
                        info,
                        k  # Agent index for logging
                    )

        # Render the game state
        if render and env.render_mode:
            time.sleep(max(RENDER_DELAY, 0.5))
            env.render()

        # Game ended
        if done:
            if env.winner < 0:
                raise ValueError(f'Invalid winner: {env.winner}')

            if verbose:
                print(f"\nGame finished!")
                print(f"Winner: Player {env.winner + 1}")
                print(env.last_game_info)
                print(f"Final scores: Player 1: {env.players[0][OWN_SCORE]}, "
                      f"Player 2: {env.players[1][OWN_SCORE]}")
                print(f"Total steps: {step_count}\n")

            if render and env.render_mode:
                env.render()
                time.sleep(RENDER_DELAY * 2)

            break

    game_info = {
        'winner': env.winner,
        'steps': step_count,
        'final_scores': [env.players[0][OWN_SCORE], env.players[1][OWN_SCORE]],
        'reason': info.get('reason', 'Unknown')
    }

    return env.winner, game_info


def play_multiple_games(env, num_games=100, render=False, training=False, verbose=False):
    """
    Play multiple games and collect statistics.

    Args:
        env: The Pass the Pigs Sequential environment
        num_games: Number of games to play
        render: Whether to render games visually
        training: Whether agents should learn
        verbose: Whether to print detailed information

    Returns:
        stats: Dictionary containing game statistics
    """
    wins = [0, 0]
    total_steps = 0
    total_scores = [0, 0]
    game_count = 0

    print(f"\nPlaying {num_games} games...")
    print(f"Agent 1: {env.agents[0].mode}")
    print(f"Agent 2: {env.agents[1].mode}")
    print("-" * 50)

    # Show progress more frequently for large game counts
    progress_interval = 1 if num_games <= 10 else (10 if num_games <= 100 else 100)

    for game_num in range(num_games):
        winner, game_info = play_single_game(env, render=render, training=training, verbose=False)

        # Update statistics
        wins[winner] += 1
        total_steps += game_info['steps']
        total_scores[0] += game_info['final_scores'][0]
        total_scores[1] += game_info['final_scores'][1]
        game_count += 1

        # Print progress
        if (game_num + 1) % progress_interval == 0 or verbose or (game_num + 1) == num_games:
            win_rate_p1 = (wins[0] / game_count) * 100
            win_rate_p2 = (wins[1] / game_count) * 100
            print(f"Game {game_num + 1}/{num_games} | "
                  f"P1 wins: {wins[0]} ({win_rate_p1:.1f}%) | "
                  f"P2 wins: {wins[1]} ({win_rate_p2:.1f}%)")

    # Calculate final statistics
    stats = {
        'total_games': num_games,
        'wins': wins,
        'win_rates': [(wins[0] / num_games) * 100, (wins[1] / num_games) * 100],
        'avg_steps': total_steps / num_games,
        'avg_scores': [total_scores[0] / num_games, total_scores[1] / num_games]
    }

    # Print summary
    print("\n" + "=" * 50)
    print("FINAL STATISTICS (SEQUENTIAL VARIANT)")
    print("=" * 50)
    print(f"Total games played: {stats['total_games']}")
    print(f"Player 1 wins: {stats['wins'][0]} ({stats['win_rates'][0]:.2f}%)")
    print(f"Player 2 wins: {stats['wins'][1]} ({stats['win_rates'][1]:.2f}%)")
    print(f"Average steps per game: {stats['avg_steps']:.1f}")
    print(f"Average final scores: P1={stats['avg_scores'][0]:.1f}, P2={stats['avg_scores'][1]:.1f}")
    print("=" * 50 + "\n")

    return stats


def run_training_session(env, num_epochs=100, games_per_epoch=50):
    """
    Run a training session for Q-learning agent.

    Args:
        env: The Pass the Pigs Sequential environment
        num_epochs: Number of training epochs
        games_per_epoch: Number of games per epoch

    Returns:
        training_stats: Dictionary containing training statistics
    """
    print("\n" + "=" * 60)
    print("TRAINING Q-LEARNING AGENT - SEQUENTIAL VARIANT")
    print("=" * 60)

    # Set up agents
    env.agents[0] = PassThePigsAgent_Sequential(mode='QTable')
    env.agents[1] = PassThePigsAgent_Sequential(mode='Baseline', threshold=20)

    print(f"\nTraining configuration:")
    print(f"  Variant: SEQUENTIAL (with opponent action observation)")
    print(f"  Agent 1: Q-Learning (training)")
    print(f"  Agent 2: Baseline (threshold=20)")
    print(f"  Epochs: {num_epochs}")
    print(f"  Games per epoch: {games_per_epoch}")
    print(f"  Total games: {num_epochs * games_per_epoch}")
    print("-" * 60)

    epoch_stats = []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Play games for this epoch
        stats = play_multiple_games(
            env,
            num_games=games_per_epoch,
            render=False,
            training=True,
            verbose=False
        )

        epoch_stats.append(stats)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.1f}s")
        print(f"  Win rate: P1={stats['win_rates'][0]:.1f}%, P2={stats['win_rates'][1]:.1f}%")
        print(f"  Epsilon: {env.agents[0].epsilon:.4f}")
        print(f"  Learning rate: {env.agents[0].learning_rate:.4f}")
        print(f"  Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_name = f"{model_name}_epoch{epoch + 1}"
            save_QT_model(env.agents[0], checkpoint_name)
            print(f"  Checkpoint saved: {checkpoint_name}")

    # Save final model
    print("\n" + "-" * 60)
    print("Training completed!")
    save_QT_model(env.agents[0], model_name)
    print(f"Final model saved: {model_name}")
    print("=" * 60 + "\n")

    return epoch_stats


def run_console_mode(env, agent_mode='Baseline', agent_threshold=20, load_model=None):
    """
    Run the game in console mode - text-based interactive play without pygame.

    Args:
        env: The Pass the Pigs Sequential environment (without render mode)
        agent_mode: AI agent strategy ('QTable', 'Baseline', 'Roller')
        agent_threshold: Threshold parameter for Baseline/Roller agents
        load_model: Path to Q-table model to load (optional)
    """
    from env import ROLL, PASS, HOG_CALL_1, HOG_CALL_2, GOAL

    print("\n" + "=" * 60)
    print("CONSOLE MODE - SEQUENTIAL VARIANT")
    print("=" * 60)
    print("\n🎲 In this variant, you can see what your opponent did!")
    print("\nControls:")
    print("  r : Roll the dice")
    print("  p : Pass (bank your turn points)")
    if WITH_HOG_CALLS:
        print("  1 : Hog Call 1 (bet opponent will roll exactly 5 points)")
        print("  2 : Hog Call 2 (bet opponent will roll exactly 10 points)")
    print("  h : Help (view full game rules)")
    print("  q : Quit game")
    print("\nFirst to 100 points wins!")
    print("=" * 60 + "\n")

    # Set up players
    env.agents[0] = None  # Human player (console input)
    env.agents[1] = PassThePigsAgent_Sequential(mode=agent_mode, threshold=agent_threshold)

    # Load Q-table model if specified
    if load_model and agent_mode == 'QTable':
        load_QT_model(env.agents[1], load_model)
        print(f"✅ Loaded Q-learning model: {load_model}\n")

    print(f"AI Opponent: {agent_mode}" + (f" (threshold={agent_threshold})" if agent_mode != 'QTable' else ""))
    print("=" * 60)

    game_count = 0
    wins = [0, 0]

    try:
        while True:
            print(f"\n{'='*60}")
            print(f"GAME {game_count + 1}")
            print(f"{'='*60}")
            winner = play_console_game(env)
            wins[winner] += 1
            game_count += 1

            print(f"\n{'='*60}")
            print(f"GAME OVER - Player {winner + 1} WINS!")
            print(f"{'='*60}")
            print(f"Current Record:")
            print(f"  Player 1 (You): {wins[0]} wins ({(wins[0]/game_count*100):.1f}%)")
            print(f"  Player 2 (AI):  {wins[1]} wins ({(wins[1]/game_count*100):.1f}%)")

            play_again = input("\nPlay another game? (y/n): ").strip().lower()
            if play_again != 'y':
                break

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    finally:
        print(f"\n{'='*60}")
        print(f"FINAL RECORD")
        print(f"{'='*60}")
        print(f"Games played: {game_count}")
        if game_count > 0:
            print(f"  Player 1 (You): {wins[0]} wins ({(wins[0]/game_count*100):.1f}%)")
            print(f"  Player 2 (AI):  {wins[1]} wins ({(wins[1]/game_count*100):.1f}%)")
        print("Thank you for playing!")


def display_game_state(env, is_human=True):
    """
    Display the current game state (scores, turn info, warnings).
    SEQUENTIAL VARIANT: Shows opponent's last action!

    Args:
        env: The Pass the Pigs Sequential environment
        is_human: Whether the current player is human (for display purposes)
    """
    current_player = env.player
    action_names = ['ROLL', 'PASS', 'HOG CALL 1', 'HOG CALL 2']

    print(f"\n-[TURN PLAYER]{'-'*46}")
    print(f"Player {current_player + 1} {'(YOU)' if is_human else '(AI)'}")
    print(f"-[SCORES]{'-'*51}")
    print(f"[Player 1] | Bank: {env.players[0][OWN_SCORE]:3d} | Turn: {env.players[0][TURN_SCORE]:3d}")
    print(f"[Player 2] | Bank: {env.players[1][OWN_SCORE]:3d} | Turn: {env.players[1][TURN_SCORE]:3d}")

    # SEQUENTIAL INFO: Show opponent's last action
    if env.players[current_player][OPP_LAST_ACTION] >= 0:
        opp_action_idx = int(env.players[current_player][OPP_LAST_ACTION])
        opp_points = int(env.players[current_player][OPP_LAST_POINTS])
        if opp_action_idx < len(action_names):
            print(f"-[OPPONENT INFO]{'-'*44}")
            print(f"👀 Opponent's last move: {action_names[opp_action_idx]} ({opp_points:+d} points)")

    if WITH_HOG_CALLS and env.players[current_player][HOG_CALL] > 0:
        print(f"-[WARNING]{'-'*50}")
        print(f"⚠️  Hog Call active! Opponent bet on you rolling {5 if env.players[current_player][HOG_CALL]==1 else 10} points!")
    print(f"{'-'*60}")


def play_console_game(env):
    """
    Play a single game in console mode.

    Args:
        env: The Pass the Pigs Sequential environment

    Returns:
        winner: Index of the winning player (0 or 1)
    """
    from env import ROLL, PASS, HOG_CALL_1, HOG_CALL_2, GOAL

    obs, info = env.reset()
    done = False
    step_count = 0

    print("\nStarting new game...\n")

    while not done:
        current_player = env.player
        is_human = (env.agents[current_player] is None)

        # Display current game state
        display_game_state(env, is_human)

        # Get action
        if is_human:
            action = get_console_action(env)
            if action is None:  # User quit
                return 1 - current_player  # Other player wins
        else:
            # AI turn
            obs_current = env._get_obs()
            action = env.agents[current_player].predict(obs_current, training=False)
            action_names = ['ROLL', 'PASS', 'HOG CALL 1', 'HOG CALL 2']
            print(f"\n🤖 AI chooses: {action_names[action]}")
            time.sleep(1)

        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        # Show what happened
        if action == ROLL and env.throw:
            pig1, pig2 = env.throw[0]
            points = env.throw[2]
            print(f"\n🎲 ROLL RESULT: {pig1} + {pig2}")
            if points > 0:
                print(f"✅ Scored {points} points!")
            else:
                if pig1 == 'Oinker':
                    print(f"💀 OINKER! Lost all banked points!")
                elif pig1 == 'Piggyback':
                    print(f"💀💀 PIGGYBACK! Game over!")
                else:
                    print(f"❌ PIG OUT! Lost turn (no points)")
        elif action in [PASS, HOG_CALL_1, HOG_CALL_2]:
            action_msgs = {
                PASS: f"✋ PASSED! Banked {env.old_obs[current_player][TURN_SCORE]} points.",
                HOG_CALL_1: f"🎯 HOG CALL 1! Betting opponent rolls 5 points. Banked {env.old_obs[current_player][TURN_SCORE]} points.",
                HOG_CALL_2: f"🎯 HOG CALL 2! Betting opponent rolls 10 points. Banked {env.old_obs[current_player][TURN_SCORE]} points."
            }
            print(f"\n{action_msgs.get(action, '')}")

        # Check for game end
        if done:
            break

        # Small delay for readability
        if not is_human:
            time.sleep(0.5)

    # Game finished
    print(f"\n{'='*60}")
    print(f"🏆 WINNER: Player {env.winner + 1} {'(YOU)' if env.winner == 0 else '(AI)'}!")
    print(f"{'='*60}")
    print(f"Final Scores:")
    print(f"  Player 1: {env.players[0][OWN_SCORE]} points")
    print(f"  Player 2: {env.players[1][OWN_SCORE]} points")
    print(f"Total turns: {step_count}")

    return env.winner


def display_game_rules():
    """
    Display the game rules from the rules file.
    """
    try:
        with open('game_rules.txt', 'r', encoding='utf-8') as f:
            rules = f.read()
        print("\n" + rules)
        print("\n" + "="*60)
        print("SEQUENTIAL VARIANT SPECIAL RULE:")
        print("="*60)
        print("In this variant, you can see your opponent's last action")
        print("and how many points they scored. Use this information")
        print("strategically to adapt your gameplay!")
        print("="*60)
        input("\nPress ENTER to continue...")
    except FileNotFoundError:
        print("\n❌ Rules file not found (game_rules.txt)")
        print("Basic rules: Roll to score points, Pass to bank them, reach 100 to win!")
        print("\nSEQUENTIAL VARIANT: You can see opponent's last move!")
        input("\nPress ENTER to continue...")


def get_console_action(env):
    """
    Get action from user via console input.

    Args:
        env: The environment

    Returns:
        action: The chosen action (ROLL, PASS, HOG_CALL_1, HOG_CALL_2) or None to quit
    """
    from env import ROLL, PASS, HOG_CALL_1, HOG_CALL_2, HOG_CALL_SCORE_1, HOG_CALL_SCORE_2

    while True:
        valid_inputs = "r (roll), p (pass)"
        if WITH_HOG_CALLS:
            valid_inputs += f", 1 (hog call {HOG_CALL_SCORE_1} points), 2 (hog call {HOG_CALL_SCORE_2} points)"
        valid_inputs += ", h (help), q (quit)"

        choice = input(f"\nYour choice [{valid_inputs}]: ").strip().lower()

        if choice == 'r':
            return ROLL
        elif choice == 'p':
            return PASS
        elif choice == '1' and WITH_HOG_CALLS:
            return HOG_CALL_1
        elif choice == '2' and WITH_HOG_CALLS:
            return HOG_CALL_2
        elif choice == 'h':
            display_game_rules()
            # Re-display game state after viewing rules
            display_game_state(env, is_human=True)
        elif choice == 'q':
            confirm = input("Are you sure you want to quit? (y/n): ").strip().lower()
            if confirm == 'y':
                return None
        else:
            print("❌ Invalid input. Please try again.")


def main():
    """Main entry point for the Sequential Pass the Pigs runner."""
    parser = argparse.ArgumentParser(
        description='Pass the Pigs - SEQUENTIAL VARIANT - Reinforcement Learning Game',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SEQUENTIAL VARIANT: Agents observe opponent's last action and points scored!

Examples:
  python main.py --mode console
      Play in text-based console mode (see opponent's moves!)

  python main.py --mode human --games 10
      Watch 10 games with visual rendering

  python main.py --mode train --epochs 50
      Train a Sequential Q-learning agent for 50 epochs

  python main.py --mode eval --games 100
      Evaluate agents over 100 games without rendering
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='console',
        choices=['console', 'human', 'train', 'eval'],
        help='Game mode: console (text-based play), '
             'human (watch with rendering), train (train Q-learning), '
             'eval (no rendering)'
    )

    parser.add_argument(
        '--games',
        type=int,
        default=10,
        help='Number of games to play (for human/eval modes)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (for train mode)'
    )

    parser.add_argument(
        '--games-per-epoch',
        type=int,
        default=50,
        help='Games per epoch during training'
    )

    parser.add_argument(
        '--agent1',
        type=str,
        default='QTable',
        choices=['QTable', 'Baseline', 'Roller'],
        help='Strategy for Agent 1'
    )

    parser.add_argument(
        '--agent2',
        type=str,
        default='Baseline',
        choices=['QTable', 'Baseline', 'Roller'],
        help='Strategy for Agent 2'
    )

    parser.add_argument(
        '--threshold1',
        type=int,
        default=25,
        help='Threshold parameter for Agent 1 (Baseline/Roller modes)'
    )

    parser.add_argument(
        '--threshold2',
        type=int,
        default=20,
        help='Threshold parameter for Agent 2 (Baseline/Roller modes)'
    )

    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        help='Load a saved Q-table model (path without extension)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed game information'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 60)
    print("PASS THE PIGS - SEQUENTIAL VARIANT")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Variant: SEQUENTIAL (observe opponent actions)")
    print(f"Hog Calls: {'Enabled' if WITH_HOG_CALLS else 'Disabled'}")

    # Create environment based on mode
    if args.mode == 'console':
        env = PassThePigs_2Players_Sequential_Env(render_mode=None)
        run_console_mode(env, agent_mode=args.agent2, agent_threshold=args.threshold2, load_model=args.load_model)

    elif args.mode == 'human':
        env = PassThePigs_2Players_Sequential_Env(render_mode='human')

        # Set up agents
        env.agents[0] = PassThePigsAgent_Sequential(mode=args.agent1, threshold=args.threshold1)
        env.agents[1] = PassThePigsAgent_Sequential(mode=args.agent2, threshold=args.threshold2)

        # Load model if specified
        if args.load_model and env.agents[0].mode == 'QTable':
            load_QT_model(env.agents[0], args.load_model)
            print(f"Loaded model: {args.load_model}")

        # Play games with rendering
        play_multiple_games(env, num_games=args.games, render=True, training=False, verbose=args.verbose)

    elif args.mode == 'train':
        env = PassThePigs_2Players_Sequential_Env(render_mode=None)
        run_training_session(env, num_epochs=args.epochs, games_per_epoch=args.games_per_epoch)

    elif args.mode == 'eval':
        env = PassThePigs_2Players_Sequential_Env(render_mode=None)

        # Set up agents
        env.agents[0] = PassThePigsAgent_Sequential(mode=args.agent1, threshold=args.threshold1)
        env.agents[1] = PassThePigsAgent_Sequential(mode=args.agent2, threshold=args.threshold2)

        # Load model if specified
        if args.load_model and env.agents[0].mode == 'QTable':
            load_QT_model(env.agents[0], args.load_model)
            print(f"Loaded model: {args.load_model}")

        # Play games without rendering
        play_multiple_games(env, num_games=args.games, render=False, training=False, verbose=args.verbose)

    # Cleanup
    env.close()
    if TRAINING and hasattr(writer, 'close'):
        writer.flush()
        writer.close()

    print("\nThank you for playing Pass the Pigs - Sequential Variant!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

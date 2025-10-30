"""
Pass the Pigs - Main Runner (CLI)

This script provides a user-friendly CLI so a human can play Pass the Pigs
against the trained Q-learning models (or other baseline agents).

Quick examples:
  - Play text-based (no pygame) vs Q-table model named QT_0 stored in output/:
      python main.py --mode console --agent2 QTable --load-model QT_0

  - Play with pygame keyboard controls vs Baseline:
      python main.py --mode interactive --agent2 Baseline --threshold2 20

Notes on model loading:
  --load-model takes the model ID without extension (e.g., QT_0). Files are
  typically in output/QT_0.json and output/QT_0.npy. This script includes a
  robust loader that tries both output/<id>.* and <id>.*.
"""

import argparse
import os
import time
import numpy as np

from env import (
    PassThePigs_2Players_Env,
    PassThePigsAgent,
    RENDER_DELAY,
    WITH_HOG_CALLS,
    OWN_SCORE,
    TURN_SCORE,
    save_QT_model,
    writer,
    model_name,
)


# -----------------------------
# Helpers
# -----------------------------
def load_qtable_into_agent(agent: PassThePigsAgent, model_id: str) -> bool:
    """Load a saved Q-table into the given agent.

    Tries the following locations (without extensions):
      - output/<model_id>
      - <model_id>

    Sets agent.learning_rate, agent.epsilon, agent.episode, agent.QTable.
    Returns True on success, False if files are missing.
    """
    base_candidates = []
    mid = os.path.splitext(model_id)[0]
    base_candidates.append(os.path.join('output', mid))
    # Also try raw name in case files are at repo root
    base_candidates.append(mid)

    for base in base_candidates:
        cfg_json = base + '.json'
        arr_npy = base + '.npy'
        # Preferred: load config + array
        if os.path.exists(cfg_json) and os.path.exists(arr_npy):
            try:
                import json
                with open(cfg_json, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)

                num_actions = cfg.get('num_actions')
                # env exposes ACTIONS length implicitly via agent.QTable shape; basic sanity check only
                agent.learning_rate = cfg.get('learning_rate', getattr(agent, 'learning_rate', 0.1))
                agent.epsilon = cfg.get('epsilon', getattr(agent, 'epsilon', 0.1))
                agent.episode = cfg.get('episode', getattr(agent, 'episode', 0))

                agent.QTable = np.load(arr_npy)

                # If num_actions provided, it should match last dim of QTable
                if isinstance(num_actions, int) and agent.QTable.shape[-1] != num_actions:
                    print(f"Warning: model '{model_id}' action count {num_actions} "
                          f"!= QTable last dim {agent.QTable.shape[-1]}")
                return True
            except Exception as e:
                print(f"Failed to load model from '{base}': {e}")
                return False
        # Fallback: only .npy exists — load array and use agent defaults for hyperparams
        if not os.path.exists(cfg_json) and os.path.exists(arr_npy):
            try:
                agent.QTable = np.load(arr_npy)
                print(f"Loaded QTable weights from '{arr_npy}' (no config JSON found). Using default agent params for play.")
                return True
            except Exception as e:
                print(f"Failed to load QTable array from '{arr_npy}': {e}")
                return False
    print(f"Model files not found for id '{model_id}'. Expected output/{mid}.json and output/{mid}.npy (or at repo root).")
    return False


def display_game_state(env, is_human=True):
    from env import HOG_CALL
    print(f"\n-[TURN PLAYER]{'-'*46}")
    print(f"Player {env.player + 1} {'(YOU)' if is_human else '(AI)'}")
    print(f"-[SCORES]{'-'*51}")
    print(f"[Player 1] | Bank: {env.players[0][OWN_SCORE]:3d} | Turn: {env.players[0][TURN_SCORE]:3d}")
    print(f"[Player 2] | Bank: {env.players[1][OWN_SCORE]:3d} | Turn: {env.players[1][TURN_SCORE]:3d}")
    if WITH_HOG_CALLS and env.players[env.player][HOG_CALL] > 0:
        print("WARNING: Hog Call active against you this turn!")
    print(f"{'-'*60}")


def display_game_rules():
    try:
        with open('game_rules.txt', 'r', encoding='utf-8') as f:
            rules = f.read()
        print("\n" + rules)
        input("\nPress ENTER to continue...")
    except FileNotFoundError:
        print("\nBasic rules: Roll to score points, Pass to bank them; first to 100 wins.")
        input("\nPress ENTER to continue...")


def get_console_action(env):
    from env import ROLL, PASS, HOG_CALL_1, HOG_CALL_2, HOG_CALL_SCORE_1, HOG_CALL_SCORE_2
    while True:
        valid = "r (roll), p (pass)"
        if WITH_HOG_CALLS:
            valid += f", 1 (hog call {HOG_CALL_SCORE_1}), 2 (hog call {HOG_CALL_SCORE_2})"
        valid += ", h (help), q (quit)"

        choice = input(f"\nYour choice [{valid}]: ").strip().lower()
        if choice == 'r':
            return ROLL
        if choice == 'p':
            return PASS
        if choice == '1' and WITH_HOG_CALLS:
            return HOG_CALL_1
        if choice == '2' and WITH_HOG_CALLS:
            return HOG_CALL_2
        if choice == 'h':
            display_game_rules()
            display_game_state(env, is_human=True)
            continue
        if choice == 'q':
            if input("Are you sure you want to quit? (y/n): ").strip().lower() == 'y':
                return None
        print("Invalid input. Please try again.")


def play_console_game(env):
    from env import ROLL, PASS, HOG_CALL_1, HOG_CALL_2, OWN_SCORE
    obs, _ = env.reset()
    done = False
    step_count = 0
    print("\nStarting new game...\n")

    while not done:
        current_player = env.player
        is_human = (env.agents[current_player] is None)

        display_game_state(env, is_human)

        if is_human:
            action = get_console_action(env)
            if action is None:
                return None
        else:
            obs_current = env._get_obs()
            action = env.agents[current_player].predict(obs_current, training=False)
            print(["ROLL", "PASS", "HOG CALL 1", "HOG CALL 2"][action])
            time.sleep(1)

        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        # Print roll result if action was ROLL
        if action == ROLL and env.throw:
            pig1, pig2 = env.throw[0]
            points = env.throw[2]
            print(f"\nROLL RESULT: {pig1} + {pig2}")
            if points > 0:
                print(f"Scored {points} points!")
            else:
                if pig1 == 'Oinker':
                    print("OINKER! Lost all banked points!")
                elif pig1 == 'Piggyback':
                    print("PIGGYBACK! Game over!")
                else:
                    print("PIG OUT! Lost turn (no points)")
        elif action in [PASS, HOG_CALL_1, HOG_CALL_2]:
            turn_pts = env.old_obs[current_player][TURN_SCORE] if hasattr(env, 'old_obs') else 0
            action_msgs = {
                PASS: f"PASSED! Banked {turn_pts} points.",
                HOG_CALL_1: f"HOG CALL 1! Betting opponent rolls 5 points. Banked {turn_pts} points.",
                HOG_CALL_2: f"HOG CALL 2! Betting opponent rolls 10 points. Banked {turn_pts} points."
            }
            print(f"\n{action_msgs.get(action, '')}")

        if done:
            break
        if not is_human:
            time.sleep(0.5)

    if action is not None:
        print(f"\n{'='*60}")
        print(f"WINNER: Player {env.winner + 1} {'(YOU)' if env.winner == 0 else '(AI)'}!")
        print(f"{'='*60}")
        print("Final Scores:")
        print(f"  Player 1: {env.players[0][OWN_SCORE]} points")
        print(f"  Player 2: {env.players[1][OWN_SCORE]} points")
        print(f"Total turns: {step_count}")
        return env.winner
    else:
        return None


# -----------------------------
# Modes
# -----------------------------
def run_console_mode(env, agent_mode='Baseline', agent_threshold=20, load_model=None):
    print("\n" + "=" * 60)
    print("CONSOLE MODE - TEXT-BASED INTERACTIVE GAME")
    print("=" * 60)
    print("\nControls:")
    print("  r : Roll the dice")
    print("  p : Pass (bank your turn points)")
    if WITH_HOG_CALLS:
        print("  1 : Hog Call 1 (bet on 5 points)")
        print("  2 : Hog Call 2 (bet on 10 points)")
    print("  h : Help (view rules)")
    print("  q : Quit game")
    print("\nFirst to 100 points wins!")
    print("=" * 60 + "\n")

    env.agents[0] = None  # Human
    env.agents[1] = PassThePigsAgent(mode=agent_mode, threshold=agent_threshold)

    if load_model and agent_mode == 'QTable':
        if load_qtable_into_agent(env.agents[1], load_model):
            print(f"Loaded Q-learning model: {load_model}\n")
        else:
            print("Continuing without loaded model (QTable initialized randomly).\n")

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
            if winner is None:
                print("\nGame quit early. No result recorded.")
                break
            wins[winner] += 1
            game_count += 1

            print(f"\n{'='*60}")
            print(f"GAME OVER - Player {winner + 1} WINS!")
            print(f"{'='*60}")
            print("Current Record:")
            print(f"  Player 1 (You): {wins[0]} wins ({(wins[0]/game_count*100):.1f}%)")
            print(f"  Player 2 (AI):  {wins[1]} wins ({(wins[1]/game_count*100):.1f}%)")

            play_again = input("\nPlay another game? (y/n): ").strip().lower()
            if play_again != 'y':
                break

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    finally:
        print(f"\n{'='*60}")
        print("FINAL RECORD")
        print(f"{'='*60}")
        print(f"Games played: {game_count}")
        if game_count > 0:
            print(f"  Player 1 (You): {wins[0]} wins ({(wins[0]/game_count*100):.1f}%)")
            print(f"  Player 2 (AI):  {wins[1]} wins ({(wins[1]/game_count*100):.1f}%)")
        print("Thank you for playing!")


def run_interactive_mode(env):
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE or R : Roll the dice")
    print("  P          : Pass (bank your points)")
    if WITH_HOG_CALLS:
        print("  1          : Hog Call 1 (bet on 5 points)")
        print("  2          : Hog Call 2 (bet on 10 points)")
    print("  Arrow Keys : Move cursor (demo)")
    print("  Close window to quit")
    print("-" * 60 + "\n")

    env.agents[0] = None  # Human keyboard
    env.agents[1] = PassThePigsAgent(mode='Baseline', threshold=20)

    game_count = 0
    wins = [0, 0]

    try:
        while True:
            print(f"\n--- Starting Game {game_count + 1} ---")
            from_main_loop_winner = play_single_game(env, render=True, training=False, verbose=True)
            winner = from_main_loop_winner[0] if isinstance(from_main_loop_winner, tuple) else from_main_loop_winner
            wins[winner] += 1
            game_count += 1

            print(f"\nCurrent record: Player 1: {wins[0]} wins, Player 2: {wins[1]} wins")
            print("Starting next game in 3 seconds...")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nGame ended: {e}")
    finally:
        print(f"\nFinal record: {game_count} games played")
        if game_count:
            print(f"  Player 1 wins: {wins[0]} ({(wins[0]/game_count*100):.1f}%)")
            print(f"  Player 2 wins: {wins[1]} ({(wins[1]/game_count*100):.1f}%)")


def play_single_game(env, render=True, training=False, verbose=True):
    obs, info = env.reset()
    done = False
    step_count = 0

    if render and env.render_mode:
        time.sleep(RENDER_DELAY)
        env.render()

    while not done:
        if env.render_mode and env.render_mode in ['human', 'interactive']:
            import pygame as pg
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    raise KeyboardInterrupt("Window closed by user")

        action = env.get_action(training)
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        if render and env.render_mode:
            time.sleep(max(RENDER_DELAY, 0.5))
            env.render()

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

    return env.winner, {
        'winner': env.winner,
        'steps': step_count,
        'final_scores': [env.players[0][OWN_SCORE], env.players[1][OWN_SCORE]],
        'reason': info.get('reason', 'Unknown')
    }


# -----------------------------
# Entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Pass the Pigs - CLI to play vs trained agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --mode console --agent2 QTable --load-model QT_0\n"
            "  python main.py --mode interactive --agent2 Baseline --threshold2 20\n"
        ),
    )

    parser.add_argument('--mode', type=str, default='console', choices=['console', 'interactive', 'human', 'eval'],
                        help='Game mode: console (text), interactive (pygame keyboard), human (watch with render), eval (no render)')
    parser.add_argument('--games', type=int, default=10, help='Number of games (human/eval modes)')
    parser.add_argument('--agent1', type=str, default='QTable', choices=['QTable', 'Baseline', 'Roller'], help='Agent 1 strategy')
    parser.add_argument('--agent2', type=str, default='Baseline', choices=['QTable', 'Baseline', 'Roller'], help='Agent 2 strategy')
    parser.add_argument('--threshold1', type=int, default=25, help='Threshold for Agent 1 (Baseline/Roller)')
    parser.add_argument('--threshold2', type=int, default=20, help='Threshold for Agent 2 (Baseline/Roller)')
    parser.add_argument('--load-model', type=str, default=None, help='QTable model ID (e.g., QT_0) without extension')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PASS THE PIGS - PLAY VS TRAINED AGENTS")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Hog Calls: {'Enabled' if WITH_HOG_CALLS else 'Disabled'}")

    if args.mode == 'console':
        env = PassThePigs_2Players_Env(render_mode=None)
        run_console_mode(env, agent_mode=args.agent2, agent_threshold=args.threshold2, load_model=args.load_model)
        env.close()
        if 'writer' in globals() and hasattr(writer, 'close'):
            writer.flush(); writer.close()
        return

    if args.mode == 'interactive':
        env = PassThePigs_2Players_Env(render_mode='interactive')
        run_interactive_mode(env)
        env.close()
        if 'writer' in globals() and hasattr(writer, 'close'):
            writer.flush(); writer.close()
        return

    # Human and eval: AI vs AI (or watch) helpers
    from env import load_QT_model
    env = PassThePigs_2Players_Env(render_mode=('human' if args.mode == 'human' else None))
    env.agents[0] = PassThePigsAgent(mode=args.agent1, threshold=args.threshold1)
    env.agents[1] = PassThePigsAgent(mode=args.agent2, threshold=args.threshold2)

    # Load model for agent 1 if needed
    if args.load_model and env.agents[0].mode == 'QTable':
        if not load_qtable_into_agent(env.agents[0], args.load_model):
            print(f"Warning: failed to load model '{args.load_model}' for Agent 1; using random QTable.")

    # Play multiple games
    num_games = max(1, int(args.games))
    wins = [0, 0]
    for g in range(num_games):
        w, _info = play_single_game(env, render=(args.mode == 'human'), training=False, verbose=args.verbose)
        wins[w] += 1
        print(f"Game {g + 1}/{num_games} complete. Score: P1={wins[0]}, P2={wins[1]}")

    print("\n" + "=" * 50)
    print("FINAL STATISTICS")
    print("=" * 50)
    print(f"  Player 1 wins: {wins[0]} ({(wins[0]/num_games*100):.1f}%)")
    print(f"  Player 2 wins: {wins[1]} ({(wins[1]/num_games*100):.1f}%)")

    env.close()
    if 'writer' in globals() and hasattr(writer, 'close'):
        writer.flush(); writer.close()


if __name__ == "__main__":
    main()

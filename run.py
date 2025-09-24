from env import PassThePigs_2Players_Env, PassThePigsAgent, play_games

if __name__ == "__main__":
    env = PassThePigs_2Players_Env(render_mode="interactive")

    env.agents[0] = PassThePigsAgent(mode="QTable")
    env.agents[1] = PassThePigsAgent(mode="Baseline", threshold=20)

    play_games(max_games=10, training=False)

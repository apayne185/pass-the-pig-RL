import numpy as np
from env import PassThePigs_2Players_Env, PassThePigsAgent, play_games

# data = np.load("output/my_qtable_2Players_c3_QT_0.npy")
# print(data.shape, np.mean(data), np.max(data))

#inspect the qtables, ensure they are evolving/not empty  -- we had small error where no results showed on tensorboard
q1 = np.load("output/my_qtable_2Players_c1_QT_0.npy")
q2 = np.load("output/my_qtable_2Players_c2_QT_0.npy")  
q3 = np.load("output/my_qtable_2Players_c3_QT_0.npy")  

print(q1.shape,q2.shape,q3.shape)
print("Sample Q-values:",q3[0:5, 0:5])







#TRAIN RUNS - if wanted

# load best QTable, test it in new env
# compare to a baseline omdel
# env = PassThePigs_2Players_Env(render_mode="human")
# agent = PassThePigsAgent(mode="QTable")
# agent.QTable = np.load("output/my_qtable_2Players_c3_QT_0.npy")

# env.agents[0]= agent
# env.agents[1]= PassThePigsAgent(mode="Baseline", threshold=100)

# num_games, num_games1 = play_games(env, 100, training=False)
# print(f"Agent win rate: {100 * (num_games - num_games1) / num_games:.2f}%")


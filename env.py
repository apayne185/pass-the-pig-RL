#Pass the Pigs 2 Players Env.py  USED TO BE MAIN.PY

"""
Pass the Pigs - 2 Players Gymnasium Environment
Each player can implement a trainable agent/model
TO DO:
* For compatibilty with OpenAI gym, only Agent[0] will be a trainable agent
* Adapt to an AEC scheme for future multiagent RL environment compatibility
"""


import glob, os, random, time, copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
# from typing import Any, Dict, Optional, Tuple, Union
from typing import Optional, Dict
import json
# from datetime import datetime
# date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}' # returns '2024-01-15 23:58:18'
import torch
from torch.utils.tensorboard import SummaryWriter
import pygame as pg
from collections import defaultdict



# In[]: GAME ENVIRONMENT SETUP
# +-------------------------------------+
# |        GAME ENVIRONMENT SETUP       |
# +-------------------------------------+
NUM_PLAYERS = 2 # WARNING!!! ONLY PREPARED FOR 2 PLAYERS
RULES_B = True  # game ends as soon as any player reaches GOAL (100 pts)
WITH_HOG_CALLS = True # False
RENDER_DELAY = 0.01  #secs per render updates
AVG_EP = 1_000


if WITH_HOG_CALLS:
    ACTIONS  = ['ROLL','PASS','HOG_CALL_1','HOG_CALL_2']
else:
    ACTIONS  = ['ROLL','PASS']

NONE     = -1
ROLL     = 0
PASS     = 1
HOG_CALL_1 = 2
HOG_CALL_2 = 3

HOG_CALL_SCORE_1 = 5       # score the hog_call is betting on, one of the more common rolls
HOG_CALL_SCORE_2 = 10

OUTS      = ['PIG_OUT','PIGGYBACK','OINKER']
PIGGYBACK = -2
OINKER    = -1
PIG_OUT   = 0

# State representation (features)
OWN_SCORE  = 0
OPP_SCORE  = 1
TURN_SCORE = 2
HOG_CALL   = 3

GOAL = 100 # Points
BUCKET = 5 # Points (for grouping scores into buckets)
# MAX_BUCKETS = GOAL//BUCKET
MAX_BUCKETS = 10    #if we have hog calls, table becomes 5d and 20^5 = 3.2 million entries

# In[]: SYSTEM DYNAMICS
#----------------------------------------------------------
# SYSTEM DYNAMICS
# Throw, probability, score
THROWS = [[('Oinker', 'Oinker'), 0.0386, 0], [('Piggyback', 'Piggyback'), 0.001, 0], [('Side Pink', 'Side Pink'), 0.0961, 1], [('Side Pink', 'Side Dot'), 0.0961, 0], [('Side Pink', 'Razorback'), 0.0465, 5], [('Side Pink', 'Trotter'), 0.031, 5], [('Side Pink', 'Snouter'), 0.0217, 10], [('Side Pink', 'Leaning Jowler'), 0.0124, 15], [('Side Dot', 'Side Pink'), 0.0961, 0], [('Side Dot', 'Side Dot'), 0.0961, 1], [('Side Dot', 'Razorback'), 0.0465, 5], [('Side Dot', 'Trotter'), 0.031, 5], [('Side Dot', 'Snouter'), 0.0217, 10], [('Side Dot', 'Leaning Jowler'), 0.0124, 15], [('Razorback', 'Side Pink'), 0.0465, 5], [('Razorback', 'Side Dot'), 0.0465, 5], [('Razorback', 'Razorback'), 0.0225, 10], [('Razorback', 'Trotter'), 0.015, 10], [('Razorback', 'Snouter'), 0.0105, 15], [('Razorback', 'Leaning Jowler'), 0.006, 20], [('Trotter', 'Side Pink'), 0.031, 5], [('Trotter', 'Side Dot'), 0.031, 5], [('Trotter', 'Razorback'), 0.015, 10], [('Trotter', 'Trotter'), 0.01, 10], [('Trotter', 'Snouter'), 0.007, 15], [('Trotter', 'Leaning Jowler'), 0.004, 20], [('Snouter', 'Side Pink'), 0.0217, 10], [('Snouter', 'Side Dot'), 0.0217, 10], [('Snouter', 'Razorback'), 0.0105, 15], [('Snouter', 'Trotter'), 0.007, 15], [('Snouter', 'Snouter'), 0.0049, 20], [('Snouter', 'Leaning Jowler'), 0.0028, 25], [('Leaning Jowler', 'Side Pink'), 0.0124, 15], [('Leaning Jowler', 'Side Dot'), 0.0124, 15], [('Leaning Jowler', 'Razorback'), 0.006, 20], [('Leaning Jowler', 'Trotter'), 0.004, 20], [('Leaning Jowler', 'Snouter'), 0.0028, 25], [('Leaning Jowler', 'Leaning Jowler'), 0.0016, 30]]
#----------------------------------------------------------
"""
# abridged system dynamics
outcomes = [PIGGYBACK, OINKER, PIG_OUT, 1, 5, 10, 15, 20, 25, 40, 60]
out_prob = [0.001, 0.0386, 0.1922, 0.1922, 0.31, 0.1168, 0.0846, 0.0525, 0.0056, 0.0049, 0.0016]
"""
# just for debugging purposes
# for k in range(n := len(THROWS)): THROWS[k][1] = 1/n # equi-probable...
# print(THROWS)





# In[]: GAME INTERFACE
# +-------------------------------------+
# |            GAME INTERFACE           |
# +-------------------------------------+
IMG_W  = 728//2
IMG_H  = 410
H_INFO = 120
FPS    = 20
VERBOSE = False
RENDER_MODE = 'None'       # None # 'human' 'interactive'
if RENDER_MODE == 'interactive':
    assert WITH_HOG_CALLS, "Interactive Mode is setup for Hog Calls options"





# +-------------------------------------+
# |   pygame - Utility Functions        |
# +-------------------------------------+
def draw_text(screen,font,text,color,pos,center=False,antialias=False):
      text_surf = font.render(text, antialias, color) # text, antialias, color
      if center:
        text_rect = text_surf.get_rect(center=pos)
        screen.blit(text_surf, text_rect)
      else:
        screen.blit(text_surf, dest=pos)





# In[]: TRAINING & AGENTS
# +-------------------------------------+
# |      Tensorboard Logging            |
# +-------------------------------------+


TRAINING = True # True
if TRAINING:
    # SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
    writer = SummaryWriter()
    # Writer will output to ./runs/ directory by default.
    # Default is runs/CURRENT_DATETIME_HOSTNAME
    # To log a scalar value, use add_scalar(tag, scalar_value, global_step=None, walltime=None).

    # for epoch in range(10_000):
    #     loss = epoch % 1_000
    #     writer.add_scalar("Loss/train", loss, epoch)
    # writer.close()

    # AVG_EP = 1_000 # number of episodes for averaged rewards reporting/logging






# +-------------------------------------+
# |         Agent Functions             |
# +-------------------------------------+
STAGE = 0
model_name  = f'QT_{STAGE}'

# Training parameters
learning_rate = 0.8 # 0.8 # 0.1
min_LR = 0.1
decay_rate_LR = 1e-6
gamma = 0.95 # 0.99

# Exploration parameters (for first training from scratch)
max_epsilon = 1.0
min_epsilon = 0.05  # min_epsilon = 0.01
decay_rate = 2e-6   # decay_rate = 0.01


def epsilon_greedy_policy(Qtable, state, epsilon): # greedy policy when epsilon=0
    random_int = random.uniform(0,1)
    if random_int > epsilon:
        action = np.argmax(Qtable[state])
    else:
        action = env.action_space.sample()
    return action


# Rate at which we set the learning rate. Typically, keep this constant
def learning_schedule(episode):
    return min_LR + (learning_rate - min_LR)*np.exp(-decay_rate_LR*episode)

#same goes for the episilon, we decrease
def epsilon_schedule(episode):
    return min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)




def save_QT_model(agent,model_name):
    # Write config data to a file
    config = {"num_actions": len(ACTIONS),
              "learning_rate": agent.learning_rate,
              "epsilon": agent.epsilon,
              "episode": agent.episode,
             }
    with open('output/'+model_name+'.json', 'w') as f:
        json.dump(config, f, indent=4)
    np.save('output/'+model_name+'.npy', agent.QTable)


def load_QT_model(agent,model_name):
    # Read config data from a file
    with open('output/'+model_name+'.json', 'r') as f:
        config = json.load(f)
    num_actions         = config['num_actions']
    if num_actions != len(ACTIONS):
        raise ValueError(f'Invalid action space {num_actions}|{len(ACTIONS)}')
    agent.learning_rate = config['learning_rate']
    agent.epsilon       = config['epsilon']
    agent.episode       = config['episode']
    agent.QTable = np.load(model_name+'.npy')






# modes: Roller (prob=p%), Baseline, QTable
class PassThePigsAgent():
    def __init__(self,mode,threshold=25):    #used to be mode='Baseline'
        self.mode = mode
        if mode == 'QTable':
            # 20 buckets of scores from 0..GOAL points
            # Addressable as QTable[tuple(state)][action]
            # self.QTable = np.zeros((MAX_BUCKETS,MAX_BUCKETS,MAX_BUCKETS,len(ACTIONS)-1,len(ACTIONS)),dtype=float)
            if WITH_HOG_CALLS: # len(ACTIONS) > 2: # option for Hog Calls
                self.QTable = np.random.rand(MAX_BUCKETS,MAX_BUCKETS,MAX_BUCKETS,len(ACTIONS)-1,len(ACTIONS))*0.001
            else:
                self.QTable = np.random.rand(MAX_BUCKETS,MAX_BUCKETS,MAX_BUCKETS,len(ACTIONS))*0.001
            self.epsilon = max_epsilon
            self.learning_rate = learning_rate
            # for statistical purposes
            self.episode = 0
            self.episode_reward = 0
            self.episode_rewards = np.zeros(AVG_EP) # last 1000 rewards
        else:
            self.threshold = threshold # optional parameter, different purposes

    def predict(self,obs,training=False):
        if self.mode == 'QTable':
            if WITH_HOG_CALLS:
                obs   = np.concatenate([np.minimum(obs[:-1]//BUCKET,MAX_BUCKETS-1),obs[-1:]]) # bucketing
            else:
                obs   = np.minimum(obs//BUCKET,MAX_BUCKETS-1) # bucketing
            state = tuple(obs)
            # print(state)
            action = epsilon_greedy_policy(self.QTable, state, self.epsilon if training else 0) # greedy policy when epsilon=0

        elif self.mode == 'Baseline':
            # basic heuristic rule...
            action = ROLL # roll
            if obs[TURN_SCORE] > self.threshold and ((obs[OWN_SCORE] + obs[TURN_SCORE]) > obs[OPP_SCORE]):
                action = PASS # pass
        else: # Roller
            prob_roll = self.threshold/100
            action = np.random.choice([ROLL,PASS], 1, p=[prob_roll,(1-prob_roll)])[0]
        return action


    def learn(self,old_obs,action,new_obs,reward,terminated,truncated,info,idx):
        if self.mode == 'QTable':
            # print('\rlearning',action,reward,' '*20,end='')
            old_obs = np.array(old_obs)
            new_obs = np.array(new_obs)

            if WITH_HOG_CALLS:
                old_obs = np.concatenate([np.minimum(old_obs[:-1]//BUCKET,MAX_BUCKETS-1),old_obs[-1:]]) # bucketing
                new_obs = np.concatenate([np.minimum(new_obs[:-1]//BUCKET,MAX_BUCKETS-1),new_obs[-1:]])

            else:
                old_obs = np.minimum(old_obs//BUCKET,MAX_BUCKETS-1) # bucketing
                new_obs = np.minimum(new_obs//BUCKET,MAX_BUCKETS-1)

            old_state=tuple(old_obs)
            new_state=tuple(new_obs)
            #--------------------------------------------------------------------------------
            qValue = self.QTable[old_state][action]
            qPred  = reward + gamma * np.max(self.QTable[new_state])*(not terminated)
            self.QTable[old_state][action] = qValue + self.learning_rate * ( qPred - qValue )
            #--------------------------------------------------------------------------------
            self.episode_reward += reward

            #this is where we save log stats to tensorboard
            if terminated:
                self.episode_rewards[self.episode % AVG_EP] = self.episode_reward
                self.episode += 1
                writer.add_scalar(f"Agent({idx})/reward", self.episode_reward, self.episode)
                writer.add_scalar(f"Agent({idx})/reward_{AVG_EP}", np.sum(self.episode_rewards)/AVG_EP, self.episode)
                writer.add_scalar(f"Agent({idx})/epsilon", self.epsilon, self.episode)
                writer.add_scalar(f"Agent({idx})/learning_rate", self.learning_rate, self.episode)
                self.epsilon = epsilon_schedule(self.episode)
                self.learning_rate = learning_schedule(self.episode)
                self.episode_reward = 0


            """
            #Q Function Update
            #(not done) keeps the terminal state as 0
            Q[state][action] += alpha * (reward + gamma * Q[nstate].max() * (not done) - Q[state][action])
            state = nstate

            if done:
                rewards.append(total_reward) #Keep track of the total rewards per episode
            """
        else:
            # do nothing...
            pass

'''
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
  for episode in trange(n_training_episodes):

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    # repeat
    for step in range(max_steps):
      action = epsilon_greedy_policy(Qtable, state, epsilon)
      new_state, reward, done, info = env.step(action)
      Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])
      # If done, finish the episode
      if done: break
      # Our state is the new state
      state = new_state
  return Qtable
'''






# In[]: ENVIRONMENT
# +-------------------------------------+
# |      Environment Functions          |
# +-------------------------------------+
## Credits: José Manuel Rey, 2020
class PassThePigs_2Players_Env(gym.Env):
    metadata = {
        "render_modes": [
            "interactive", # pygame interactive game (demo)
            "human",       # pygame interface
            "rgb_array",   # not implemented
        ],
        "modes": [
            "rules_A",   # not implemented
            "rules_B",   # default
        ],
        "FPS": 20,  # pygame
    }

    def __init__(self, render_mode: Optional[str] = None, simultaneous_mode: bool=False):     #automatically not cournot mode
        super(PassThePigs_2Players_Env, self).__init__()
        self.render_mode = render_mode
        self.last_game_info = ""
        self.simultaneous_mode = simultaneous_mode    #COURNOT MODE = TRUE

        self.players = []
        self.agents  = [None, None] # for 2 players
        self.winner  = -1

        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Discrete(len(ACTIONS))
        # 0 - Roll
        # 1 - Pass & No Hog Call
        # 2 - Pass & Hog Call (prediction = 5 points)
        # 3 - Pass & Hog Call (prediction = 10 points)

        # own score, other players'score, turn score, type_of_hog_call_placed
        """
        spaces = {
                'scores': gym.spaces.Box(low=0, high=100, shape=(N_PLAYERS+1,)),
                'hog_call': gym.spaces.Box(low=0, high=1, shape=(1,)),
                 }
        self.observation_space = gym.spaces.Dict(spaces)

        self.observation_space = spaces.Box(low=0, high=100, shape=(N_PLAYERS+1+1,), dtype=int)
        """

        # own score, opponent score, turn score, type_of_hog_call_placed
        if WITH_HOG_CALLS: # len(ACTIONS) > 2: # option for Hog Calls
            low  = np.array([0,0,0,0])
            high = np.array([GOAL,GOAL,GOAL,len(ACTIONS)-2]) # hog_call: 0 (None|Pass), 1 (type 1), 2 (type 2)

        else:
            low  = np.array([0,0,0])
            high = np.array([GOAL,GOAL,GOAL])


        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=int)
        print('render_mode',render_mode)
        if render_mode:
            print('initializing pygame...')
            pg.init()

            self.screen = pg.display.set_mode((2*IMG_W, IMG_H + H_INFO))

            # Title
            pg.display.set_caption("Pass the pigs")
            icon = pg.image.load('assets/game/icon.png')
            pg.display.set_icon(icon)

            pg.font.init() # you have to call this at the start, if you want to use this module.
            # print(pg.font.get_fonts())
            self.font_comic = pg.font.SysFont('Comic Sans MS', 30) # This creates a new object on which you can call the render method.
            self.font_default = pg.font.SysFont('Consolas', 20)

            fdir = 'assets/single_images/'
            self.imgs =  glob.glob(fdir+'*.jpg')
            print(self.imgs)

            sample = random.sample(self.imgs, 2)
            file, ext = os.path.splitext(sample[0])
            path, file = os.path.dirname(file), os.path.basename(file)

            # Background
            self.background = pg.image.load('assets/game/assets_logo.png')
            self.bkg = self.background.get_at((10, 10))
            # testing movement...
            self.xCursor = 0 # Creating the variable for the x coordinate of the object
            self.yCursor = 0 # Creating the variable for the y coordinate of the object
            self.vel = 10
            self.count = 0

            # To prevent repeated key events in Pygame, you can use pygame.key.set_repeat() with a delay of 0.
            pg.key.set_repeat(0)


    def _roll(self):
        idx = np.random.choice(range(len(THROWS)), 1, p=[t[1] for t in THROWS])
        self.throw = THROWS[idx[0]]
        if VERBOSE: print(self.throw)
        if self.render_mode:
             # sample = random.sample(self.imgs, 2)
            self.sample = []
            print([img for img in self.imgs if self.throw[0][0] in img])
            s0 = random.sample([img for img in self.imgs if self.throw[0][0] in img], 1)[0]
            self.sample.append(s0)
            print([img for img in self.imgs if self.throw[0][1] in img])
            s1 = random.sample([img for img in self.imgs if self.throw[0][1] in img], 1)[0]
            self.sample.append(s1)


    #INTERACTIVE MODE ONLY!!!!
    def _get_action(self):
          if not self.render_mode == "interactive": return

          action = NONE
          for event in pg.event.get():
              if event.type == pg.QUIT:
                  pg.quit()
              elif event.type == pg.MOUSEBUTTONDOWN:
                  # pygame.mouse.set_visible(0)
                  mouse_x, mouse_y = pg.mouse.get_pos()
                  print(mouse_x,mouse_y)
                  self.xCursor,self.yCursor = mouse_x, mouse_y
              elif event.type == pg.KEYDOWN:
                  if event.key == pg.K_SPACE or event.key == pg.K_r: # Roll
                      pg.event.clear()
                      self.count += 1; print(self.count,'Roll')
                      action = ROLL
                      play_sound = pg.mixer.Sound('assets/game/beep-sound-8333.mp3')
                      play_sound.play() # time.sleep(5); play_sound.stop() # let the sound play for 5 seconds.

                  elif event.key == pg.K_p: # Pass
                      pg.event.clear()
                      self.count += 1; print(self.count,'P')
                      action = PASS

                  elif event.key == pg.K_h or event.key == pg.K_1: # Pass and Hog Call (1)
                      pg.event.clear()
                      self.count += 1; print(self.count,'H_1')
                      action = HOG_CALL_1

                  elif event.key == pg.K_2: # Pass and Hog Call (2)
                      pg.event.clear()
                      self.count += 1; print(self.count,'H_2')
                      action = HOG_CALL_2

          keys = pg.key.get_pressed() # less reliable method in my opinion

          # Let’s make a character move on the screen.
          if keys[pg.K_LEFT]:  self.xCursor -= self.vel
          if keys[pg.K_RIGHT]: self.xCursor += self.vel
          if keys[pg.K_UP]:    self.yCursor -= self.vel
          if keys[pg.K_DOWN]:  self.yCursor += self.vel
          if keys[pg.K_w]:     os.system("beep")

          return action

    def get_action(self,training=False):
        if self.render_mode == "interactive":
            return self._get_action()
        else:
            obs = self._get_obs()
            # print(f'player {self.player}',obs)
            action = self.agents[self.player].predict(obs,training)
            return action

    def _get_obs(self):
        return np.array(self.players[self.player],dtype=int) # observation ONLY for current player

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        # RESET...
        self.throw = None
        self.player = 0
        #---------------------------------------------------------------------------
        # STATE OF THE GAME
        # player data: own total score, opp total score, turn score, hog call placed

        if WITH_HOG_CALLS: # len(ACTIONS) > 2: # option for Hog Calls
            self.players = [[0,0,0,0],
                            [0,0,0,0]]
        else:
            self.players = [[0,0,0],
                            [0,0,0]]

        # WARNING!!!
        # The following initialization DOES NOT WORK (the inner lists would point to the SAME object)
        # self.players = [[0]*self.observation_space.shape[0]]*NUM_PLAYERS
        # Use append()
        self.players = []
        for k in range(NUM_PLAYERS): # ONLY TESTED FOR 2 PLAYERS
            self.players.append([0]*self.observation_space.shape[0])
        # print('self.players',self.players)
        #---------------------------------------------------------------------------
        self.winner  = -1
        self.last_actions = [NONE]*NUM_PLAYERS

        return self._get_obs(), {}

    def done_snapshot(self, reason):
        winners = 'ABX'
        self.last_game_info = f"Winner {winners[self.winner]} ({self.players[0][OWN_SCORE]} vs {self.players[1][OWN_SCORE]}) {reason}"


    def step(self, action):
          player  = self.player
          players = self.players
          done = False; reason = ''; reward = 0 # Just for action NONE (only in interactive mode)
          if action == NONE:
              if self.render_mode != 'interactive':
                raise ValueError(f'Invalid action {action}')
              else:
                return self._get_obs(), reward, done, False, {'reason':reason,'winner':self.winner}

          # RECORD FOR TRAINING PURPOSES
          # copy.deepcopy(x)): A deep copy creates a completely independent copy of the original list, including all nested elements.
          self.old_obs = copy.deepcopy(players)

          # STEP
          opponent = (player + 1) % NUM_PLAYERS # ONLY WORKS FOR TWO PLAYERS
          if action in [PASS, HOG_CALL_1, HOG_CALL_2]:
                      reward = 0
                      players[player][OWN_SCORE] += players[player][TURN_SCORE]
                    #   print(f"[DEBUG] Player {player} scores: own={players[player][OWN_SCORE]}, turn={players[player][TURN_SCORE]}")
                      players[opponent][OPP_SCORE] = players[player][OWN_SCORE]
                      players[player][TURN_SCORE] = 0 # reset

                      if WITH_HOG_CALLS:
                            players[player][HOG_CALL]   = 0  # clear hog call, only one per turn
                            players[opponent][HOG_CALL] = action - 1   #hog call is stored on opponents state because they now need to roll

                      self.player = opponent   #takes turns

          elif action == ROLL:
                      self._roll()
                      score = self.throw[2]
                      # OUTS      = ['PIG_OUT','PIGGYBACK','OINKER']
                      if score == 0: # bad roll...
                          if self.throw[0][0] == 'Oinker':
                                # outcome = OINKER
                                # if VERBOSE: print(OUTS[outcome])
                                reward = -players[player][OWN_SCORE]
                                players[player][OWN_SCORE] = 0 # Back to Zero
                          elif self.throw[0][0] == 'Piggyback':
                                # outcome = PIGGYBACK
                                # if VERBOSE: print(OUTS[outcome])
                                reward = -players[player][OWN_SCORE]
                                players[player][OWN_SCORE] = 0 # Back to Zero
                                #------------------------------
                                done = True; reason = 'Piggyback'; self.winner = opponent; self.done_snapshot(reason)
                                #------------------------------
                          else:
                                # outcome = PIG_OUT
                                # if VERBOSE: print(OUTS[outcome])
                                reward = 0
                                players[player][OWN_SCORE] += 0 # No change

                          players[opponent][OPP_SCORE] = players[player][OWN_SCORE]
                          players[player][TURN_SCORE] = 0 # reset

                          if WITH_HOG_CALLS:
                                players[player][HOG_CALL]   = 0 # clear hog call
                                players[opponent][HOG_CALL] = 0 # no hog call made

                          self.player = opponent       #takes turns

                      else: # good roll...
                        if WITH_HOG_CALLS and players[player][HOG_CALL] > 0: # opponent had made a "hog call"
                            hog_call = players[player][HOG_CALL]
                            if (hog_call==1 and score==HOG_CALL_SCORE_1) or (hog_call==2 and score==HOG_CALL_SCORE_2): # correct hog_call
                                reward = - min(2*score, players[player][OWN_SCORE])
                                players[player][OWN_SCORE] -= 2*score

                                if players[player][OWN_SCORE] < 0:
                                    players[player][OWN_SCORE] = 0

                                players[opponent][OWN_SCORE] += 2*score
                                players[player][OPP_SCORE]   = players[opponent][OWN_SCORE]
                                players[opponent][OPP_SCORE] = players[player][OWN_SCORE]
                                players[player][TURN_SCORE] = 0 # reset
                                players[player][HOG_CALL]   = 0 # clear hog call
                                players[opponent][HOG_CALL] = 0 # no hog call made
                                self.player = opponent

                                reward += 1.0   #intermediate (immediate) rewards for correct hog call

                            else: # incorrect hog_call
                                reward = 2*score

                                players[opponent][OWN_SCORE] -= 2*score
                                if players[opponent][OWN_SCORE] < 0:
                                    players[opponent][OWN_SCORE] = 0
                                players[player][OPP_SCORE]   = players[opponent][OWN_SCORE]

                                players[player][TURN_SCORE] += 2*score # turn score
                                if players[player][TURN_SCORE] + players[player][OWN_SCORE] > GOAL:
                                    players[player][OWN_SCORE] += players[player][TURN_SCORE]
                                    #------------------------------
                                    done = True; reason = 'Goal reached (hog call)'; self.winner = player; self.done_snapshot(reason)
                                    #------------------------------

                                players[player][HOG_CALL]   = 0 # clear hog call
                                players[opponent][HOG_CALL] = 0 # no hog call made

                                reward -= 1.0     #intermediate reward

                        else: # good roll, no hog_call
                            reward = score
                            # outcome = score
                            # if VERBOSE:  print('OUTCOME',outcome)
                            players[player][TURN_SCORE] += score # turn score
                            if players[player][TURN_SCORE] + players[player][OWN_SCORE] > GOAL:
                                players[player][OWN_SCORE] += players[player][TURN_SCORE]
                                #------------------------------
                                done = True; reason = 'Goal reached'; self.winner = player; self.done_snapshot(reason)
                                #------------------------------


          # RECORD FOR TRAINING PURPOSES
          # copy.deepcopy(x)): A deep copy creates a completely independent copy of the original list, including all nested elements.
          self.new_obs = copy.deepcopy(players)
          self.last_actions[player] = action

          if not done and action != ROLL:
             self.player = (self.player + 1) % NUM_PLAYERS

          if done:
              print("DONE!")

          return self._get_obs(), reward, done, False, {'reason':reason,'winner':self.winner}

    def render(self):
          if not self.render_mode: return
          screen = self.screen
          player = self.player
          players = self.players
          # RENDER
          pg.time.delay(1000//FPS) # delay milliseconds 1000/FPS
          screen.fill(self.bkg)
          screen.blit(self.background,(0,0))
          draw_text(screen,self.font_comic,self.last_game_info,(255, 255, 255),(IMG_W,IMG_H-40),center=True,antialias=False)

          if self.throw: # throw:
            # pig1_text = os.path.basename(self.sample[0])
            # pig2_text = os.path.basename(self.sample[1])
            pig1_text = self.throw[0][0]
            pig2_text = self.throw[0][1]
            if 'XPass' in self.sample[0]:
                pig1 = pg.image.load(self.sample[0])
                screen.blit(pig1,(0,0))
                draw_text(screen,self.font_comic,pig1_text,(0, 0, 0),(IMG_W,40),center=True,antialias=False)
            else:
                pig1 = pg.image.load(self.sample[0])
                pig2 = pg.image.load(self.sample[1])
                screen.blit(pig1,(0,0))
                screen.blit(pig2,(IMG_W,0))
                draw_text(screen,self.font_comic,pig1_text,(0, 0, 0),(IMG_W//2,40),center=True,antialias=False)
                draw_text(screen,self.font_comic,pig2_text,(0, 0, 0),(IMG_W+IMG_W//2,40),center=True,antialias=False)
            draw_text(screen,self.font_comic,f'{self.throw[2]} points',(0, 0, 0),(IMG_W,IMG_H-40),center=True,antialias=False)

          # informative texts
          pg.draw.rect(screen, (255, 0, 0),(IMG_W if player > 0 else 0, IMG_H, IMG_W, H_INFO),0,10)
          for p in range(len(players)):
                offx = IMG_W if p>0 else 0
                draw_text(screen,self.font_default,f'Player {p+1}',(255, 255, 255),(offx+10,IMG_H+10+20*0),center=False,antialias=False)
                draw_text(screen,self.font_default,f'Own Score {players[p][OWN_SCORE]}', (255, 255, 255),(offx+10,IMG_H+10+20*1),center=False,antialias=False)
                draw_text(screen,self.font_default,f'Turn Score {players[p][TURN_SCORE]}', (255, 255, 255),(offx+10,IMG_H+10+20*2),center=False,antialias=False)
                draw_text(screen,self.font_default,f'Opp Score {players[p][OPP_SCORE]}', (255, 255, 255),(offx+10,IMG_H+10+20*3),center=False,antialias=False)
                draw_text(screen,self.font_default,f'Hog Call {players[p][HOG_CALL]}', (255, 255, 255),(offx+10,IMG_H+10+20*4),center=False,antialias=False)

          pg.draw.rect(screen, (255, 0, 0),(self.xCursor, self.yCursor, 50, 50),2,10) # rounded rect

          pg.display.update()

          return

    def close(self) -> None:
        if self.render_mode:
            pg.quit()


# In[]: PLAY GAMES
run_game = 0

def play_games(env, max_games=10_000,verbose=False,training=False):
    global run_game
    # isRunning = True
    num_games = 0
    # num_games1 = 0 # games won by player B/opposition
    wins = [0,0]     #wins for player 1 [0], wins for player 2 [1]

    # while(isRunning):
    while num_games < max_games:
        obs, info = env.reset()
        done = False

        if render_mode:
            time.sleep(RENDER_DELAY)
            env.render()


        while(not done):
            action = env.get_action(training)
            obs, reward, done, truncated, info = env.step(action)

            if training:
                for k in range(NUM_PLAYERS):
                    # if change of score in current player: learn (action,reward)
                    old_score = env.old_obs[k][OWN_SCORE]
                    new_score = env.new_obs[k][OWN_SCORE]
                    dif_score = env.new_obs[k][OWN_SCORE] - env.old_obs[k][OWN_SCORE]

                    # rew = dif
                    rew = 0 # sparse rewards   only on win/lose
                    if done:
                        if env.winner==k:
                            rew= 1.0      #reward normalized to 1
                        else:
                            rew= -1.0      #penalty

                    if rew != 0:
                        # print(k,'difs',env.new_obs[k][OWN_SCORE],env.old_obs[k][OWN_SCORE],dif)
                        env.agents[k].learn(
                            env.old_obs[k],
                            env.last_actions[k],
                            env.new_obs[k],
                            rew, # dif,
                            done,
                            truncated,
                            info,
                            k # idx for logging purposes
                            )

            if render_mode:
                time.sleep(RENDER_DELAY)
                env.render()


            # CHECK TERMINATION OF GAME
            if done:
                if env.winner < 0:
                    raise ValueError(f'Invalid winner {env.winner}')

                # num_games1 += env.winner
                wins[env.winner] += 1

                if verbose:
                    print(f"Game {num_games+1}: Winner= Player {env.winner+1} ")
                    print(env.last_game_info)
                    print(env.players)

                env.render()
                if render_mode: time.sleep(RENDER_DELAY)
                break

        num_games += 1
        win_ratio = num_games/num_games*100      #was ratio1
        print(f"NUM GAMES: {num_games}\n")

        if training:
            writer.add_scalar("Games/ratio(0)", win_ratio, run_game)
            writer.add_scalar("Games/ratio(1)", 100 - win_ratio, run_game)
            run_game += 1
            print(f"TRAINING: RUN GAME: {run_game}\n")

        # isRunning = num_games < max_games

    return num_games, wins[1]        #total games, player 2 wins








# In[]: MAIN
if __name__ == "__main__":
    render_mode = RENDER_MODE
    env = PassThePigs_2Players_Env(render_mode = render_mode)

    # from stable_baselines3.common.env_checker import check_env
    # It will check your custom environment and output additional warnings if needed
    # check_env(env) # WARNING!!! Fails mysteriously...

    # BENCHMARK COMPETITION BETWEEN MODELS
    mat = np.zeros((20,20))

    setup = 'QTable_epochs_vs_Roller'
    setup = 'Baseline_vs_Roller'
    setup = 'Baseline_vs_Baseline'

    env.agents[0] = PassThePigsAgent(mode='QTable')

    run_epoch = 0
    GAMES_PER_EPOCH = 10      #used to be 15
    rows= cols = 15           #used to be 20     (so 20x20x15=6000), simplified because this took 10+ hrs to run
    start = time.time()


    # execution of games organized in a sort of matrix disposition for easier presentation/analysis of results
    for row in range(rows):
        for col in range(cols):
            # GAME START...
            # env.agents[0] = PassThePigsAgent(mode='Baseline',threshold=row*5)
            env.agents[1] = PassThePigsAgent(mode='Baseline',threshold=col*5)
            # env.agents[1] = PassThePigsAgent(mode='Roller',threshold=75)
            # env.agents[1] = PassThePigsAgent(mode='Roller',threshold=100) # 100% (always) roll...

            num_games, num_games1 = play_games(env, GAMES_PER_EPOCH,training=TRAINING)

            ratio1 = num_games1/num_games*100
            ellapsed = time.time() - start # This gives the execution time in seconds.
            print(f'\r({row},{col}) {num_games} RATIO: {ratio1:.2f}% {time.strftime("%H:%M:%S", time.gmtime(ellapsed))}'+' '*20, end="")
            if TRAINING:
                writer.add_scalar("ratio(0)", 100-ratio1, run_epoch)
                run_epoch += 1

            mat[row,col] = ratio1

    np.save(f'output/my_mat_2Players_{setup}.npy', mat)
    np.savetxt(f'output/my_mat_2Players_{setup}.csv', mat, delimiter=',')
    rows,cols,size = mat.shape[0],mat.shape[1],mat.size
    print(rows,cols,size,mat.shape,mat.size)
    # my_list: x,y,z
    my_list = np.vstack([np.arange(size)%cols,np.arange(size)//rows,mat.flatten()]).T
    np.save(f'output/my_list_2Players_{setup}.npy', my_list)
    np.savetxt(f'output/my_list_2Players_{setup}.csv', my_list, delimiter=',')

    if env.agents[0].mode == "QTable":
        np.save(f'output/my_qtable_2Players_{model_name}.npy', env.agents[0].QTable)
        save_QT_model(env.agents[0],model_name)

    env.close()
    if TRAINING:
        writer.flush()
        writer.close()

    exit()
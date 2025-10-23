#Pass the Pigs 2 Players SEQUENTIAL VARIANT Env.py

"""
Pass the Pigs - 2 Players Sequential Gymnasium Environment
SEQUENTIAL VARIANT: Players observe opponent's last action and outcome

Key differences from standard variant:
- Enhanced observation space includes opponent's last action and points scored
- Agents can adapt strategy based on what opponent just did
- More realistic gameplay with visible opponent behavior
"""


import glob, os, random, time, copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict
import json
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

# State representation (features) - ENHANCED FOR SEQUENTIAL
OWN_SCORE  = 0
OPP_SCORE  = 1
TURN_SCORE = 2
HOG_CALL   = 3
OPP_LAST_ACTION = 4  # NEW: What action did opponent just take?
OPP_LAST_POINTS = 5  # NEW: How many points did opponent just score?

GOAL = 100 # Points
BUCKET = 5 # Points (for grouping scores into buckets)
MAX_BUCKETS = 10    #if we have hog calls, table becomes higher dimensional

# In[]: SYSTEM DYNAMICS
#----------------------------------------------------------
# SYSTEM DYNAMICS
# Throw, probability, score
THROWS = [[('Oinker', 'Oinker'), 0.0386, 0], [('Piggyback', 'Piggyback'), 0.001, 0], [('Side Pink', 'Side Pink'), 0.0961, 1], [('Side Pink', 'Side Dot'), 0.0961, 0], [('Side Pink', 'Razorback'), 0.0465, 5], [('Side Pink', 'Trotter'), 0.031, 5], [('Side Pink', 'Snouter'), 0.0217, 10], [('Side Pink', 'Leaning Jowler'), 0.0124, 15], [('Side Dot', 'Side Pink'), 0.0961, 0], [('Side Dot', 'Side Dot'), 0.0961, 1], [('Side Dot', 'Razorback'), 0.0465, 5], [('Side Dot', 'Trotter'), 0.031, 5], [('Side Dot', 'Snouter'), 0.0217, 10], [('Side Dot', 'Leaning Jowler'), 0.0124, 15], [('Razorback', 'Side Pink'), 0.0465, 5], [('Razorback', 'Side Dot'), 0.0465, 5], [('Razorback', 'Razorback'), 0.0225, 10], [('Razorback', 'Trotter'), 0.015, 10], [('Razorback', 'Snouter'), 0.0105, 15], [('Razorback', 'Leaning Jowler'), 0.006, 20], [('Trotter', 'Side Pink'), 0.031, 5], [('Trotter', 'Side Dot'), 0.031, 5], [('Trotter', 'Razorback'), 0.015, 10], [('Trotter', 'Trotter'), 0.01, 10], [('Trotter', 'Snouter'), 0.007, 15], [('Trotter', 'Leaning Jowler'), 0.004, 20], [('Snouter', 'Side Pink'), 0.0217, 10], [('Snouter', 'Side Dot'), 0.0217, 10], [('Snouter', 'Razorback'), 0.0105, 15], [('Snouter', 'Trotter'), 0.007, 15], [('Snouter', 'Snouter'), 0.0049, 20], [('Snouter', 'Leaning Jowler'), 0.0028, 25], [('Leaning Jowler', 'Side Pink'), 0.0124, 15], [('Leaning Jowler', 'Side Dot'), 0.0124, 15], [('Leaning Jowler', 'Razorback'), 0.006, 20], [('Leaning Jowler', 'Trotter'), 0.004, 20], [('Leaning Jowler', 'Snouter'), 0.0028, 25], [('Leaning Jowler', 'Leaning Jowler'), 0.0016, 30]]
#----------------------------------------------------------





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
    writer = SummaryWriter()






# +-------------------------------------+
# |         Agent Functions             |
# +-------------------------------------+
STAGE = 0
model_name  = f'QT_Sequential_{STAGE}'

# Training parameters
learning_rate = 0.8
min_LR = 0.1
decay_rate_LR = 1e-6
gamma = 0.95

# Exploration parameters
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 2e-6


def epsilon_greedy_policy(Qtable, state, epsilon):
    random_int = random.uniform(0,1)
    if random_int > epsilon:
        action = np.argmax(Qtable[state])
    else:
        action = random.choice(range(len(ACTIONS)))
    return action


def learning_schedule(episode):
    return min_LR + (learning_rate - min_LR)*np.exp(-decay_rate_LR*episode)

def epsilon_schedule(episode):
    return min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)




def save_QT_model(agent,model_name):
    config = {"num_actions": len(ACTIONS),
              "learning_rate": agent.learning_rate,
              "epsilon": agent.epsilon,
              "episode": agent.episode,
              "variant": "sequential",
             }
    with open('output/'+model_name+'.json', 'w') as f:
        json.dump(config, f, indent=4)
    np.save('output/'+model_name+'.npy', agent.QTable)


def load_QT_model(agent,model_name):
    with open('output/'+model_name+'.json', 'r') as f:
        config = json.load(f)
    num_actions = config['num_actions']
    if num_actions != len(ACTIONS):
        raise ValueError(f'Invalid action space {num_actions}|{len(ACTIONS)}')
    variant = config.get('variant', 'standard')
    if variant != 'sequential':
        print(f"⚠️  Warning: Loading {variant} model into sequential environment")
    agent.learning_rate = config['learning_rate']
    agent.epsilon       = config['epsilon']
    agent.episode       = config['episode']
    agent.QTable = np.load('output/'+model_name+'.npy')






# modes: Roller (prob=p%), Baseline, QTable
class PassThePigsAgent_Sequential():
    def __init__(self,mode,threshold=25):
        self.mode = mode
        if mode == 'QTable':
            # SEQUENTIAL VARIANT: Enhanced state space
            # State: [own_score, opp_score, turn_score, hog_call, opp_last_action, opp_last_points]
            if WITH_HOG_CALLS:
                # Dimensions: own_score_bucket, opp_score_bucket, turn_score_bucket,
                #             hog_call (0-2), opp_last_action (0-3), opp_last_points_bucket (0-9), actions
                self.QTable = np.random.rand(MAX_BUCKETS, MAX_BUCKETS, MAX_BUCKETS,
                                            len(ACTIONS)-1, len(ACTIONS), MAX_BUCKETS,
                                            len(ACTIONS))*0.001
            else:
                # Dimensions: own_score_bucket, opp_score_bucket, turn_score_bucket,
                #             opp_last_action (0-1), opp_last_points_bucket (0-9), actions
                self.QTable = np.random.rand(MAX_BUCKETS, MAX_BUCKETS, MAX_BUCKETS,
                                            2, MAX_BUCKETS, len(ACTIONS))*0.001
            self.epsilon = max_epsilon
            self.learning_rate = learning_rate
            self.episode = 0
            self.episode_reward = 0
            self.episode_rewards = np.zeros(AVG_EP)
        else:
            self.threshold = threshold

    def predict(self,obs,training=False):
        if self.mode == 'QTable':
            # Process observation for sequential variant
            if WITH_HOG_CALLS:
                # obs: [own_score, opp_score, turn_score, hog_call, opp_last_action, opp_last_points]
                bucketed = np.minimum(obs[:3]//BUCKET, MAX_BUCKETS-1)  # Bucket scores
                hog_call = obs[3:4]  # Keep hog call as is
                opp_action = obs[4:5]  # Keep action as is
                opp_points_bucket = np.minimum(obs[5:6]//BUCKET, MAX_BUCKETS-1)  # Bucket points
                obs_processed = np.concatenate([bucketed, hog_call, opp_action, opp_points_bucket])
            else:
                # obs: [own_score, opp_score, turn_score, opp_last_action, opp_last_points]
                bucketed = np.minimum(obs[:3]//BUCKET, MAX_BUCKETS-1)
                opp_action = obs[3:4]
                opp_points_bucket = np.minimum(obs[4:5]//BUCKET, MAX_BUCKETS-1)
                obs_processed = np.concatenate([bucketed, opp_action, opp_points_bucket])

            state = tuple(obs_processed.astype(int))
            action = epsilon_greedy_policy(self.QTable, state, self.epsilon if training else 0)

        elif self.mode == 'Baseline':
            # Baseline can now use opponent information strategically
            action = ROLL
            # More aggressive if opponent just scored big
            threshold_adjust = self.threshold
            if len(obs) >= 6 and obs[OPP_LAST_POINTS] >= 10:
                threshold_adjust -= 5  # Be more aggressive

            if obs[TURN_SCORE] > threshold_adjust and ((obs[OWN_SCORE] + obs[TURN_SCORE]) > obs[OPP_SCORE]):
                action = PASS
        else:  # Roller
            prob_roll = self.threshold/100
            action = np.random.choice([ROLL,PASS], 1, p=[prob_roll,(1-prob_roll)])[0]
        return action


    def learn(self,old_obs,action,new_obs,reward,terminated,truncated,info,idx):
        if self.mode == 'QTable':
            old_obs = np.array(old_obs)
            new_obs = np.array(new_obs)

            if WITH_HOG_CALLS:
                # Process old state
                bucketed_old = np.minimum(old_obs[:3]//BUCKET, MAX_BUCKETS-1)
                hog_call_old = old_obs[3:4]
                opp_action_old = old_obs[4:5]
                opp_points_bucket_old = np.minimum(old_obs[5:6]//BUCKET, MAX_BUCKETS-1)
                old_obs_processed = np.concatenate([bucketed_old, hog_call_old, opp_action_old, opp_points_bucket_old])

                # Process new state
                bucketed_new = np.minimum(new_obs[:3]//BUCKET, MAX_BUCKETS-1)
                hog_call_new = new_obs[3:4]
                opp_action_new = new_obs[4:5]
                opp_points_bucket_new = np.minimum(new_obs[5:6]//BUCKET, MAX_BUCKETS-1)
                new_obs_processed = np.concatenate([bucketed_new, hog_call_new, opp_action_new, opp_points_bucket_new])
            else:
                bucketed_old = np.minimum(old_obs[:3]//BUCKET, MAX_BUCKETS-1)
                opp_action_old = old_obs[3:4]
                opp_points_bucket_old = np.minimum(old_obs[4:5]//BUCKET, MAX_BUCKETS-1)
                old_obs_processed = np.concatenate([bucketed_old, opp_action_old, opp_points_bucket_old])

                bucketed_new = np.minimum(new_obs[:3]//BUCKET, MAX_BUCKETS-1)
                opp_action_new = new_obs[3:4]
                opp_points_bucket_new = np.minimum(new_obs[4:5]//BUCKET, MAX_BUCKETS-1)
                new_obs_processed = np.concatenate([bucketed_new, opp_action_new, opp_points_bucket_new])

            old_state = tuple(old_obs_processed.astype(int))
            new_state = tuple(new_obs_processed.astype(int))

            #--------------------------------------------------------------------------------
            qValue = self.QTable[old_state][action]
            qPred  = reward + gamma * np.max(self.QTable[new_state])*(not terminated)
            self.QTable[old_state][action] = qValue + self.learning_rate * ( qPred - qValue )
            #--------------------------------------------------------------------------------
            self.episode_reward += reward

            if terminated:
                self.episode_rewards[self.episode % AVG_EP] = self.episode_reward
                self.episode += 1
                writer.add_scalar(f"Sequential_Agent({idx})/reward", self.episode_reward, self.episode)
                writer.add_scalar(f"Sequential_Agent({idx})/reward_{AVG_EP}", np.sum(self.episode_rewards)/AVG_EP, self.episode)
                writer.add_scalar(f"Sequential_Agent({idx})/epsilon", self.epsilon, self.episode)
                writer.add_scalar(f"Sequential_Agent({idx})/learning_rate", self.learning_rate, self.episode)
                self.epsilon = epsilon_schedule(self.episode)
                self.learning_rate = learning_schedule(self.episode)
                self.episode_reward = 0






# In[]: ENVIRONMENT
# +-------------------------------------+
# |      Environment Functions          |
# +-------------------------------------+
class PassThePigs_2Players_Sequential_Env(gym.Env):
    """
    Sequential variant of Pass the Pigs.

    Key difference: Observation includes opponent's last action and points scored.
    This allows agents to adapt their strategy based on opponent behavior.
    """

    metadata = {
        "render_modes": [
            "interactive",
            "human",
            "rgb_array",
        ],
        "variant": "sequential",
        "FPS": 20,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super(PassThePigs_2Players_Sequential_Env, self).__init__()
        self.render_mode = render_mode
        self.last_game_info = ""

        self.players = []
        self.agents  = [None, None]
        self.winner  = -1

        self.action_space = spaces.Discrete(len(ACTIONS))

        # SEQUENTIAL VARIANT: Enhanced observation space
        # [own_score, opp_score, turn_score, hog_call, opp_last_action, opp_last_points]
        if WITH_HOG_CALLS:
            low  = np.array([0, 0, 0, 0, 0, 0])
            high = np.array([GOAL, GOAL, GOAL, len(ACTIONS)-2, len(ACTIONS)-1, GOAL])
        else:
            low  = np.array([0, 0, 0, 0, 0])
            high = np.array([GOAL, GOAL, GOAL, len(ACTIONS)-1, GOAL])

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=int)

        print('🎲 SEQUENTIAL VARIANT initialized')
        print(f'   Observation space: {self.observation_space.shape}')
        print(f'   render_mode: {render_mode}')

        if render_mode:
            print('   Initializing pygame...')
            pg.init()
            self.screen = pg.display.set_mode((2*IMG_W, IMG_H + H_INFO))
            pg.display.set_caption("Pass the Pigs - Sequential Variant")
            icon = pg.image.load('assets/game/icon.png')
            pg.display.set_icon(icon)
            pg.font.init()
            self.font_comic = pg.font.SysFont('Comic Sans MS', 30)
            self.font_default = pg.font.SysFont('Consolas', 20)
            fdir = 'assets/single_images/'
            self.imgs = glob.glob(fdir+'*.jpg')
            self.background = pg.image.load('assets/game/assets_logo.png')
            self.bkg = self.background.get_at((10, 10))
            self.xCursor = 0
            self.yCursor = 0
            self.vel = 10
            self.count = 0
            pg.key.set_repeat(0)


    def _roll(self):
        idx = np.random.choice(range(len(THROWS)), 1, p=[t[1] for t in THROWS])
        self.throw = THROWS[idx[0]]
        if VERBOSE: print(self.throw)
        if self.render_mode:
            self.sample = []
            s0 = random.sample([img for img in self.imgs if self.throw[0][0] in img], 1)[0]
            self.sample.append(s0)
            s1 = random.sample([img for img in self.imgs if self.throw[0][1] in img], 1)[0]
            self.sample.append(s1)


    def _get_action(self):
          if not self.render_mode == "interactive": return

          action = NONE
          for event in pg.event.get():
              if event.type == pg.QUIT:
                  pg.quit()
              elif event.type == pg.MOUSEBUTTONDOWN:
                  mouse_x, mouse_y = pg.mouse.get_pos()
                  self.xCursor,self.yCursor = mouse_x, mouse_y
              elif event.type == pg.KEYDOWN:
                  if event.key == pg.K_SPACE or event.key == pg.K_r:
                      pg.event.clear()
                      self.count += 1
                      action = ROLL
                      play_sound = pg.mixer.Sound('assets/game/beep-sound-8333.mp3')
                      play_sound.play()
                  elif event.key == pg.K_p:
                      pg.event.clear()
                      self.count += 1
                      action = PASS
                  elif event.key == pg.K_h or event.key == pg.K_1:
                      pg.event.clear()
                      self.count += 1
                      action = HOG_CALL_1
                  elif event.key == pg.K_2:
                      pg.event.clear()
                      self.count += 1
                      action = HOG_CALL_2

          keys = pg.key.get_pressed()
          if keys[pg.K_LEFT]:  self.xCursor -= self.vel
          if keys[pg.K_RIGHT]: self.xCursor += self.vel
          if keys[pg.K_UP]:    self.yCursor -= self.vel
          if keys[pg.K_DOWN]:  self.yCursor += self.vel

          return action

    def get_action(self,training=False):
        if self.render_mode == "interactive":
            return self._get_action()
        else:
            obs = self._get_obs()
            action = self.agents[self.player].predict(obs,training)
            return action

    def _get_obs(self):
        return np.array(self.players[self.player],dtype=int)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.throw = None
        self.player = 0

        # SEQUENTIAL VARIANT: Initialize with enhanced state
        # [own_score, opp_score, turn_score, hog_call, opp_last_action, opp_last_points]
        self.players = []
        for k in range(NUM_PLAYERS):
            if WITH_HOG_CALLS:
                # Initialize: [own_score=0, opp_score=0, turn_score=0, hog_call=0, opp_last_action=0, opp_last_points=0]
                self.players.append([0, 0, 0, 0, 0, 0])
            else:
                # Initialize: [own_score=0, opp_score=0, turn_score=0, opp_last_action=0, opp_last_points=0]
                self.players.append([0, 0, 0, 0, 0])

        self.winner  = -1
        self.last_actions = [ROLL] * NUM_PLAYERS  # Initialize to ROLL (valid action)
        self.last_points = [0] * NUM_PLAYERS  # Track points scored by each player

        return self._get_obs(), {}

    def done_snapshot(self, reason):
        winners = 'ABX'
        self.last_game_info = f"Winner {winners[self.winner]} ({self.players[0][OWN_SCORE]} vs {self.players[1][OWN_SCORE]}) {reason}"


    def step(self, action):
          player  = self.player
          players = self.players
          done = False
          reason = ''
          reward = 0
          points_this_action = 0  # Track points scored this action

          if action == NONE:
              if self.render_mode != 'interactive':
                raise ValueError(f'Invalid action {action}')
              else:
                return self._get_obs(), reward, done, False, {'reason':reason,'winner':self.winner}

          # RECORD FOR TRAINING
          self.old_obs = copy.deepcopy(players)

          # STEP
          opponent = (player + 1) % NUM_PLAYERS

          if action in [PASS, HOG_CALL_1, HOG_CALL_2]:
                      reward = 0
                      points_this_action = players[player][TURN_SCORE]  # Banking turn score
                      players[player][OWN_SCORE] += players[player][TURN_SCORE]
                      players[opponent][OPP_SCORE] = players[player][OWN_SCORE]
                      players[player][TURN_SCORE] = 0

                      if WITH_HOG_CALLS:
                            players[player][HOG_CALL]   = 0
                            players[opponent][HOG_CALL] = action - 1

                      # UPDATE SEQUENTIAL INFO: Opponent will see this action
                      players[opponent][OPP_LAST_ACTION] = action
                      players[opponent][OPP_LAST_POINTS] = points_this_action

                      self.player = opponent

          elif action == ROLL:
                      self._roll()
                      score = self.throw[2]

                      if score == 0:  # Bad roll
                          if self.throw[0][0] == 'Oinker':
                                reward = -players[player][OWN_SCORE]
                                points_this_action = -players[player][OWN_SCORE]
                                players[player][OWN_SCORE] = 0
                          elif self.throw[0][0] == 'Piggyback':
                                reward = -players[player][OWN_SCORE]
                                points_this_action = -players[player][OWN_SCORE]
                                players[player][OWN_SCORE] = 0
                                done = True
                                reason = 'Piggyback'
                                self.winner = opponent
                                self.done_snapshot(reason)
                          else:
                                reward = 0
                                points_this_action = 0
                                players[player][OWN_SCORE] += 0

                          players[opponent][OPP_SCORE] = players[player][OWN_SCORE]
                          players[player][TURN_SCORE] = 0

                          if WITH_HOG_CALLS:
                                players[player][HOG_CALL]   = 0
                                players[opponent][HOG_CALL] = 0

                          # UPDATE SEQUENTIAL INFO
                          players[opponent][OPP_LAST_ACTION] = action
                          players[opponent][OPP_LAST_POINTS] = points_this_action

                          self.player = opponent

                      else:  # Good roll
                        if WITH_HOG_CALLS and players[player][HOG_CALL] > 0:
                            hog_call = players[player][HOG_CALL]
                            if (hog_call==1 and score==HOG_CALL_SCORE_1) or (hog_call==2 and score==HOG_CALL_SCORE_2):
                                # Correct hog call
                                reward = - min(2*score, players[player][OWN_SCORE])
                                points_this_action = - min(2*score, players[player][OWN_SCORE])
                                players[player][OWN_SCORE] -= 2*score

                                if players[player][OWN_SCORE] < 0:
                                    players[player][OWN_SCORE] = 0

                                players[opponent][OWN_SCORE] += 2*score
                                players[player][OPP_SCORE]   = players[opponent][OWN_SCORE]
                                players[opponent][OPP_SCORE] = players[player][OWN_SCORE]
                                players[player][TURN_SCORE] = 0
                                players[player][HOG_CALL]   = 0
                                players[opponent][HOG_CALL] = 0

                                # UPDATE SEQUENTIAL INFO
                                players[opponent][OPP_LAST_ACTION] = action
                                players[opponent][OPP_LAST_POINTS] = points_this_action

                                self.player = opponent
                                reward += 1.0

                            else:
                                # Incorrect hog call
                                reward = 2*score
                                points_this_action = 2*score

                                players[opponent][OWN_SCORE] -= 2*score
                                if players[opponent][OWN_SCORE] < 0:
                                    players[opponent][OWN_SCORE] = 0
                                players[player][OPP_SCORE]   = players[opponent][OWN_SCORE]

                                players[player][TURN_SCORE] += 2*score
                                if players[player][TURN_SCORE] + players[player][OWN_SCORE] > GOAL:
                                    players[player][OWN_SCORE] += players[player][TURN_SCORE]
                                    done = True
                                    reason = 'Goal reached (hog call)'
                                    self.winner = player
                                    self.done_snapshot(reason)

                                players[player][HOG_CALL]   = 0
                                players[opponent][HOG_CALL] = 0

                                # UPDATE SEQUENTIAL INFO (stays with current player)
                                players[opponent][OPP_LAST_ACTION] = action
                                players[opponent][OPP_LAST_POINTS] = points_this_action

                                reward -= 1.0

                        else:  # Good roll, no hog call
                            reward = score
                            points_this_action = score
                            players[player][TURN_SCORE] += score

                            if players[player][TURN_SCORE] + players[player][OWN_SCORE] > GOAL:
                                players[player][OWN_SCORE] += players[player][TURN_SCORE]
                                done = True
                                reason = 'Goal reached'
                                self.winner = player
                                self.done_snapshot(reason)

                            # UPDATE SEQUENTIAL INFO (player continues, but update for future)
                            # This will be visible when turn switches
                            players[opponent][OPP_LAST_ACTION] = action
                            players[opponent][OPP_LAST_POINTS] = points_this_action

          # RECORD FOR TRAINING
          self.new_obs = copy.deepcopy(players)
          self.last_actions[player] = action
          self.last_points[player] = points_this_action

          if done:
              print("DONE!")

          return self._get_obs(), reward, done, False, {'reason':reason,'winner':self.winner}

    def render(self):
          if not self.render_mode: return
          screen = self.screen
          player = self.player
          players = self.players

          pg.time.delay(1000//FPS)
          screen.fill(self.bkg)
          screen.blit(self.background,(0,0))
          draw_text(screen,self.font_comic,self.last_game_info,(255, 255, 255),(IMG_W,IMG_H-40),center=True,antialias=False)

          if self.throw:
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

          # Informative texts - ENHANCED FOR SEQUENTIAL
          pg.draw.rect(screen, (255, 0, 0),(IMG_W if player > 0 else 0, IMG_H, IMG_W, H_INFO),0,10)
          for p in range(len(players)):
                offx = IMG_W if p>0 else 0
                draw_text(screen,self.font_default,f'Player {p+1}',(255, 255, 255),(offx+10,IMG_H+10+20*0),center=False,antialias=False)
                draw_text(screen,self.font_default,f'Own Score {players[p][OWN_SCORE]}', (255, 255, 255),(offx+10,IMG_H+10+20*1),center=False,antialias=False)
                draw_text(screen,self.font_default,f'Turn Score {players[p][TURN_SCORE]}', (255, 255, 255),(offx+10,IMG_H+10+20*2),center=False,antialias=False)
                draw_text(screen,self.font_default,f'Opp Score {players[p][OPP_SCORE]}', (255, 255, 255),(offx+10,IMG_H+10+20*3),center=False,antialias=False)
                if WITH_HOG_CALLS:
                    draw_text(screen,self.font_default,f'Hog Call {players[p][HOG_CALL]}', (255, 255, 255),(offx+10,IMG_H+10+20*4),center=False,antialias=False)
                # SEQUENTIAL INFO DISPLAY
                action_names = ['ROLL', 'PASS', 'HC1', 'HC2']
                opp_action_idx = int(players[p][OPP_LAST_ACTION]) if players[p][OPP_LAST_ACTION] < len(action_names) else 0
                draw_text(screen,self.font_default,f'Opp Last: {action_names[opp_action_idx]} ({int(players[p][OPP_LAST_POINTS])}pts)',
                         (255, 255, 0),(offx+10,IMG_H+10+20*5),center=False,antialias=False)

          pg.draw.rect(screen, (255, 0, 0),(self.xCursor, self.yCursor, 50, 50),2,10)
          pg.display.update()
          return

    def close(self) -> None:
        if self.render_mode:
            pg.quit()








# In[]: MAIN
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PASS THE PIGS - SEQUENTIAL VARIANT")
    print("="*60)
    print("This variant allows agents to observe opponent's last action")
    print("and points scored, enabling adaptive strategies.")
    print("="*60 + "\n")

    render_mode = None  # Set to 'human' for visual
    env = PassThePigs_2Players_Sequential_Env(render_mode=render_mode)

    # Simple test
    env.agents[0] = PassThePigsAgent_Sequential(mode='Baseline', threshold=25)
    env.agents[1] = PassThePigsAgent_Sequential(mode='Baseline', threshold=20)

    print("\nRunning 10 test games...")
    wins = [0, 0]
    for game in range(10):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            action = env.get_action(training=False)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

            if done:
                wins[env.winner] += 1
                print(f"Game {game+1}: Winner = Player {env.winner+1}, Steps = {steps}")
                break

    print(f"\nResults: Player 1: {wins[0]} wins, Player 2: {wins[1]} wins")
    print(f"Win rate P1: {wins[0]/10*100:.1f}%")

    env.close()
    if TRAINING:
        writer.flush()
        writer.close()

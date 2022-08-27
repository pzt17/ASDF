import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
from dqn import *

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

from wrappers import make_atari, wrap_deepmind, wrap_pytorch

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

model = CnnDQN(env.observation_space.shape, env.action_space.n)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

if USE_CUDA:
    model = model.cuda()
    
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)



num_frames = 20000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    
    action = model.act(state, epsilon) 
    next_state, reward, done, _ = env.step(action) 
    env.render()
    state = next_state
    episode_reward += reward
   
    if done:
        state = env.reset()
        print('Episode Reward:', episode_reward)
        all_rewards.append(episode_reward)
        episode_reward = 0
	
clear_output(True)
plt.figure(figsize=(20,5))
#plt.subplot(131)
plt.title('Pong - Reward without Attack')
plt.plot(all_rewards)
plt.xlabel('episodes')
plt.ylabel('Reward')
plt.show()
plt.close()



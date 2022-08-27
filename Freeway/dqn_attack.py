import math, random
import gym
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import random

import torch.autograd as autograd 
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms

from dqn import *
#from pgd import * 


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

from wrappers import make_atari, wrap_deepmind, wrap_pytorch

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")



model = CnnDQN(env.observation_space.shape, env.action_space.n)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

if USE_CUDA:
    model = model.cuda()
    
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)




num_frames = 10000
batch_size = 5
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0
replay_initial = 5
replay_buffer = ReplayBuffer(100000)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

def compute_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.order_sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    state.requires_grad = True;
    
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    next_state.requires_grad = True
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
    
   
    optimizer.zero_grad()
    loss.backward()

    return state.grad.sign()


def pgd_attack( images, alpha= 5.0) :
    states_loss = compute_loss(batch_size)
    images  = Variable(torch.FloatTensor(np.float32(images)))
    Jx = states_loss[0]
    images = images + alpha*Jx
    #images  = images + alpha*random.randrange(-1,2,2)
    return images

state = env.reset()
ave_award = 0.0
round_cnt = 0
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    
    action = model.act(state, epsilon)
    adv_state = state
    if len(replay_buffer) >= replay_initial and random.randrange(100)%4 == 0:
      	snapshot = env.ale.cloneState()
      	next_state, reward, done, _ = env.step(action)
      	env.ale.restoreState(snapshot)
      	replay_buffer.push(state, action, reward, next_state, done)
      	adv_state = pgd_attack(state)
      	adv_state = adv_state.cpu().data.numpy()
      	replay_buffer.remove_right()
    	
    action = model.act(adv_state,epsilon)
    next_state, reward, done, _ = env.step(action) 
    env.render()
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward
   
    if done:
        state = env.reset()
        print('Episode Reward:', episode_reward)
        all_rewards.append(episode_reward)
        ave_award += episode_reward
        episode_reward = 0
        round_cnt += 1
        
print('Average Reward:', ave_award/round_cnt)
	
clear_output(True)
plt.figure(figsize=(20,5))
#plt.subplot(131)
plt.title('Pong - Reward with Attack')
plt.plot(all_rewards)
plt.xlabel('episodes')
plt.ylabel('Reward')
plt.show()
plt.close()



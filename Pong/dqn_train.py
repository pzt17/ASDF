import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from collections import deque
from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from IPython.display import clear_output
import matplotlib.pyplot as plt

from dqn import *

torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
	
model = CnnDQN(env.observation_space.shape, env.action_space.n)
#delete the two lines below if you don't want to retrain, or continue training the model
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()
##########
if USE_CUDA:
    	model = model.cuda()
    
optimizer = optim.Adam(model.parameters(), lr=0.00001)

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)
	
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
	
num_frames = 50000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
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
    optimizer.step()
    
    return loss
    
def plot(frame_idx, rewards, losses):

    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()
    plt.close()
  
    



state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        print("Reward:",episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(batch_size)
        losses.append(loss.data.item())
        if frame_idx % 10000 == 0:
        	print('Loss:',loss.data.item())
        
    if frame_idx % 10000 == 0:
    	print('Frame Index:',frame_idx)
    	print(" ")	
        
	
clear_output(True)
plt.figure(figsize=(20,5))
plt.subplot(131)
plt.title('Reward')
plt.plot(all_rewards)
plt.subplot(132)
plt.title('loss')
plt.plot(losses)
plt.show()
plt.close()
          		
#Delete the '#' symbol in next line, if you want to retrain the model.
#torch.save(model.state_dict(),'checkpoint.pth')
	



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
import queue
from queue import Queue

from dqn import *
from asdn import *


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

from wrappers import make_atari, wrap_deepmind, wrap_pytorch

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

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




num_frames = 30000
batch_size = 1
gamma      = 0.99

all_rewards = []
episode_reward = 0
replay_initial = 50 
replay_buffer = ReplayBuffer(100000)

optimizer = optim.Adam(model.parameters(), lr=0.00001)

def compute_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

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
    Jx = states_loss[batch_size-1]
    images = images + alpha*Jx
    return images
   

state = env.reset()
ave_award = 0.0
round_cnt = 0

asdn0 = ASDN(env.observation_space.shape)
asdn0.load_state_dict(torch.load('ASDNpoint0.pth'))
asdn0.eval()
if USE_CUDA:
    asdn0 = asdn0.cuda()

asdn1 = ASDN(env.observation_space.shape)
asdn1.load_state_dict(torch.load('ASDNpoint1.pth'))
asdn1.eval()
if USE_CUDA:
    asdn1 = asdn1.cuda()
    
asdn2 = ASDN(env.observation_space.shape)
asdn2.load_state_dict(torch.load('ASDNpoint2.pth'))
asdn2.eval()
if USE_CUDA:
    asdn2 = asdn2.cuda()

asdn3 = ASDN(env.observation_space.shape)
asdn3.load_state_dict(torch.load('ASDNpoint3.pth'))
asdn3.eval()
if USE_CUDA:
    asdn3 = asdn3.cuda()
    
asdn4 = ASDN(env.observation_space.shape)
asdn4.load_state_dict(torch.load('ASDNpoint4.pth'))
asdn4.eval()
if USE_CUDA:
    asdn4 = asdn4.cuda()
    
def asdn_predict(cur_state, prev_idx):
    state, action, reward, next_state, done = replay_buffer.sample(5)
    cur_state = cur_state - state[prev_idx]
    cur_state = np.expand_dims(cur_state,axis = 0)
    cur_state = Variable(torch.FloatTensor(np.float32(cur_state)))
    
    if prev_idx == 0:
    	prob_value  = asdn0(cur_state)
    if prev_idx == 1:
    	prob_value  = asdn1(cur_state)
    if prev_idx == 2:
    	prob_value  = asdn2(cur_state)
    if prev_idx == 3:
    	prob_value  = asdn3(cur_state)
    if prev_idx == 4:
    	prob_value  = asdn4(cur_state)
    	
    m = nn.Sigmoid()
    prob_value = m(prob_value)
    return prob_value[0].data.item()


  
    
losses = []
correct = []

for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    adv_state = state
    if frame_idx%2 == 1 and len(replay_buffer) >= 6:
    	snapshot = env.ale.cloneState()
    	next_state, reward, done, _ = env.step(action)
    	env.ale.restoreState(snapshot)
    	replay_buffer.push(state, action, reward, next_state, done)
    	adv_state = pgd_attack(state)
    	adv_state = adv_state.cpu().data.numpy()
    	replay_buffer.remove_right()
    	
    	pv_sum = 0.0
    	for i in range(5):
    		cur_pv = asdn_predict(adv_state,i)
    		pv_sum = pv_sum + cur_pv
    	pv_sum /= 5.0
    	if pv_sum >= 0.5:
    		correct.append(1)
    	else:
    		correct.append(0)
    	
    elif frame_idx%2 == 0 and len(replay_buffer) >= 6:
    	pv_sum = 0.0
    	for i in range(5):
    		cur_pv = asdn_predict(state,i)
    		pv_sum = pv_sum + cur_pv
    	pv_sum /= 5.0
    	if pv_sum < 0.5:
    		correct.append(1)
    	else:
    		correct.append(0)
  
    	
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
        
#print('Aveage Loss:')
#print(sum(losses)/len(losses))
#print(correct)
print('Accuracy:')
print(sum(correct)/len(correct))
	



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


model = CnnDQN(env.observation_space.shape, env.action_space.n)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

if USE_CUDA:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)




num_frames = 20000
batch_size = 1
gamma      = 0.99

all_rewards = []
episode_reward = 0
replay_initial = 50 
replay_buffer = ReplayBuffer(10)

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
    images  = Variable(torch.FloatTensor(np.float32(images)))
    images = images + random.uniform(0.01,alpha) * random.randrange(-1,2,2)
    return images
   

state = env.reset()
ave_award = 0.0
round_cnt = 0

asdn0 = ASDN(env.observation_space.shape)
#asdn0.load_state_dict(torch.load('ASDNpoint0.pth'))
#asdn0.eval()
if USE_CUDA:
    asdn0 = asdn0.cuda()
optimizer0 = optim.SGD(asdn0.parameters(), lr=0.01)

asdn1 = ASDN(env.observation_space.shape)
#asdn1.load_state_dict(torch.load('ASDNpoint1.pth'))
#asdn1.eval()
if USE_CUDA:
    asdn1 = asdn1.cuda()
optimizer1 = optim.SGD(asdn0.parameters(), lr=0.01)

asdn2 = ASDN(env.observation_space.shape)
#asdn2.load_state_dict(torch.load('ASDNpoint2.pth'))
#asdn2.eval()
if USE_CUDA:
    asdn2 = asdn2.cuda()
optimizer2 = optim.SGD(asdn2.parameters(), lr=0.01)

asdn3 = ASDN(env.observation_space.shape)
#asdn3.load_state_dict(torch.load('ASDNpoint3.pth'))
#asdn3.eval()
if USE_CUDA:
    asdn3 = asdn3.cuda()
optimizer3 = optim.SGD(asdn3.parameters(), lr=0.01)

asdn4 = ASDN(env.observation_space.shape)
#asdn4.load_state_dict(torch.load('ASDNpoint4.pth'))
#asdn4.eval()
if USE_CUDA:
    asdn4 = asdn4.cuda()
optimizer4 = optim.SGD(asdn4.parameters(), lr=0.01)

def asdn_loss0(flag,cur_state):
    states, action, reward, next_state, done   = replay_buffer.order_sample(5)
    cur_state = cur_state - states[0]
    cur_state = np.expand_dims(cur_state,axis = 0)
    cur_state = Variable(torch.FloatTensor(np.float32(cur_state)),requires_grad = True)
    prob_value  = asdn0(cur_state)
    m = nn.Sigmoid()
    prob_value = m(prob_value)
    target_value = torch.FloatTensor([[0.0]])
    if flag == 1:
    	target_value = torch.FloatTensor([[1.0]])
	
    loss = (prob_value - Variable(target_value)).pow(2).mean()
   
    optimizer0.zero_grad()
    loss.backward()
    optimizer0.step()
    
    return loss
    
def asdn_loss1(flag,cur_state):
    states, action, reward, next_state, done   = replay_buffer.order_sample(5)
    cur_state = cur_state - states[1]
    cur_state = np.expand_dims(cur_state,axis = 0)
    cur_state = Variable(torch.FloatTensor(np.float32(cur_state)),requires_grad = True)
    prob_value  = asdn1(cur_state)
    m = nn.Sigmoid()
    prob_value = m(prob_value)
    target_value = torch.FloatTensor([[0.0]])
    if flag == 1:
    	target_value = torch.FloatTensor([[1.0]])
	
    loss = (prob_value - Variable(target_value)).pow(2).mean()
   
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    
    return loss

def asdn_loss2(flag,cur_state):
    states, action, reward, next_state, done   = replay_buffer.order_sample(5)
    cur_state = cur_state - states[2]
    cur_state = np.expand_dims(cur_state,axis = 0)
    cur_state = Variable(torch.FloatTensor(np.float32(cur_state)),requires_grad = True)
    prob_value  = asdn2(cur_state)
    m = nn.Sigmoid()
    prob_value = m(prob_value)
    target_value = torch.FloatTensor([[0.0]])
    if flag == 1:
    	target_value = torch.FloatTensor([[1.0]])
	
    loss = (prob_value - Variable(target_value)).pow(2).mean()
   
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    
    return loss
    
def asdn_loss3(flag,cur_state):
    states, action, reward, next_state, done   = replay_buffer.order_sample(5)
    cur_state = cur_state - states[3]
    cur_state = np.expand_dims(cur_state,axis = 0)
    cur_state = Variable(torch.FloatTensor(np.float32(cur_state)),requires_grad = True)
    prob_value  = asdn3(cur_state)
    m = nn.Sigmoid()
    prob_value = m(prob_value)
    target_value = torch.FloatTensor([[0.0]])
    if flag == 1:
    	target_value = torch.FloatTensor([[1.0]])
	
    loss = (prob_value - Variable(target_value)).pow(2).mean()
   
    optimizer3.zero_grad()
    loss.backward()
    optimizer3.step()
    
    return loss

def asdn_loss4(flag,cur_state):
    states, action, reward, next_state, done   = replay_buffer.order_sample(5)
    cur_state = cur_state - states[4]
    cur_state = np.expand_dims(cur_state,axis = 0)
    cur_state = Variable(torch.FloatTensor(np.float32(cur_state)),requires_grad = True)
    prob_value  = asdn4(cur_state)
    m = nn.Sigmoid()
    prob_value = m(prob_value)
    target_value = torch.FloatTensor([[0.0]])
    if flag == 1:
    	target_value = torch.FloatTensor([[1.0]])
	
    loss = (prob_value - Variable(target_value)).pow(2).mean()
   
    optimizer4.zero_grad()
    loss.backward()
    optimizer4.step()
    
    return loss
    
losses = []


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
    	
    	loss0 = asdn_loss0(1,adv_state)
    	loss1 = asdn_loss1(1,adv_state)
    	loss2 = asdn_loss2(1,adv_state)
    	loss3 = asdn_loss3(1,adv_state)
    	loss4 = asdn_loss4(1,adv_state)
    	loss = loss0 + loss1 + loss2 + loss3 + loss4 
    	losses.append(loss)
    	#print(loss)
    	
    elif frame_idx%2 == 0 and len(replay_buffer) >= 6:
    	loss0 = asdn_loss0(0,state)
    	loss1 = asdn_loss1(0,state)
    	loss2 = asdn_loss2(0,state)
    	loss3 = asdn_loss3(0,state)
    	loss4 = asdn_loss4(0,state)
    	loss = loss0 + loss1 + loss2 + loss3 + loss4 
    	losses.append(loss)
    	#print(loss)
 
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
        
print('Aveage Loss:')
print(sum(losses)/len(losses))
	
#Delete the '#' symbol in next line, if you want to retrain the ASDN.
torch.save(asdn0.state_dict(),'ASDNpoint0.pth')
torch.save(asdn1.state_dict(),'ASDNpoint1.pth')
torch.save(asdn2.state_dict(),'ASDNpoint2.pth')
torch.save(asdn3.state_dict(),'ASDNpoint3.pth')
torch.save(asdn4.state_dict(),'ASDNpoint4.pth')


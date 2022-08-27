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
    
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 200000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)




num_frames = 30000
batch_size = 32
gamma      = 0.99

all_rewards = []
episode_reward = 0
replay_initial = 50 
replay_buffer = ReplayBuffer(100000)

optimizer = optim.Adam(model.parameters(), lr=0.00001)


adv_buffer = deque(maxlen=100000)

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

ANET = CnnDQN(env.observation_space.shape, env.action_space.n)
#delete the two lines below if you don't want to retrain, or continue training the model
ANET.load_state_dict(torch.load('ANETpoint.pth'))
ANET.eval()
##########
if USE_CUDA:
    ANET = ANET.cuda()
   
adv_optimizer = optim.Adam(ANET.parameters(), lr=0.00001)
    
def anet_loss(bs = 32):
	real_states, adv_states = zip(*random.sample(adv_buffer, bs))
	real_states = Variable(torch.FloatTensor(np.float32(real_states)))
	adv_states = Variable(torch.FloatTensor(np.float32(adv_states)),requires_grad = True)
	real_q = model(real_states)
	
	real_act = torch.zeros((bs,env.action_space.n))
	for i in range(bs):
		real_act[i][real_q.max(1)[1].data[i]]  = 1.0
	real_act = Variable(real_act)
	
	adv_q = ANET(adv_states)
	#print(adv_q)
	m = nn.Softmax(dim = 1)
	adv_q = m(adv_q)
	#print(prob_value)
    	
	loss = (real_q - adv_q).pow(2).mean()
	#print(loss)
	adv_optimizer.zero_grad()
	loss.backward()
	adv_optimizer.step()
	
    
losses = []
correct = []
adv_rewards = 0.0
ave_adv_reward = 0.0

for frame_idx in range(1, num_frames + 1):
    #print(env.action_space.n)
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    adv_state = state
    if len(replay_buffer) >= 32:
    	snapshot = env.ale.cloneState()
    	next_state, reward, done, _ = env.step(action)
    	env.ale.restoreState(snapshot)
    	replay_buffer.push(state, action, reward, next_state, done)
    	adv_state = pgd_attack(state)
    	adv_state = adv_state.cpu().data.numpy()
    	replay_buffer.remove_right()
    	
    	#testing area below
    	adv_buffer.append((state, adv_state))
    	if len(adv_buffer) >= 32:
    		anet_loss()
    		adv_action = ANET.act(adv_state,epsilon)
    		snapshot = env.ale.cloneState()
    		next_state2, adv_reward, done2, _2 = env.step(adv_action)
    		env.ale.restoreState(snapshot)
    		adv_rewards += adv_reward
    		
    	#tesing area above
    	
    
    	
    next_state, reward, done, _ = env.step(action) 
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward
   
    if done:
        state = env.reset()
        print('Episode Reward:', episode_reward)
        all_rewards.append(episode_reward)
        print('Adversarial Episode Reward:', adv_rewards)
        ave_award += episode_reward
        ave_adv_reward += adv_rewards
        episode_reward = 0
        adv_rewards = 0.0
        round_cnt += 1
        
        print('frame_idx:')
        print(frame_idx)
        
#print('Aveage Loss:')
#print(sum(losses)/len(losses))
#print(correct)

print('Average Adversarial Award:')
print(ave_adv_reward/round_cnt)

#Delete the '#' symbol in next line, if you want to retrain the ASDN.	
torch.save(ANET.state_dict(),'ANETpoint.pth')



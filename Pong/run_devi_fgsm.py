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

def pgd_attack( images, action, alpha= 5.0) : 
    
    images  = Variable(torch.FloatTensor(np.float32(images)))
    
    for i in range(1):
    	snapshot = env.ale.cloneState()
    	next_state, reward, done, _ = env.step(action)
    	env.ale.restoreState(snapshot)
    	state = images.cpu().data.numpy()
    	replay_buffer.push(state, action, reward, next_state, done)
    	states_loss = compute_loss(batch_size)
    	Jx = states_loss[0]
    	images = images + alpha*Jx
    	replay_buffer.remove_right()
    
    return images
    
def asdn_predict(cur_state, prev_idx):
    state, action, reward, next_state, done = replay_buffer.order_sample(5)
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
state = env.reset()
ave_award = 0.0
round_cnt = 0

mark_buffer = deque(maxlen = 5)
init_cnt = 0
tot_cnt = 0
cor_cnt = 0
tot_neg = 0
pre_neg = 0
sto_cnt = 0

prew_cnt = 0
nr_cnt = 0

devi = 0
devi_con = 0

p_cnt = 0
tp_cnt = 0

totp_cnt = 0

divide_cnt = 4
true_state = state

devi_len = 0

for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    init_cnt += 1
    
    cur_flag = 0

    	
    if init_cnt <= 6:
    	mark_buffer.append(0)
    	
    if init_cnt >= 7:
    	sum_mark = 0.0
    	for i in range(5):
    		sum_mark += mark_buffer[i]
    	if sum_mark >= 5:
    	
    		devi = 1
    		devi_len += 1
    		action = model.act(state,epsilon)
    		ok = 0
    		#tot_cnt += 1
	
    		prob_adv = 0.0
    		eff_cnt = 0
    		mark_buffer.append(0)
    		
    		ori_state = state
    		if random.randrange(100)%divide_cnt == 0:
    			state = pgd_attack(state,action)
    			state = state.cpu().data.numpy()
    		
    			cur_flag = 1
    				
    		action = model.act(ori_state,epsilon)	
    		###
    		next_state, reward, done, _ = env.step(action) 
    		env.render()
    		replay_buffer.push(state, action, reward, next_state, done)
    		state = next_state
    		episode_reward += reward
    		sto_cnt +=  1
    		
    		if done:
        		state = env.reset()
        		print('Episode Reward:', episode_reward)
        		all_rewards.append(episode_reward)
        		ave_award += episode_reward
        		episode_reward = 0
        		round_cnt += 1
        		init_cnt = 0
    		continue
    else:
    	action = model.act(state,epsilon)
    	next_state, reward, done, _ = env.step(action) 
    	env.render()
    	replay_buffer.push(state, action, reward, next_state, done)
    	state = next_state
    	episode_reward += reward
    	continue
    	
    true_state = state
    if len(replay_buffer) >= 50:
    		
    		if random.randrange(100)%divide_cnt == 0:
    			state = pgd_attack(state,action)
    			state = state.cpu().data.numpy()
    		
    			cur_flag = 1
    
    if devi == 1:
    	devi_len += 1
    	
    	if devi_con >= 3:
    		devi_con = 0
    		action = model.act(state,epsilon)
    		ok = 0
    		#tot_cnt += 1
	
    		prob_adv = 0.0
    		eff_cnt = 0
    		mark_buffer.append(0)
    		
    		ori_state = state
    		if random.randrange(100)%divide_cnt == 0:
    			state = pgd_attack(ori_state,action)
    			state = state.cpu().data.numpy()
    		
    			cur_flag = 1
    				
    		action = model.act(ori_state,epsilon)	
    		###
    		next_state, reward, done, _ = env.step(action) 
    		env.render()
    		replay_buffer.push(state, action, reward, next_state, done)
    		state = next_state
    		episode_reward += reward
    		sto_cnt +=  1
    		
    		if done:
        		state = env.reset()
        		print('Episode Reward:', episode_reward)
        		all_rewards.append(episode_reward)
        		ave_award += episode_reward
        		episode_reward = 0
        		round_cnt += 1
        		init_cnt = 0
    		continue
    	else:
    		prob_adv = 0.0
    		eff_cnt = 0
   	 	for i in range(5):
    			if mark_buffer[i] == 0:
    				prob_adv += asdn_predict(state,4-i)
    				eff_cnt += 1
    		
    		prob_adv = prob_adv/eff_cnt
    
    		action = model.act(state,epsilon)
   
    		#if cur_flag == 1:
    			#tot_neg += 1
    		
    		#tot_cnt += 1
    		if prob_adv >= 0.5:
    			action = model.act(state,epsilon)
    			mark_buffer.append(1)
    			devi_con += 1
    			if cur_flag == 1:
    				#cor_cnt += 1
    				#pre_neg += 1
    				pas = 1
    		else:
    			mark_buffer.append(0)
    			for i in range(4):
    				mark_buffer[-i] = 1
    			devi = 0
    			devi_con = 0
    			#if cur_flag == 0:
    				#cor_cnt += 1
    				
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
        		init_cnt = 0
    	
    	continue
    		
    prob_adv = 0.0
    eff_cnt = 0
    for i in range(5):
    	if mark_buffer[i] == 0:
    		prob_adv += asdn_predict(state,4-i)
    		eff_cnt += 1
    prob_adv = prob_adv/eff_cnt
    
    action = model.act(state,epsilon)
    #if you want to know the outcome without detection network, you can simply change 0.5 to 1.5
    if cur_flag == 1:
    	totp_cnt +=1
    	tot_neg += 1
    
    tot_cnt += 1
    if prob_adv >= 0.5:
    	action = model.act(true_state,epsilon)
    	mark_buffer.append(1)
    	p_cnt += 1
    
    	if cur_flag == 1:
    		cor_cnt += 1
    		pre_neg += 1
    		tp_cnt += 1
    else:
    	mark_buffer.append(0)
    	if cur_flag == 0:
    		cor_cnt += 1
    	
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
        init_cnt = 0
        
print('Average Reward:', ave_award/round_cnt)
print('Prediction Acuracy:',cor_cnt/tot_cnt)
print('Precision:',tp_cnt/p_cnt)
print('Recall:',tp_cnt/totp_cnt)
#print('Acc:',pre_neg/tot_neg)
print('Stop Count',sto_cnt)
print('Deviation Length:',devi_len)
#print('Mapping:',nr_cnt/prew_cnt)
#print('prev false:',prew_cnt)
print('tot_cnt',tot_cnt)
	
clear_output(True)
plt.figure(figsize=(20,5))
plt.subplot(131)
plt.title('Reward')
plt.plot(all_rewards)
plt.show()
plt.close()



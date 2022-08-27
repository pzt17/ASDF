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
#%matplotlib inline

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

from collections import deque
from wrappers import make_atari, wrap_deepmind, wrap_pytorch



    


class ASDN(nn.Module):
    def __init__(self, input_shape):
        super(ASDN, self).__init__()
        
        self.input_shape = input_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size= 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
  
        self.fc = nn.Sequential(
        	nn.Linear(self.feature_size(), 512),
            	nn.ReLU(),
            	nn.Linear(512, 10),
            	nn.ReLU(),
            	nn.Linear(10,1)
        )
        
        
        
    def forward(self, x):
    	x = self.features(x)
    	x = x.view(x.size(0), -1)
    	x = self.fc(x)
    	return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data[0]
        return action




















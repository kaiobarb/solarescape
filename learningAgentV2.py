import pygame
from ple import PLE
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from solarescape_env import SolarescapeEnv

class Network(nn.Module):
    ## Let's start with a simple network:
    # Input layer the state (the state) is going to be len 4
    # Output layer (the actions) is going to be len 4
    def __init__(self, input_size, action_size):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = action_size
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implement replay memory so as to not forget things
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implement Deep Q Learning
class DQN():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        # probs = F.softmax(self.model(Variable(state, volatile = True))*10) # T(Temperature)=10
        # action = probs.multinomial(1)
        # return action.data[0,0]

        # state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(Variable(state, volatile = True)) 
        action = probs.data.max(1)[1].cpu()
        print(action)
        return action

        # curiosity = random.random()
        # if curiosity > self.gamma:
        #     state = torch.from_numpy(state).float().unsqueeze(0)
        #     probs = dqn(Variable(state, volatile=True))
        #     return probs.data.max(1)[1].cpu()
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
        
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

game = SolarescapeEnv(width=856, height=856, dt=1)
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
p = PLE(game, fps=30, frame_skip = 3, num_steps = 1,
        force_fps = False, display_screen=False)

# possible_actions = {0:None}
# possible_actions.update({i:a for i,a in enumerate([K_w, K_a, K_s, K_d], start=1)})

class LearningAgent():
    def __init__(self, actions):
        self.actions = actions
        self.brain = DQN(4, 4, 0.9)

    def pickAction(self, action):
        return p.act(action)


if __name__ == '__main__':
    print(game.getActions())
    reward = 0
    steps = 1000
    la = LearningAgent(list(game.getActions()))
    """
    #where is the documentation for extract_image? I imagine it comes from utils
    snapshot = extract_image(p.getScreenRGB(), (80,80), thresh=thresh)
    #what is this
    stack_snaps = np.stack((snapshot, snapshot, snapshot, snapshot), axis=0)
    """
    i = 0
    while p.game_over() == False:
        # We want to train
        i+=1
        print(i)
        state = list(p.getGameState().values())
        reward = p.score()
        action = la.brain.update(reward, state)
        print(action)
        la.pickAction(action)
        
    score = p.score()
    p.reset_game()
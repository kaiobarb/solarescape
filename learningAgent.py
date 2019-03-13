import time

#import cv2 #what is this?
#import os #what is this?

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple

import numpy as np

from solarescape_env import *
from ple import PLE


#def getState(self):
#    state = {
#        "agent_y": self.agent.pos.y,
#        "agent_x": self.agent.pos.x,
#        "agent_velocity_x": self.agent.vel.x,
#        "agent_velocity_y": self.agent.vel.y
#    }
#    return state

def extract_image(image_data, size, thresh=True):
    #resize image and change to grayscale
    snapshot = cv2.cvtColor(cv2.resize(image_data, size), cv2.COLOR_BGR2GRAY)

    #threshold function applies fixed-level thresholding (?) to a single-channel array
    #threshold(array, threshold, maximum value, thresholding type
    if thresh:
        _, snapshot = cv2.threshold(snapshot, 100, 255, cv2.THRESH_BINARY)
    return snapshot

def play(f_action, state, select_action, perform_action, possible_actions, model, kwargs):
    hx = kwargs.get("hx", None)
    cx = kwargs.get("hx", None)
    isT = kwargs.get("isTrain", False)
    action, hx, cx, info_dict = select_action(state, hx, cx, model, isT)
    reward = perform_action(f_action, possible_actions, action)
    return reward, action, hx, cx, info_dict

def train_and_play(f_action, state, select_action, perform_action, possible_actions, optimize, model, kwargs):
    reward, action, hx, cx, info_dict = play(f_action, state, select_action, perform_action, possible_actions, model, kwargs)
    optimize()
    return reward, action, hx, cx, info_dict

def ensure_shared_grads(model, shared_model):  #I don't think  I even need this one
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

################################################################################################
####################### Here is DQN stuff, taken from darleybarreto's github ###################
################################################################################################

opt = {
    "RMSprop": torch.optim.RMSprop
}

class DQN(nn.Module):
    def __init__(self, actions, shape, fc): #what is fc exactly??
        super(DQN, self).__init__()

        in_, out_, kernel, stride = (list(zip(*shape))[i] for i in range(len(shape) + 1))

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_[0], out_[0], kernel_size=kernel[0], stride=stride[0]),
            nn.BatchNorm2d(out_[0]),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_[1], out_[1], kernel_size=kernel[1], stride=stride[1]),
            nn.BatchNorm2d(out_[1]),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_[2], out_[2], kernel_size=kernel[2], stride=stride[2]),
            nn.BatchNorm2d(out_[2]),
            nn.ReLU())

        self.fc1 = nn.Linear(fc[0], fc[1])
        self.fc2 = nn.Linear(fc[1], actions)

    def forward(self, next): #I believe that next is a good name, 'x' used in GitHub code
        next = self.layer1(next)
        next = self.layer2(next)
        next = self.layer3(next)

        #next.size(0) gets the 0 component of its size.
        #next.vew() basically reshapes the tensor, and the -1 tells Pytorch
        #to find the best number of columns to fit our data
        next = next.view(next.size(0), -1)
        next = F.relu(self.fc1(next))
        next = F.relu(self.fc2(next))

        return next

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    class ReplayMemory(object):

        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def push(self, *args):
            """Saves a transition"""
            if len(self.memory) < self.capacity:  #if capacity is not full yet?
                self.memory.append(None)          #why append None?
            [stack_snaps, action, st, reward] = args

            stack_snaps = torch.from_numpy(stack_snaps.astype(float).reshape((1, *stack_snaps.shape)))
            st = torch.from_numpy(st.astype(float).reshape(1, *st.shape))  #wtf is st???

            self.memory[self.position] = Transition(*[stack_snaps,\
                                                      action,\
                                                      st,\
                                                      torch.Tensor([reward])])
            #does above imply that stack_snaps is current state, while st is next state??? seems to...
            self.position = (self.position +1) % self.capacity
            #in above, is position a time step, not space? and is the modulus a way of randomizing??

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory) #this seems unnecessary, is this redefining inherited len call?

    def create_model(actions, shape, fully_connected, learning_rate=1e-2, opt_='RMSprop', **kwargs):
        BATCH_SIZE = kwargs.get("BATCH_SIZE", 64)
        GAMMA      = kwargs.get("GAMMA", 0.999)
        EPS_START  = kwargs.get("EPS_START", 0.9)
        EPS_END    = kwargs.get("EPS_END", 0.05)
        EPS_DECAY  = kwargs.get("EPS_DECAY", 200)
        path       = kwargs.get("path", None)
        memory     = kwargs.get("memory", ReplayMemory(10000))

        dqn = DQN(actions, shape, fully_connected)  ##OHHH that's what fc stands for?? Why?

        if path:
            dqn.load_state_dict(torch.load(path))

        optimizer = opt[opt_](dqn.parameters(), lr = learning_rate)
        steps_done = 0
        last_sync = 0

        def select_greedy_action(state, hx, cx, model, isTrain):
            nonlocal steps_done #what is nonlocal? here says: statement seems to have no effect

            sample = random.random()

            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            if sample > eps_threshold:
                state = torch.from_numpy(state).float().unsqueeze(0) #what
                probs = dqn(Variable(state, volatile = True))

                return probs.data.max(1)[1].cpu(), None, None, None #this gets index of max log-probability, apparently

            else:
                return torch.LongTensor([[random.randrang(actions)]]), None, None, None #why double []s?

        def perform_action(f_action, possible_actions, action):
            #darleybarreto's code just says 'reward' here for some reason
            return f_action(possible_actions[action[0][0]]) #I deleted possible_actions...

        def optimize():
            #Perform experience replay and train the network - darleybarreto
            nonlocal last_sync #again with this nonlocal...

            if len(memory)<BATCH_SIZE: #if we've overshot bounds of batch? seems like this is saying if we haven't tho...
                return

            transitions = memory.sample(BATCH_SIZE)
            #Use the replay buffer to sample a batch of transitions

            batch = Transition(*zip(*transitions)) #what does zip do?

            state_batch = Variable(torch.cat(batch.state)).float()
            #batch.state is a tuple of states
            action_batch = Variable(torch.cat(batch.action)).long()
            #batch.action is a tuple of actions
            reward_batch = Variable(torch.cat(batch.reward)).float()
            #batch.reward is a tuple of rewards

            non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
            #wot is this above

            non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile = True).float()

            state_action_values = dqn(state_batch).gather(1, action_batch)

            next_state_values = Variable(torch.zeros(BATCH_SIZE))

            next_state_values[non_final_mask] = dqn(non_final_next_states).max(1)[0]

            next_states_values.volatile = False

            expected_state_actions_values = (next_state_values * GAMMA) + reward_batch

            loss = F.smooth_l1_loss(state_action_values, expected_state_actions_values)

            optimizer.zero_grad()

            loss.backward()

            for param in dqn.parameters():
                param.grad.data.clamp_(-1,1)

            optimizer.step()

        def save_model(path):
                if path:
                    torch.save(dqn.state_dict(), path)

        def push_to_memory(*args):
            memory.push(*args)

        return push_to_memory, select_greedy_action, perform_action, optimize, save_model


################################################################################################
################################# End of DQN portion ###########################################
################################################################################################

game = SolarescapeEnv(width=856, height=856, dt=1)
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
p = PLE(game, fps=30, frame_skip = 3, num_steps = 1,
        force_fps = False, display_screen=False)

possible_actions = {0:None}
possible_actions.update({i:a for i,a in enumerate([K_w, K_a, K_s, K_d], start=1)})

class LearningAgent():
    def __init__(self, actions):
        self.actions = actions

    def pickAction(action):
        return p.act(action)

def p_action(action):
    return p.act(action)

if __name__ == '__main__':
    #p.init() #do I even need this? Kaio didn't seem to be using it for naive agent
    print(game.getActions())
    reward = 0
    steps = 1000
    la = LearningAgent(list(game.getActions()))
    #where is the documentation for extract_image? I imagine it comes from utils
    snapshot = extract_image(p.getScreenRGB(), (80,80), thresh=thresh)
    #what is this
    stack_snaps = np.stack((snapshot, snapshot, snapshot, snapshot), axis=0)

    while p.game_over() == False:
        snapshot = extract_image(p.getScreenRGB(), (80, 80), thresh=thresh)
        snapshot = np.reshape(snapshot, (1, 80, 80))
        st = np.append(stack_snaps[1:4, :, :], snapshot, axis=0) #what does st stand for?

        if train:
            reward, action, _, _, _ = train_and_play(p_action, st, select_action, perform_action, possible_actions, optimize, None, {})
            push_to_memory(stack_snaps, action, st, reward)
        else:
            play(p_action, st, select_action, perform_action, possible_actions, None, {})

        stack_snaps = st

    score = p.score()
    p.reset_game()
    if train:
        save_model(save_path)

    #return score #how to do this when this isn't technically a function?
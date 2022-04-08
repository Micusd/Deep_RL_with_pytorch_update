import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
import os
import pickle
import time
from collections import deque

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from wrappers import wrap, wrap_cover, SubprocVecEnv
from runner import Runner


'''A2C Settings'''
TRAJ_LEN = 100
ENT_COEF = 1e-2
LAMBDA = 0.95

'''Environment Settings'''
# sequential images to define state
STATE_LEN = 4
# openai gym env name
ENV_NAME = 'PongNoFrameskip-v4'
# number of environments for A2C
N_ENVS = 4
# Total simulation step
N_STEP = 10**7
# gamma for MDP
GAMMA = 0.99
# visualize for agent playing
RENDERING = False

'''Training settings'''
# check GPU usage
USE_GPU = torch.cuda.is_available()
print('USE GPU: '+str(USE_GPU))
# learning rage
LR = 1e-4
# clip gradient
MAX_GRAD_NORM = 0.1
# log optimization
LOG_OPT = False

'''Save&Load Settings'''
# log frequency
LOG_FREQ = 100
# check save/load
SAVE = True
LOAD = False
# paths for predction net, target net, result log
NET_PATH = './data/model/a2c_net.pkl'

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # architecture def
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(STATE_LEN, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(7 * 7 * 64, 256)
        # actor
        self.actor = nn.Linear(256, N_ACTIONS)
        # critic
        self.critic = nn.Linear(256, 1)

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x is a tensor of (m, 4, 84, 84)
        x = self.feature_extraction(x / 255.0)
        # x.size(0) : mini-batch size
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        # use log_softmax for numerical stability
        action_log_prob = F.log_softmax(self.actor(x), dim=1)
        state_value = self.critic(x)

        return action_log_prob, state_value

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class A2C:
    def __init__(self):
        self.net = ConvNet()
        # use gpu
        if USE_GPU:
            self.net = self.net.cuda()

        # simulator step conter
        self.memory_counter = 0

        # define optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)

    def save_model(self):
        self.net.cpu()
        self.net.save(NET_PATH)
        if USE_GPU:
            self.net.cuda()

    def load_model(self):
        self.net.cpu()
        self.net.load(NET_PATH)
        if USE_GPU:
            self.net.cuda()

    def choose_action(self, x):
        self.memory_counter += 1
        # Assume that x is a np.array of shape (nenvs, 4, 84, 84)
        x = torch.FloatTensor(x)
        if USE_GPU:
            x = x.cuda()
        # get action log probs and state values
        action_log_probs, state_values = self.net(x)  # (nenvs, N_ACTIONS)
        probs = F.softmax(action_log_probs, dim=1).data.cpu().numpy()
        probs = (probs + 1e-8) / np.sum((probs + 1e-8), axis=1, keepdims=True)
        # sample actions
        actions = np.array([np.random.choice(N_ACTIONS, p=probs[i]) for i in range(len(probs))])
        # convert tensor to np.array
        action_log_probs, state_values = action_log_probs.data.cpu().numpy(), state_values.squeeze(1).data.cpu().numpy()
        # calc selected logprob
        selected_log_probs = np.array([action_log_probs[i][actions[i]] for i in range(len(probs))])
        return actions, state_values, selected_log_probs

    def learn(self, obs, returns, masks, actions, values, selected_log_probs):

        # calculate the advantages
        advs = returns - values

        # np.array -> torch.Tensor
        obs = torch.FloatTensor(obs)  # (m, 4, 84, 84)
        returns = torch.FloatTensor(returns)  # (m)
        advs = torch.FloatTensor(advs)  # (m)
        actions = torch.LongTensor(actions)  # (m)
        selected_log_probs = torch.FloatTensor(selected_log_probs)  # (m)
        values = torch.FloatTensor(values)  # (m)
        if USE_GPU:
            obs = obs.cuda()
            returns = returns.cuda()
            advs = advs.cuda()
            actions = actions.cuda()
            selected_log_probs = selected_log_probs.cuda()
            values = values.cuda()

        # get action log probs and state values
        action_log_probs, state_values = self.net(obs)
        # (m, N_ACTIONS), (m, 1)

        # calc probs
        probs = F.softmax(action_log_probs, dim=1)
        # (m, N_ACTIONS)

        # calc entropy loss
        ent_loss = ENT_COEF * ((action_log_probs * probs).sum(dim=1)).mean()
        # (1)

        # calc log probs
        cur_log_probs = action_log_probs.gather(1, actions.unsqueeze(1))
        # cur : (m, 1)

        # actor loss
        actor_loss = torch.mean(- cur_log_probs.squeeze(1) * advs)  # (1)
        # critic loss
        critic_loss = F.smooth_l1_loss(state_values.squeeze(1), returns)  # (1)

        loss = actor_loss + critic_loss + ent_loss  # (1)

        actor_loss, critic_loss, ent_loss, total_loss = actor_loss.data.cpu().numpy(), \
                                                        critic_loss.data.cpu().numpy(), ent_loss.data.cpu().numpy(), loss.data.cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        return round(float(actor_loss), 4), round(float(critic_loss), 4), \
               round(float(ent_loss), 4), round(float(total_loss), 4)


if __name__ == '__main__':
    # define gym
    env = SubprocVecEnv([wrap_cover(ENV_NAME) for i in range(N_ENVS)])
    # check gym setting
    N_ACTIONS = env.action_space.n;
    print('N_ACTIONS : ', N_ACTIONS)  # 6
    N_STATES = env.observation_space.shape;
    print('N_STATES : ', N_STATES)  # (4, 84, 84)

    a2c = A2C()
    runner = Runner(env=env, model=a2c, nsteps=TRAJ_LEN, gamma=GAMMA, lam=LAMBDA)

    # model load with check
    if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
        a2c.load_model()
        pkl_file = open(RESULT_PATH, 'rb')
        result = pickle.load(pkl_file)
        pkl_file.close()
        print('Load complete!')
    else:
        result = []
        print('Initialize results!')

    print('Collecting experience...')

    # episode step for accumulate reward
    epinfobuf = deque(maxlen=100)
    # in A2C, we iterate over optimization step
    nbatch = N_ENVS * TRAJ_LEN
    nupdates = N_STEP // nbatch
    # check learning time
    start_time = time.time()

    for update in range(1, nupdates + 1):
        # get minibatch
        obs, returns, masks, actions, values, neglogpacs, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        slices = (obs, returns, masks, actions, values, neglogpacs)
        actor_loss, critic_loss, ent_loss, total_loss = a2c.learn(*slices)

        # print opt log
        if LOG_OPT and update % LOG_FREQ == 0:
            print('Iter ', _,
                  'actor loss : ', round(actor_loss, 3),
                  'critic loss : ', round(critic_loss, 3),
                  'ent loss : ', round(ent_loss, 3),
                  'total loss : ', round(total_loss, 3))

        if update % LOG_FREQ == 0:
            # print log and save
            # check time interval
            time_interval = round(time.time() - start_time, 2)
            # calc mean return
            mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)
            result.append(mean_100_ep_return)
            # print epi log
            print('N update: ', update,
                  '| Mean ep 100 return: ', mean_100_ep_return,
                  '/Used Time:', time_interval,
                  '/Used Step:', a2c.memory_counter * N_ENVS)
            # save model
            if SAVE:
                a2c.save_model()

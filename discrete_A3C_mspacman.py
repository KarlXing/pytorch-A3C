"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import v_wrap, push_and_pull, record, set_init
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
import glob
import datetime
import pickle
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.95
MAX_EP = 10000
DIR = "/home/jinwei/Documents/Git/pytorch-A3C/Result/A3C/"
# DIR = "/Users/karl/Documents/Git/pytorch-A3C/Result/A3C"
DIE_PENALTY = -20
MAX_WORKERS = 16

env = gym.make('MsPacman-v0')
N_A = env.action_space.n


class Net(nn.Module):
    def __init__(self, a_dim):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.f1_a = nn.Linear(22*16*64, 512)
        self.f2_a = nn.Linear(512, a_dim)
        self.f1_v = nn.Linear(22*16*64, 512)
        self.f2_v = nn.Linear(512, 1)

        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        a_x = F.tanh(self.f1_a(x))
        a_x = self.f2_a(a_x)
        v_x = F.relu(self.f1_v(x))
        v_x = self.f2_v(v_x)

        return a_x, v_x

    def choose_action(self, s):
        self.eval()
        logits, v = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2).mean()
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        dist_entropy = m.entropy().mean()
        a_loss = -(m.log_prob(a) * td.detach().squeeze()).mean()
        total_loss = 0.5*c_loss + a_loss - 0.01*dist_entropy 
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_A)           # local network
        self.env = gym.make('MsPacman-v0').unwrapped
        self.env.reset()
        self.ep_r = 0
        _, _, _, info = self.env.step(0)
        self.lives_sum = info['ale.lives']

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = np.transpose(self.env.reset(),(2,0,1))/255.0
            lives = self.lives_sum
            buffer_s, buffer_a, buffer_r = [], [], []
            self.ep_r = 0.
            actions = []
            while True:
                total_step += 1
                self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                # a = np.random.randint(low = 0, high = 8)
                actions.append(str(a))
                s_, r, done, info = self.env.step(a)
                s_ = np.transpose(s_, (2,0,1))/255.0
                livesLeft = info['ale.lives']          # punish everytime the agent loses life
                if livesLeft != lives:
                    r = DIE_PENALTY
                    lives = livesLeft
                self.ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    # if self.name == 'w0':
                    #     self.env.render()
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.ep_r, self.res_queue, self.name, self.lives_sum, DIE_PENALTY)
                        break
                s = s_
        self.res_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    gnet = Net(N_A)        # global network
    model_dir = DIR+"model/*"      # load recent trained global network
    model_files = sorted(glob.iglob(model_dir), key=os.path.getctime, reverse=True)
    if len(model_files) != 0:
        gnet.load_state_dict(torch.load(model_files[0]))
        print("load global network from ", model_files[0])
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0005)      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    start_time = str(datetime.datetime.now())        # to track the time

    # parallel training
    num_workers = min(MAX_WORKERS, mp.cpu_count())
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(num_workers)]
    [w.start() for w in workers]
    
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    end_time = str(datetime.datetime.now())         # end time
    torch.save(gnet.state_dict(), DIR+"model/model_"+start_time+"-"+end_time)
    with open(DIR+"record/record_"+start_time+"-"+end_time,"wb") as f:
        pickle.dump(res, f)
    [w.join() for w in workers]
    print("complete")

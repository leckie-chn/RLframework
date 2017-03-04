import math
import random

import numpy as np


class Environment(object):

    def __init__(self):
        self.capacity = np.asarray([20.0, 30.0, 50.0])
        self.bufsize = np.asarray([10.0, 10.0, 15.0])
        self.buffer = np.zeros_like(self.bufsize)
        self.totcap = np.sum(self.capacity)
        self.t = 0.0
        self.T = 10.0
        self.flow_in = 0.0
        self.state = None

    def get_state(self):
        self.flow_in = math.sin(t / (math.pi * 2 * self.T)) * self.totcap * 1.05
        self.t += 1.0
        self.state = np.concatenate(([self.flow_in], self.buffer))
        return self.state

    def take_action(self, action):
        flows = self.flow_in * action + self.buffer
        cur_flow = np.maximum(flows, self.capacity)
        self.buffer = np.maximum(np.zeros_like(flows), flows - self.capacity)
        drop_flow = np.sum(self.buffer - self.bufsize)
        self.buffer = np.minimum(self.buffer, self.bufsize)
        load_factor = cur_flow / self.capacity
        reward = 2.0 - np.max(load_factor) - math.exp(drop_flow / self.flow_in * 100)
        next_state = self.get_state()
        isTerminal = 1 if self.t > self.T * 2.5 else 0
        return next_state, reward, isTerminal

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
        self.flow_in = 0.1
        self.state = None

    def get_state(self):
        self.flow_in = math.sin(self.t / (math.pi * 2 * self.T) + 0.1) * self.totcap * 1.05
        self.t += 1.0
        self.state = np.concatenate(([self.flow_in], self.buffer))
        return self.state

    def take_action(self, action):
        flows = self.flow_in * action + self.buffer
        cur_flow = np.minimum(flows, self.capacity)
        self.buffer = flows - cur_flow
        drop_flow = np.maximum(np.sum(self.buffer - self.bufsize), np.zeros_like(self.buffer))
        self.buffer = np.minimum(self.buffer, self.bufsize)
        load_factor = cur_flow / self.capacity
        reward = 1.0 - np.max(load_factor) - math.pow(0 if self.flow_in == 0 else drop_flow / self.flow_in * 100, 4.0)
        next_state = self.get_state()
        isTerminal = True if self.t > self.T * 5 * math.pi else False
        return next_state, reward, isTerminal

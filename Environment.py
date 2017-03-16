import math
import random

import numpy as np

def get_state_size():
    return 4

def get_action_size():
    return 3

class Environment(object):
    def __init__(self, top_flow_rate = 1.0):
        self.capacity = np.asarray([30.0, 40.0, 30.0])
        self.bufsize = np.asarray([10.0, 15.0, 20.0])
        self.buffer = np.zeros_like(self.bufsize)
        self.topflow = np.sum(self.capacity) * top_flow_rate
        self.t = 0.0
        self.T = 50.0
        self.flow_in = 0.0
        self.state = None

    def get_state(self):
        self.flow_in = (math.sin(math.pi * 2.0 * self.t / self.T + 0.1) + 1.0)* self.topflow * 0.5
        self.t += 1.0
        self.state = np.concatenate(([self.flow_in], self.buffer))
        return self.state

    def take_action(self, action):
        flows = self.flow_in * action + self.buffer
        cur_flow = np.minimum(flows, self.capacity)
        self.buffer = flows - cur_flow
        drop_flow = np.sum(np.maximum(self.buffer - self.bufsize, np.zeros_like(self.buffer)))
        self.buffer = np.minimum(self.buffer, self.bufsize)
        load_factor = cur_flow / self.capacity
        drop_penalty = 0.0 if self.flow_in == 0 else drop_flow / self.flow_in * 100
        reward = 1.0 - np.max(load_factor) - drop_penalty
        next_state = self.get_state()
        isTerminal = True if self.t > self.T * 2.0 else False
        return next_state, reward, isTerminal

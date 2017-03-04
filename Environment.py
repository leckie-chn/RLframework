import math
import random

import numpy as np


class Environment(object):

    def __init__(self):
        self.capacity = np.asarray([20.0, 30.0, 50.0])
	self.totcap = np.sum(self.capacity)
        self.state = None

    def get_state(self):
        self.state = random.uniform(0, self.totcap)
        return self.state

    def take_action(self, action):
        load_factor = self.state * action / self.capacity
        reward = 1 - (np.max(load_factor) - np.min(load_factor))
        return reward

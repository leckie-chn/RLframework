
from collections import deque
import random



class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
    
    def get_batch(self, batch_size):
        # random.seed(1234)
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
            # print "self.num_experiences < batch_size"
            # return None
        else:
            return random.sample(self.buffer, batch_size)
    
    def add(self, state, action, reward, next_state, terminal):
        experience = (state, action, reward, next_state, terminal)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
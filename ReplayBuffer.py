
from collections import deque
import random



class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences_1 = 0
        self.num_experiences_2 = 0
        self.buffer_1 = deque()
        self.buffer_2 = deque()
        # random.seed(0)
    
    def get_batch(self, batch_size):
        # random.seed(1234)
        if self.num_experiences_2==0:
            print "self.num_experiences_2==0"
            return None
        elif self.num_experiences_2 < batch_size:
            batch = random.sample(self.buffer_2, self.num_experiences_2)
        else:
            batch = random.sample(self.buffer_2, batch_size)
        if self.num_experiences_1 < len(batch):
            batch.extend( random.sample(self.buffer_1, self.num_experiences_1) )
        else:
            batch.extend( random.sample(self.buffer_1, len(batch)) )
        random.shuffle(batch)
        return batch
    
    def add(self, state, action, reward, terminal):
        experience = [state, action, reward]
        if terminal==1:
            if self.num_experiences_1 < self.buffer_size:
                self.buffer_1.append(experience)
                self.num_experiences_1 += 1
            else:
                self.buffer_1.popleft()
                self.buffer_1.append(experience)
        elif terminal==2:
            if self.num_experiences_2 < self.buffer_size:
                self.buffer_2.append(experience)
                self.num_experiences_2 += 1
            else:
                self.buffer_2.popleft()
                self.buffer_2.append(experience)
    
    def get_batch_experience(self, batch_size):
        random.seed(0)
        if self.num_experiences_1 < batch_size:
            return random.sample(self.buffer_1, self.num_experiences_1)
        else:
            return random.sample(self.buffer_1, batch_size)
    
    def add_experience(self, experience):
        if self.num_experiences_1 < self.buffer_size:
            self.buffer_1.append(experience)
            self.num_experiences_1 += 1
        else:
            self.buffer_1.popleft()
            self.buffer_1.append(experience)



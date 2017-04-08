from collections import deque
import numpy as np
import random


class ReplayBuffer(object):
    """Abstract Replay Buffer Class"""

    def __init__(self):
        pass

    def add(self, state, action, reward, next_state):
        pass

    def get_batch(self, batch_size):
        pass

    def tick(self):
        pass


class TimeoutReplayBuffer(ReplayBuffer):
    """Replay Buffer with round timeout"""

    def __init__(self, round_timeout, state_dim, action_dim):
        super(TimeoutReplayBuffer, self).__init__()
        self.round_timeout = round_timeout
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pool = deque()
        self.dcount = 0  # number of data buffered
        # self.sample_weight = 1.0 / np.arange(1, self.round_timeout + 1)
        # self.sample_weight /= np.sum(self.sample_weight)
        self.pool.append({
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
        })

    def add(self, state, action, reward, next_state):
        current_batch = self.pool[0]
        current_batch['states'].append(state)
        current_batch['actions'].append(action)
        current_batch['rewards'].append(reward)
        current_batch['next_states'].append(next_state)

    def get_batch(self, batch_size):
        state_batch = np.empty((batch_size, self.state_dim))
        action_batch = np.empty((batch_size, self.action_dim))
        reward_batch = np.empty((batch_size))
        next_state_batch = np.empty((batch_size, self.state_dim))
        sample_weight = 1.0 / np.arange(1, len(self.pool) + 1)
        sample_weight /= np.sum(sample_weight)
        sample_num = np.random.multinomial(batch_size, sample_weight)
        item_count = 0
        for round_index, round_batch in enumerate(self.pool):
            for i in xrange(sample_num[round_index]):
                index = random.choice(range(len(round_batch['states'])))
                state_batch[item_count, :] = round_batch['states'][index]
                action_batch[item_count, :] = round_batch['actions'][index]
                reward_batch[item_count] = round_batch['rewards'][index]
                next_state_batch[item_count, :] = round_batch['next_states'][index]
                item_count += 1
        return state_batch, action_batch, reward_batch, next_state_batch

    def tick(self):
        """delete out-of-round data & init batch for new round data"""
        self.pool.appendleft({
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
        })
        if len(self.pool) > self.round_timeout:
            self.pool.pop()


class PriorityReplayBuffer(ReplayBuffer):
    """Replay Buffer with priority sampling"""

    def __init__(self):
        super(PriorityReplayBuffer, self).__init__()
        # TODO
        pass


import numpy as np
import math

def CreateEnvironment(opt):
    if opt == 'single-central':
        return 4, 3, SingleCentral()
    elif opt == 'double-central':
        return 9, 4, DoubleCentral()
    else:
        return None

class SingleCentral(object):
    def __init__(self, tm_count = 628):
        self.capacity = [20.0, 30.0, 50.0]
        self.point_count = tm_count
        self.flow = 30 + np.sin(np.linspace(0, 2*math.pi, num=self.point_count)) * 10
        self.cur_flow = np.zeros_like(self.capacity)
        self.isTerminal = False
        self.tm_step = np.random.randint(self.point_count)
        self.flow_in = max(self.flow[self.tm_step] + np.random.normal(0.0, 5.0), 0.0)

    def get_state(self):
        return None if self.isTerminal is True else \
            np.concatenate((np.array([self.flow_in]), self.cur_flow / self.capacity))

    def correct_action(self):
        return self.capacity / np.sum(self.capacity)

    def take_action(self, action):
        """
        :type action: np.ndarray
        :return:
        """
        # if action.shape != (3):
        #    raise ValueError("action should be numpy array of shape (3), not {}".format(action.shape))
        new_flow = action * self.flow_in
        self.cur_flow += new_flow
        usage_rate = self.cur_flow / self.capacity
        self.cur_flow = np.maximum(np.zeros_like(self.capacity), self.cur_flow - self.capacity)
        self.tm_step += 1
        self.flow_in = max(self.flow[self.tm_step] + np.random.normal(0.0, 5.0), 0.0)
        self.isTerminal = self.tm_step >= self.point_count
        reward = 1 - np.max(usage_rate)
        return reward


class DoubleCentral(object):
    def __init__(self):
        self.capacity = np.array([100.0, 200.0, 200.0, 100.0, 200.0, 200.0, 200.0])
        self.cur_flow = np.zeros_like(self.capacity)
        self.usgrate = np.zeros_like(self.capacity)
        self.point_count = 628
        self.flow_a = 120 + np.sin( np.linspace(0, 2*math.pi, num=self.point_count) )*20
        self.flow_b = 70 + np.cos( np.linspace(0, 5*math.pi, num=self.point_count) + 1.0)*10
        self.max_step_count = 20
        self.tm_step = np.random.randint(self.point_count)
        self.isTerminal = False

    def get_state(self):
        # np.random.seed(0)
        # self.data_step_count = 0
        return np.concatenate((np.array([self.flow_a[self.tm_step], self.flow_b[self.tm_step]]),self.usgrate))

    def take_action(self, action):
        print "cur_state ==>", list(self.cur_state)
        print "action ==>", action
        action = np.array(action)
        # self.data_step_count += 1

        new_flow = np.array([
            self.flow_a[self.tm_step] * action[0],
            self.flow_b[self.tm_step] * action[2],
            self.flow_a[self.tm_step] * action[1],
            0.0,
            self.flow_b[self.tm_step] * action[3],
            0.0,
            0.0
        ])

        new_flow[3] = new_flow[0] + new_flow[1]
        new_flow[5] = new_flow[1]
        new_flow[6] = new_flow[0]


        self.cur_flow += new_flow
        self.usgrate = self.cur_flow / self.capacity
        self.cur_flow = np.maximum(np.zeros_like(self.cur_flow), self.cur_flow - self.capacity)
        reward = 1.0 - np.max(self.usgrate) - np.sum(self.cur_flow / self.capacity)
        self.isTerminal = self.tm_step >= self.point_count # TODO terminate when too much overflow
        self.tm_step += 1
        return reward, self.get_state()

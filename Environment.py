import numpy as np
import math

def SetEnvironment(opt, capaticy, tm_count):
    global _opt
    global _capacity
    global _tm_count
    _opt, _capacity, _tm_count = opt, capaticy, tm_count

def CreateEnvironment():
    global _opt
    global _capacity
    global _tm_count
    if _opt == 'single-central':
        return 4, 3, SingleCentral(_capacity, _tm_count)
    elif _opt == 'double-central':
        return 9, 4, DoubleCentral(_capacity, _tm_count)
    else:
        return None


class Environment(object):
    def __init__(self, capacity, tm_count):
        self.capacity = capacity  # type: list[float]
        self.cur_flow = np.zeros_like(self.capacity, dtype=np.float)
        self.isTerminal = False
        self.tot_stamp = tm_count
        self.tm_step = np.random.randint(self.tot_stamp)
        self.flows = self.create_flow()
        self.flow_dim = self.flows.shape[1]
        self.flow_in = np.maximum(self.flows[self.tm_step, :] + np.random.normal(0.0, 5.0, (self.flow_dim)),
                                  np.zeros((self.flow_dim), dtype=np.float))

    def create_flow(self):
        """create flow sample numpy array of shape (flow_dim, tot_stamp) """
        pass

    def get_state(self):
        return None if self.isTerminal is True else \
            np.concatenate((self.flow_in, self.cur_flow / self.capacity))

    def take_flow(self, action):
        """
        change self.current flow according to action
        :param action:
        :return:
        """
        pass

    def take_action(self, action):
        self.take_flow(action)
        usage_rate = self.cur_flow / self.capacity
        self.tm_step += 1
        if self.tm_step < self.tot_stamp:
            self.flow_in = np.maximum(self.flows[self.tm_step, :] + np.random.normal(0.0, 5.0, (self.flow_dim)),
                                    np.zeros((self.flow_dim)))
        else:
            self.isTerminal = True
        reward = 1 - np.max(usage_rate)
        return reward

    def correct_action(self):
        pass


class SingleCentral(Environment):
    def __init__(self, capacity, tm_count):
        if len(capacity) != 3:
            raise ValueError("There are 3 Edges in the Network, not {}".format(len(capacity)))
        super(SingleCentral, self).__init__(capacity, tm_count)

    def create_flow(self):
        return 30 * np.sin(np.linspace(0, 2 * math.pi, num=self.tot_stamp))[:, np.newaxis] + 20

    def correct_action(self):
        return self.capacity / np.sum(self.capacity)

    def take_flow(self, action):
        self.cur_flow += action * self.flow_in


class DoubleCentral(Environment):
    def __init__(self, capacity, tm_count):
        if len(capacity) != 7:
            raise ValueError("")
        super(DoubleCentral, self).__init__(capacity, tm_count)

    def create_flow(self):
        return np.transpose(np.array([
            120 + np.sin(np.linspace(0, 2 * math.pi, num=self.tot_stamp)) * 20,
            70 + np.sin(np.linspace(0, 5 * math.pi, num=self.tot_stamp) + 1.0) * 10
        ]))

    def correct_action(self):
        pass

    def take_flow(self, action):
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

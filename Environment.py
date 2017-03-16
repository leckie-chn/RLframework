
import numpy as np
import math


class Environment(object):
    def __init__(self):
        self.link_capacity_1 = 100.0; self.link_capacity_2 = 200.0; self.link_capacity_3 = 200.0;
        self.link_capacity_4 = 100.0; self.link_capacity_5 = 200.0; self.link_capacity_6 = 200.0; self.link_capacity_7 = 200.0;
        self.link_capacity = np.array([self.link_capacity_1, self.link_capacity_2, self.link_capacity_3, 
                                     self.link_capacity_4, self.link_capacity_5, self.link_capacity_6, self.link_capacity_7])
        self.point_count = 628 # 10ms has one flow demand, control circle is 10ms
        self.link_capacity_sum_a = self.link_capacity_1 + min(self.link_capacity_2, self.link_capacity_5, self.link_capacity_7) # use min NOT max
        # self.flow_a = np.array([80]*self.point_count) / self.link_capacity_sum_a
	# self.flow_a = np.array([120]*self.point_count) / self.link_capacity_sum_a
        self.flow_a = ( 120 + np.sin( np.linspace(0, 2*math.pi, num=self.point_count) )*20 ) / self.link_capacity_sum_a # [100-140]
        self.link_capacity_sum_b = self.link_capacity_4 + min(self.link_capacity_3, self.link_capacity_6, self.link_capacity_7)
        # self.flow_b = np.array([40]*self.point_count) / self.link_capacity_sum_b
	# self.flow_b = np.array([80]*self.point_count) / self.link_capacity_sum_b
        self.flow_b = ( 70 + np.cos( np.linspace(0, 2*math.pi, num=self.point_count) )*10 ) / self.link_capacity_sum_b # [60-80]
        self.max_step_count = 20
        # np.random.seed(0)
        
    def set_init_state(self):
        # np.random.seed(0)
        self.data_step = np.random.randint(self.point_count)
        # self.data_step_count = 0
        self.cur_state = np.array(
            [self.flow_a[self.data_step], self.flow_b[self.data_step]]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # cur_useage_rate
            + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # cur_remain_rate
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # cur_overload_rate
        )
        return self.cur_state
    
    def take_action(self, action):
        print "cur_state ==>", list(self.cur_state)
        print "action ==>", action
        action = np.array(action)
        # self.data_step_count += 1
        
        cur_flow_a = self.cur_state[0] * self.link_capacity_sum_a
        cur_flow_b = self.cur_state[1] * self.link_capacity_sum_b
        cur_flow_a_1 = cur_flow_a * action[0]
        cur_flow_a_2 = cur_flow_a * action[1]
        cur_flow_b_3 = cur_flow_b * action[2]
        cur_flow_b_4 = cur_flow_b * action[3]
        next_through_put_1 = cur_flow_a_1 + self.cur_state[-7] * self.link_capacity_1
        next_through_put_2 = cur_flow_a_2 + self.cur_state[-6] * self.link_capacity_2
        next_through_put_3 = cur_flow_b_3 + self.cur_state[-5] * self.link_capacity_3
        next_through_put_4 = cur_flow_b_4 + self.cur_state[-4] * self.link_capacity_4
        next_through_put_5 = cur_flow_a_2 + self.cur_state[-3] * self.link_capacity_5
        next_through_put_6 = cur_flow_b_3 + self.cur_state[-2] * self.link_capacity_6
        next_through_put_7 = cur_flow_a_2 + cur_flow_b_3 + self.cur_state[-1] * self.link_capacity_7
        next_through_put = np.array([next_through_put_1, next_through_put_2, next_through_put_3, 
                                     next_through_put_4, next_through_put_5, next_through_put_6, next_through_put_7])
        next_useage_rate = next_through_put / self.link_capacity
        print "next_useage_rate ==>", next_useage_rate
        next_remain_rate = 1 - next_useage_rate
        # print "next_remain_rate ==>", next_remain_rate
        next_overload_rate = next_useage_rate - 1
        for i, nor in enumerate(next_overload_rate):
            if nor<0.0:
                next_overload_rate[i] = 0.0
        self.data_step += 1
        self.data_step %= self.point_count
        next_state = np.array(
            [self.flow_a[self.data_step], self.flow_b[self.data_step]] + list(next_useage_rate) + list(next_remain_rate) + list(next_overload_rate)
        )
        self.cur_state = next_state
        
        max_useage_rate = max(next_useage_rate)
        reward = 1 - max_useage_rate
        if max_useage_rate > 1.2:
            terminal = 1
            reward += -1.0
        # elif self.data_step_count == self.max_step_count:
        #     reward += 1.0
        #     terminal = 2
        else:
            terminal = 0
        print "reward ==>", reward
        
        return reward, next_state, terminal, next_useage_rate

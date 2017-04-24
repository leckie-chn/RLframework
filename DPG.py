import numpy as np
import tensorflow as tf

from DPGModel import DPGModel
from Environment import CreateEnvironment
from ReplayBuffer import TimeoutReplayBuffer


class DPGAgent(object):
    """Agent for Deterministic Policy Gradient Algorithm"""

    def __init__(self, model_, max_round=1000, gamma=0.99,
                 eps_start=1.00, eps_end=0.1, eps_rounds=1000, n_step=1, n_sample=50, n_thread=None,
                 r_timeout=5, batch_size=32, test_round=None, log_dir = 'logs/summary'):
        self.max_round = max_round
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_rounds = eps_rounds
        self.n_step = n_step
        self.n_sample = n_sample
        self.n_thread = n_thread
        self.batch_size = batch_size
        self.test_round = int(test_round) if test_round is not None else None

        state_dim, action_dim, self.env = CreateEnvironment()
        self.model = model_ # type: DPGModel
        self.replaybuffer = TimeoutReplayBuffer(r_timeout, state_dim, action_dim)
        # self.history = {
        #     'loss': np.empty((max_round)),
        #     'action_error': np.empty((max_round)),
        #     'test_data': []
        # }
        self.summary_writer = tf.summary.FileWriter(logdir=log_dir)

    def _async_explore(self, env, model, eps, n_step):
        """deprecated"""
        pass
        # TODO implement async exploration, wait for model._make_predict_function() issue

    def _explore(self, env, model, eps):
        """
        :type env: Environment
        :type model: DPGModel
        :param eps: noise factor
        :type eps: float
        :return: state, action, reward, next_state
        """
        # TODO add n-step support
        state = env.get_state()
        action = model.predict_Pi(state[np.newaxis, :])[0]
        action = np.maximum(np.random.normal(action, eps, action.shape), np.ones_like(action) * 1e-3)
        action = action / np.sum(action)  # add Gaussian noise and normalize
        reward = env.take_action(action)
        next_state = env.get_state()
        return state, action, reward, next_state

    def _bootstrap(self, states, rewards, model, gamma):
        """
        generate target-Q value using bootstrap methods
        :param states: state batch
        :type states: np.array
        :param rewards: reward batch before bootstraping
        :type rewards: np.array[float]
        :type model: DPGModel 
        :param gamma: discount factor on critic network
        :type gamma: float
        :return: targetQ value
        """
        nextQ = model.predict_Q(states, model.predict_Pi(states))[:, 0]
        return rewards + gamma * nextQ

    def _get_env(self):
        # TODO random sample environment from the Environment Pool
        if self.env.isTerminal is True:
            _, _, self.env = CreateEnvironment()
        return self.env

    def _put_env(self):
        # TODO
        pass

    def _weight_diff(self, old_weight, new_weight):
        """
        :type old_weight: list[np.array]
        :type new_weight: iist[np.array]
        :rtype: float
        """
        old_flat = np.concatenate(tuple([np.array(w).flatten() for w in old_weight]))
        new_flat = np.concatenate(tuple([np.array(w).flatten() for w in new_weight]))
        return np.linalg.norm(old_flat - new_flat)

    def train(self, verbose=0):
        """
        the process of running A3C algorithm
        :return: training history
        """
        eps = self.eps_start
        for roundNo in xrange(self.max_round):
            for i in xrange(self.n_sample):
                state, action, reward, next_state = self._explore(self._get_env(), self.model, eps)
                self.replaybuffer.add(state, action, reward, next_state)
            # perform gradient descent on Actor & Critic
            state_batch, action_batch, reward_batch, next_state_batch = self.replaybuffer.get_batch(self.batch_size)
            summary_loss = self.model.train_Q(state_batch, action_batch,
                                                    self._bootstrap(next_state_batch, reward_batch, self.model,
                                                                    self.gamma))
            self.model.train_Pi(state_batch, action_batch)
            if eps >= self.eps_end:
                eps -= (self.eps_start - self.eps_end) / self.eps_rounds
            self.replaybuffer.tick()

            summary_abs_error = self.test()
            # save log data
            self.summary_writer.add_summary(summary_loss, roundNo)
            self.summary_writer.add_summary(summary_abs_error, roundNo)
            # self.history['loss'][roundNo] = np.average(loss)
            # self.history['action_error'][roundNo] = self.test(
            #     self.test_round is not None and roundNo % self.test_round == 0)
            # print "round {}: average loss = {}, abs action error = {}".format(roundNo, self.history['loss'][roundNo],
            #                                                                   self.history['action_error'][roundNo])
        self.summary_writer.close()

    def test(self):
        _, _, env = CreateEnvironment()
        action_history = []
        correct_action_history = []
        while env.isTerminal is False:
            state = env.get_state()
            action = self.model.predict_Pi(state[np.newaxis, :])[0]
            action_history.append(action)
            correct_action_history.append(env.correct_action())
            _ = env.take_action(action)
        return self.model.metric_Pi(action_history, correct_action_history)
        # self.history['test_action'].append(action_history)
        # self.history['test_correct'].append(correct_action_history)

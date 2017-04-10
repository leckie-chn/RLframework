import threading
import numpy as np
import tensorflow as tf
import keras.backend as K
import copy

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Environment import CreateEnvironment
from ReplayBuffer import TimeoutReplayBuffer


class DPGAgent(object):
    """Agent for Deterministic Policy Gradient Algorithm"""

    def __init__(self, envopt='single-central', max_round=1000, gamma=0.99, lr_A=1e-3, lr_C=1e-2,
                 eps_start=1.00, eps_end=0.1, eps_rounds=1000, n_step=1, n_sample=50, n_thread=None,
                 r_timeout=5, batch_size=32):
        self.max_round = max_round
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_rounds = eps_rounds
        self.n_step = n_step
        self.n_sample = n_sample
        self.n_thread = n_thread
        self.batch_size = batch_size
        self.envopt = envopt
        state_dim, action_dim, self.env = CreateEnvironment(self.envopt)
        session = tf.Session()
        self.actor = ActorNetwork(session, state_dim, action_dim, lr_A)  # type: ActorNetwork
        self.critic = CriticNetwork(session, state_dim, action_dim, lr_C)  # type: CriticNetwork
        self.replaybuffer = TimeoutReplayBuffer(r_timeout, state_dim, action_dim)
        self.history = {
            'loss': np.empty((max_round)),
            'reward': np.empty((max_round)),
            'action_error': np.empty((max_round)),
        }

    def _async_explore(self, env, actor, eps, n_step):
        """deprecated"""
        pass
        # TODO implement async exploration, wait for model._make_predict_function() issue

    def _explore(self, env, actor, eps):
        """
        :type env: Environment
        :type actor: ActorNetwork
        :param eps: noise factor
        :type eps: float
        :return: state, action, reward, next_state
        """
        # TODO add n-step support
        state = env.get_state()
        action = actor.model.predict(state[np.newaxis, :])[0]
        action = np.maximum(np.random.normal(action, eps, action.shape), np.ones_like(action) * 1e-3)
        action = action / np.sum(action)  # add Gaussian noise and normalize
        reward = env.take_action(action)
        next_state = env.get_state()
        return state, action, reward, next_state

    def _bootstrap(self, states, rewards, actor, critic, gamma):
        """
        generate target-Q value using bootstrap methods
        :param states: state batch
        :type states: np.array
        :param rewards: reward batch before bootstraping
        :type rewards: np.array[float]
        :type actor: ActorNetwork
        :type critic: CriticNetwork
        :param gamma: discount factor on critic network
        :type gamma: float
        :return: targetQ value
        """
        nextQ = critic.model.predict([states, actor.model.predict(states)])[:, 0]
        return rewards + gamma * nextQ

    def _get_env(self):
        # TODO random sample environment from the Environment Pool
        if self.env.isTerminal is True:
            _, _, self.env = CreateEnvironment(self.envopt)
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

    def train(self):
        """
        the process of running A3C algorithm
        :return: training history
        """
        eps = self.eps_start
        for roundNo in xrange(self.max_round):
            for i in xrange(self.n_sample):
                state, action, reward, next_state = self._explore(self._get_env(), self.actor, eps)
                self.replaybuffer.add(state, action, reward, next_state)
            # perform gradient descent on Actor & Critic
            state_batch, action_batch, reward_batch, next_state_batch = self.replaybuffer.get_batch(self.batch_size)
            loss = self.critic.model.train_on_batch([state_batch, action_batch],
                                                    self._bootstrap(next_state_batch, reward_batch, self.actor,
                                                                    self.critic, self.gamma))
            grad_for_actor = self.critic.gradients(state_batch, action_batch)
            print "round {}: average norm of grad_for_actor = {}".format(roundNo, np.average(np.sum(np.square(grad_for_actor), axis=1)))
            old_weight = copy.deepcopy(self.actor.model.get_weights())
            self.actor.train(state_batch, grad_for_actor)
            new_weight = self.actor.model.get_weights()
            weight_delta = self._weight_diff(old_weight, new_weight)
            print "round {} actor model delta = {}".format(roundNo, weight_delta)
            if eps >= self.eps_end:
                eps -= (self.eps_start - self.eps_end) / self.eps_rounds
            self.history['loss'][roundNo] = np.average(loss)
            self.history['reward'][roundNo], self.history['action_error'][roundNo] = self.test()
            print "round {}: average loss = {}, average reward = {}, abs action error = {}".format(roundNo,
                                                                                                   self.history['loss'][
                                                                                                       roundNo],
                                                                                                   self.history[
                                                                                                       'reward'][
                                                                                                       roundNo],
                                                                                                   self.history[
                                                                                                       'action_error'][
                                                                                                       roundNo])
            self.replaybuffer.tick()

    def test(self):
        _, _, env = CreateEnvironment('single-central')
        total_reward = 0.0
        action_history = []
        correct_action_history = []
        while env.isTerminal is False:
            state = env.get_state()
            action = self.actor.model.predict(state[np.newaxis, :])[0]
            action_history.append(action)
            correct_action_history.append(env.correct_action())
            reward = env.take_action(action)
            total_reward += reward
        # self.history['test_action'].append(action_history)
        # self.history['test_correct'].append(correct_action_history)
        avg_reward = total_reward / env.tm_step
        avg_error = np.average(np.abs(np.array(action_history) - np.array(correct_action_history)))
        return avg_reward, avg_error

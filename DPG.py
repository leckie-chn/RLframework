import threading
import numpy as np

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Environment import CreateEnvironment
from ReplayBuffer import TimeoutReplayBuffer


class DPGAgent(object):
    """Agent for Deterministic Policy Gradient Algorithm"""

    def __init__(self, actor_, critic_, envopt='single-central', max_round=1000, gamma=0.99,
                 eps_start=1.00, eps_end=0.1, eps_rounds=1000, n_step=1, n_sample=50, n_thread=None,
                 r_timeout=5, batch_size=32, test_round=None):
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

        self.envopt = envopt
        state_dim, action_dim, self.env = CreateEnvironment(self.envopt)
        self.actor = actor_  # type: ActorNetwork
        self.critic = critic_  # type: CriticNetwork
        self.replaybuffer = TimeoutReplayBuffer(r_timeout, state_dim, action_dim)
        self.history = {
            'loss': np.empty((max_round)),
            'action_error': np.empty((max_round)),
            'test_data': []
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

    def train(self, verbose=0):
        """
        the process of running A3C algorithm
        :return: training history
        """
        eps = self.eps_start
        if verbose > 0:
            old_weight = self.actor.model.get_weights()
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
            self.actor.train(state_batch, grad_for_actor)
            if eps >= self.eps_end:
                eps -= (self.eps_start - self.eps_end) / self.eps_rounds
            self.replaybuffer.tick()

            # save log data
            self.history['loss'][roundNo] = np.average(loss)
            self.history['action_error'][roundNo] = self.test(
                self.test_round is not None and roundNo % self.test_round == 0)
            print "round {}: average loss = {}, abs action error = {}".format(roundNo, self.history['loss'][roundNo],
                                                                              self.history['action_error'][roundNo])
            if verbose > 0:
                new_weight = self.actor.model.get_weights()
                weight_delta = self._weight_diff(old_weight, new_weight)
                old_weight = new_weight
                print "round {} actor model delta = {}".format(roundNo, weight_delta)

    def test(self, save_plot=False):
        _, _, env = CreateEnvironment('single-central')
        action_history = []
        correct_action_history = []
        while env.isTerminal is False:
            state = env.get_state()
            action = self.actor.model.predict(state[np.newaxis, :])[0]
            action_history.append(action)
            correct_action_history.append(env.correct_action())
            _ = env.take_action(action)
        # self.history['test_action'].append(action_history)
        # self.history['test_correct'].append(correct_action_history)
        avg_error = np.average(np.abs(np.array(action_history) - np.array(correct_action_history)))
        if save_plot is True:
            self.history['test_data'].append({
                'action': action_history,
                'correct': correct_action_history,
            })
        return avg_error

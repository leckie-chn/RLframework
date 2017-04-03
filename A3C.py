import threading
import numpy as np
import tensorflow as tf
from Queue import Queue

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Environment import Environment


class AgentThread(threading.Thread):
    """Agent Thread for collecting data"""
    def __init__(self, actor_, critic_, n_step_, gamma_):
        super(AgentThread, self).__init__()
        self.env = None         # type: Environment
        self.actor = actor_     # type: ActorNetwork
        self.critic = critic_   # type: CriticNetwork
        self.eps = None         # type: float
        self.n_step = n_step_   # type: int
        self.gamma = gamma_
        self.data = None        # type: dict

    def syncModel(self, env_, actor_, critic_, eps_):
        """synchronize A-C models before collecting data"""
        # TODO copy env, actor, critic
        self.env = env_     # shallow copy
        self.actor.model.set_weights(actor_.model.get_weights())        # deep copy, by weights
        self.critic.model.set_weights(critic_.model.get_weights())      # deep copy, by weights
        self.eps = eps_     # shallow copy
        self.data = {}

    def run(self):
        """run behavior policy self.actor to collect experience data in self.data"""
        state = self.env.get_state()
        action = self.actor.model.predict(state[np.newaxis, :])[0]
        action = np.maximum(np.random.normal(action, self.eps, action.shape), np.ones_like(action) * 1e-3)
        action = action / np.sum(action)                    # add Gaussian noise and normalize
        self.data['state'] = state
        self.data['action'] = action
        targetQ = 0.0
        tgamma = 1.0                                        # aggregated gamma
        for i in xrange(self.n_step):
            reward, state = self.env.take_action(action)
            targetQ += reward * tgamma
            if self.env.isTerminal == True:
                break
            action = self.actor.model.predict(state)[0]
            tgamma *= self.gamma
        targetQ += self.critic.model.predict([state, action])[0] * tgamma
        self.data['targetQ'] = targetQ
        print "Agent Thread Done"

class A3CRunner:
    """Runner for Asynchronous Advantage Actor Critic Algorithm"""
    def __init__(self, session, state_dim, action_dim, param_=None):
        if param_ is None:
            param_ = {}
        default_param = {
            'n_thread': 50,     # number of Asynchronous Agent Threads
            'n_step': 10,       # max number of step for bootstraping
            'eps_start': 1.00,  # value of eps at the beginning
            'eps_end':  0.15,   # value of eps at the end
            'eps_rounds': 200,  # number of rounds taken from eps_start to eps_end
            'max_round': 1000,  # number of rounds to be trained
            'gamma': 0.99,      # discount factor
            'lr_A': 1e-3,       # learning rate for actor network
            'lr_C': 1e-2,       # learning rate for critic network
            # 'test': True,  TODO: test policy every round
        }

        self.param = {}
        for key in default_param:
            if key in param_:
                self.param[key] = param_[key]
            else:
                self.param[key] = default_param[key]
        self.actor = ActorNetwork(tf.Session(), state_dim, action_dim, self.param['lr_A']) # type: ActorNetwork
        self.critic = CriticNetwork(tf.Session(), state_dim, action_dim, self.param['lr_C'])   # type: CriticNetwork
        self.envPool = Queue(self.param['n_thread'])
        self.loss_history = np.empty((self.param['max_round']))         # loss history for Q-Network
        self.reward_hisotry = np.empty((self.param['max_round']))       # reward for each test
        self.AgentPool = [AgentThread(ActorNetwork(session, state_dim, action_dim, self.param['lr_A']),
                                      CriticNetwork(session, state_dim, action_dim, self.param['lr_C']),
                                      self.param['n_step'], self.param['gamma']) for i in xrange(self.param['n_thread'])]

    def getEnv(self):
        """Get an Environment Simulator from self.envPool"""
        if self.envPool.empty():
            return Environment()
        else:
            return self.envPool.get()

    def returnEnv(self, env_):
        """Return an Environemt Simluator back to self.envPool if not Terminal """
        if env_.isTerminal == False:
            self.envPool.put(env_)

    def run(self):
        """
        the process of running A3C algorithm
        :return: training history
        """
        eps = self.param['eps_start']
        for roundNo in xrange(self.param['max_round']):
            for agent in self.AgentPool:
                agent.syncModel(self.getEnv(), self.actor, self.critic, eps)
                agent.start()
            for agent in self.AgentPool:
                agent.join()
            DataPool = {
                'states': [],
                'actions': [],
                'targetQ': [],
            }
            for agent in self.AgentPool:
                DataPool['states'].append(agent.data['state'])
                DataPool['actions'].append(agent.data['action'])
                DataPool['targetQ'].append(agent.data['targetQ'])
            # perform gradient descent on Actor & Critic
            loss = self.critic.model.train_on_batch([DataPool['states'], DataPool['actions']], DataPool['targetQ'])
            grad_for_actor = self.critic.gradients(DataPool['states'], DataPool['actions'])
            self.actor.train(DataPool['states'], grad_for_actor)
            if eps >= self.param['eps_end']:
                eps -= (self.param['eps_start'] - self.param['eps_end']) / self.param['eps_rounds']
            self.loss_history[roundNo] = np.average(loss)
            self.reward_hisotry[roundNo] = self.test()

    def test(self):
        env = Environment()
        max_step = 100
        total_reward = 0.0
        for i in xrange(max_step):
            state = env.get_state()
            action = self.actor.model.predict(state)[0]
            reward, state = env.take_action(action)
            total_reward += reward
            if env.isTerminal:
                return total_reward / (i + 1)
        return total_reward / max_step

    def getActor(self):
        return self.actor
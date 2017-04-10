

from DPG import DPGAgent
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Environment import CreateEnvironment

import pickle
import cProfile
import tensorflow as tf


env_setting = 'single-central'
state_dim, action_dim, _ = CreateEnvironment(env_setting)

sess = tf.Session()
actor = ActorNetwork(sess, state_dim, action_dim, 1e-3)
critic = CriticNetwork(sess, state_dim, action_dim, 1e-4)
agent = DPGAgent(actor_=actor, critic_=critic, max_round=10, n_sample=10, batch_size=32, gamma=0.0)



profile_path = None
if profile_path is not None:
    cProfile.run(agent.train(), profile_path)
else:
    agent.train()

actor.model.save('saved_networks/actor.h5')
critic.model.save('saved_networks/critic.h5')

fl = open('history.json', 'w')
pickle.dump(agent.history, fl)
fl.close()


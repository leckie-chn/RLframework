

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
critic = CriticNetwork(sess, state_dim, action_dim, 0.005)
#actor.model.load_weights('saved_networks/actor_weight.h5')
#critic.model.load_weights('saved_networks/critic_weight.h5')
agent = DPGAgent(actor_=actor, critic_=critic, max_round=2000, n_sample=256, batch_size=512,
                 gamma=0.9, eps_end=0.01, eps_rounds=990)



profile_path = None
if profile_path is not None:
    cProfile.run(agent.train(), profile_path)
else:
    agent.train()

actor.model.save_weights('saved_networks/actor_weight.h5')
critic.model.save_weights('saved_networks/critic_weight.h5')

fl = open('history.pkl', 'w')
pickle.dump(agent.history, fl)
fl.close()


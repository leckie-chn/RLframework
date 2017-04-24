import tensorflow as tf

from DPG import DPGAgent
from DPGModel import DPGModel
from Environment import CreateEnvironment, SetEnvironment

env_setting = 'single-central'
SetEnvironment(env_setting, [20, 30, 50], 628)
state_dim, action_dim, _ = CreateEnvironment()

model = DPGModel(state_dim, action_dim)

print "trainable weight in policy"
for var in model.policy_weights:
    print var.name, var.get_shape()

print "all trainable weights"
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print var.name, var.get_shape()
# sys.exit(0)

#actor = ActorNetwork(sess, state_dim, action_dim, 1e-3)
#critic = CriticNetwork(sess, state_dim, action_dim, 1e-4)
#actor.model.load_weights('saved_networks/actor_weight.h5')
#critic.model.load_weights('saved_networks/critic_weight.h5')
agent = DPGAgent(model, max_round=100, n_sample=256, batch_size=512,
                 gamma=0.0, eps_end=0.01, eps_rounds=1980)
agent.train()

#actor.model.save_weights('saved_networks/actor_weight.h5')
#critic.model.save_weights('saved_networks/critic_weight.h5')


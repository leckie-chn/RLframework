from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer
from Environment import Environment

import tensorflow as tf
from keras import backend as K

import numpy as np
import json
import sys


def OU_Function(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)

def Play_Game(use_old_weight, train_indicator):
    
    REPLAY_BUFFER_SIZE = 628*5
    TRAIN_BATCH_SIZE = 32*5
    GAMMA = 0.99
    TAU = 0.001 # Target Network HyperParameters
    LR_A = 0.0001 # Learning rate for Actor
    LR_C = 0.001 # Lerning rate for Critic
    EXPLORE = 0.0001
    
    state_dim = 4
    action_dim = 3
    episode_count = 20000
    max_step_count = 100000
    epsilon = 1.0
    
    np.random.seed(1234)
    
    
    
    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    actor = ActorNetwork(sess, state_dim, action_dim, TAU, LR_A)
    critic = CriticNetwork(sess, state_dim, action_dim, TAU, LR_C)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    environment = Environment()
    
    
    if use_old_weight:
        try:
            actor.model.load_weights("actor_model.h5")
            critic.model.load_weights("critic_model.h5")
            actor.target_model.load_weights("actor_model.h5")
            critic.target_model.load_weights("critic_model.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")
    
    for episode in range(episode_count):
        state = environment.set_init_state()
        for time_step in range(max_step_count):
            epsilon -= EXPLORE
            print "="*10, "episode", episode, "***** time_step", time_step
            action = actor.model.predict(np.array(state).reshape(1,state_dim))[0]            
            print "action ==>", action
            if epsilon>0 and train_indicator is True:
                noise = [0]*len(action)
                for i, act in enumerate(action):
                    noise[i] = epsilon * (act*0.2 + 0.05*np.random.randn(1))
                    action[i] = min(1.0, max(action[i]+noise[i], 0.0))
                action = list( np.array(action) / sum(action) )
                print "action ==>", action
            
            reward, next_state, terminal = environment.take_action(action)
            state = next_state
            if terminal:
                break
            
            replay_buffer.add(state, action, reward, next_state, terminal)
            # print "************** train model"
            batch = replay_buffer.get_batch(TRAIN_BATCH_SIZE)
            if batch is not None:
                state_batch = np.asarray([e[0] for e in batch])
                action_batch = np.asarray([e[1] for e in batch])
                reward_batch = np.asarray([e[2] for e in batch])
                next_state_batch = np.asarray([e[3] for e in batch])
                terminal_batch = np.asarray([e[4] for e in batch])
                target_y_batch = np.asarray([0.0]*len(batch))
                
                target_Q_value_batch = critic.target_model.predict([next_state_batch, actor.target_model.predict(next_state_batch)])
                # use the output of target_model-network instead of model-network!
                print "target_Q_value_batch ==>", target_Q_value_batch[:5]
                for k in range(len(batch)):
                    if terminal_batch[k]:
                        target_y_batch[k] = reward_batch[k]
                    else:
                        target_y_batch[k] = reward_batch[k] + GAMMA*target_Q_value_batch[k]
                        # NO need to use np.max(target_Q_value_batch), we think current actions are the best of ActorNetwork
                print "target_y_batch ==>", target_y_batch[:5]
                
                # train critic
                critic.model.train_on_batch([state_batch, action_batch], target_y_batch)
                # train actor
                action_batch_for_grad = actor.model.predict(state_batch)
                action_grads_of_critic = critic.gradients(state_batch, action_batch_for_grad)
                actor.train(state_batch, action_grads_of_critic)
                # update target network
                actor.train_target_network()
                critic.train_target_network()
            
        actor.target_model.save_weights("saved_networks/actor_model.h5", overwrite=True)
        with open("saved_networks/actor_model.json", "w") as outfile:
            json.dump(actor.target_model.to_json(), outfile)
        critic.target_model.save_weights("saved_networks/criti_cmodel.h5", overwrite=True)
        with open("saved_networks/critic_model.json", "w") as outfile:
            json.dump(critic.target_model.to_json(), outfile)


if __name__ == "__main__":
    # use_old_weight, 
    if len(sys.argv)==2:
        use_old_weight = sys.argv[0]
        train_indicator = sys.argv[1]
    else:
        use_old_weight = False
        train_indicator = True
    Play_Game(use_old_weight, train_indicator)

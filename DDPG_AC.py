
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer
from Environment import Environment

import tensorflow as tf
from keras import backend as K

import numpy as np
import json
import sys

import cPickle


import matplotlib.pyplot as plt
'''
modify matplotlib so that we can draw picture under linux:
>> find ~ -name matplotlibrc
/home/maohangyu/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc
/home/maohangyu/anaconda2/pkgs/matplotlib-2.0.0-np111py27_0/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc
>> vim /home/maohangyu/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc
change line-38 from "backend      : *Agg" to "backend      : Agg"
'''
def plot_test():
    a = np.arange(0,10,1)
    b = a.__pow__(2.0)
    c = np.arange(15,61,5)
    plt.figure(figsize=(8,6))
    plt.title('Plot Example', size=14)
    plt.xlabel('x-axis', size=14)
    plt.ylabel('y-axis', size=14)
    plt.plot(a, b, color='b', linestyle='--', marker='o', label='b data')
    plt.plot(a, c, color='r', linestyle='-', label='c data')
    plt.legend(loc='upper left')
    plt.savefig('plot_test.png')

def plot(data, name):
    plt.figure(figsize=(80,6))
    plt.plot(data, color='b', linestyle='-')
    plt.savefig(name)
    # scp *.png maohangyu@162.105.146.201:/home/maohangyu/

def plot_two_data(data1, data2, name):
    plt.figure(figsize=(80,6))
    plt.plot(data1, color='b', linestyle='-', label='true action')
    plt.plot(data2, color='r', linestyle='-', label='best action')
    plt.savefig(name)
    # scp *.png maohangyu@162.105.146.201:/home/maohangyu/

if __name__ == "__main__":
    # use_old_weight, 
    if len(sys.argv)==2:
        use_old_weight = sys.argv[0]
        train_indicator = sys.argv[1]
    else:
        use_old_weight = False
        train_indicator = True
    
    REPLAY_BUFFER_SIZE = 628*10
    TRAIN_BATCH_SIZE = 64
    GAMMA = 0.9
    TAU = 0.001 # Target Network HyperParameters
    LR_A = 0.001 # Learning rate for Actor
    LR_C = 0.01 # Lerning rate for Critic
    
    state_dim = 23
    action_dim = 4
    episode_count = 5000
    max_step_count = 20
    
    np.random.seed(1234)
    
    
    
    # Tensorflow GPU optimization
    tf.set_random_seed(0)
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
    
    loss_buffer = []
    action_buffer = []
    best_action_buffer = []
    next_useage_rate_buffer = []
    for episode in range(episode_count):
        state = environment.set_init_state()
        temp_buffer = []
        for time_step in range(max_step_count):
            print "="*10, "episode", episode, "***** time_step", time_step
            noise_state = np.array(state).reshape(1,state_dim) + np.random.randint(0, 1000, (1,state_dim))/(10000.0+episode*10)
            flow_a = noise_state[0][0]*300.0; flow_b = noise_state[0][1]*300.0; 
            best_action_a_1 = (flow_a+flow_b)/(4*flow_a); best_action_a_2 = 1-best_action_a_1; 
            best_action_b_4 = (flow_a+flow_b)/(4*flow_b); best_action_b_3 = 1-best_action_b_4; 
            best_action_buffer.append([best_action_a_1, best_action_a_2, best_action_b_3, best_action_b_4])
            print "best_action ==>", best_action_a_1, best_action_a_2, best_action_b_3, best_action_b_4
            action = actor.model.predict( noise_state )[0]
            action_buffer.append(action)
            reward, next_state, terminal, next_useage_rate = environment.take_action(action)
            next_useage_rate_buffer.append(next_useage_rate)
            replay_buffer.add_experience([state, action, reward, next_state, terminal])
            temp_buffer.append([state, action, reward, next_state, terminal])
            state = next_state
            
            batch = replay_buffer.get_batch_experience(TRAIN_BATCH_SIZE)
            if terminal==1:
                batch.extend(temp_buffer)
            state_batch = np.asarray([e[0] for e in batch])
            action_batch = np.asarray([e[1] for e in batch])
            reward_batch = np.asarray([e[2] for e in batch])
            next_state_batch = np.asarray([e[3] for e in batch])
            terminal_batch = np.asarray([e[4] for e in batch])
            target_y_batch = np.asarray([0.0]*len(batch))
            
            target_Q_value_batch = critic.target_model.predict([next_state_batch, actor.target_model.predict(next_state_batch)])
            # use the output of target_model-network instead of model-network!
            for k in range(len(batch)):
                if terminal_batch[k]:
                    target_y_batch[k] = reward_batch[k]
                else:
                    target_y_batch[k] = reward_batch[k] + GAMMA*target_Q_value_batch[k]
                    # NO need to use np.max(target_Q_value_batch), we think current actions are the best of ActorNetwork
            
            # train critic
            loss = critic.model.train_on_batch([state_batch, action_batch], target_y_batch)
            loss_buffer.append(loss)
            print "loss ==>", loss
            # train actor
            action_batch_for_grad = actor.model.predict(state_batch)
            action_grads_of_critic = critic.gradients(state_batch, action_batch_for_grad)
            actor.train(state_batch, action_grads_of_critic)
            # update target network
            actor.train_target_network()
            critic.train_target_network()
            
            if terminal==1:
                break
        
        if episode % 1000 == 0:
            # print actor.model.get_weights()
            actor.target_model.save_weights("saved_networks/actor_model.h5", overwrite=True)
            with open("saved_networks/actor_model.json", "w") as outfile:
                json.dump(actor.target_model.to_json(), outfile)
            critic.target_model.save_weights("saved_networks/criti_cmodel.h5", overwrite=True)
            with open("saved_networks/critic_model.json", "w") as outfile:
                json.dump(critic.target_model.to_json(), outfile)
    
    plot(loss_buffer, "loss.png")
    action_buffer = np.array(action_buffer)
    plot(action_buffer[:,0], "action_1.png")
    plot(action_buffer[:,1], "action_2.png")
    plot(action_buffer[:,2], "action_3.png")
    plot(action_buffer[:,3], "action_4.png")
    
    next_useage_rate_buffer = np.array(next_useage_rate_buffer).reshape(len(next_useage_rate_buffer),-1)
    plot(next_useage_rate_buffer[:,0], "useage_1.png")
    plot(next_useage_rate_buffer[:,3], "useage_4.png")
    plot(next_useage_rate_buffer[:,6], "useage_7.png")
    
    action_buffer = np.array(action_buffer)
    best_action_buffer = np.array(best_action_buffer)
    plot_two_data(action_buffer[:,0], best_action_buffer[:,0], "action_1.png")
    plot_two_data(action_buffer[:,1], best_action_buffer[:,1], "action_2.png")
    plot_two_data(action_buffer[:,2], best_action_buffer[:,2], "action_3.png")
    plot_two_data(action_buffer[:,3], best_action_buffer[:,3], "action_4.png")
    
    plot(loss_buffer[-2000:], "last100episode_loss.png")
    plot(next_useage_rate_buffer[-2000:,0], "last100episode_useage_1.png")
    plot(next_useage_rate_buffer[-2000:,3], "last100episode_useage_4.png")
    plot(next_useage_rate_buffer[-2000:,6], "last100episode_useage_7.png")
    plot_two_data(action_buffer[-2000:,0], best_action_buffer[-2000:,0], "last100episode_action_1.png")
    plot_two_data(action_buffer[-2000:,1], best_action_buffer[-2000:,1], "last100episode_action_2.png")
    plot_two_data(action_buffer[-2000:,2], best_action_buffer[-2000:,2], "last100episode_action_3.png")
    plot_two_data(action_buffer[-2000:,3], best_action_buffer[-2000:,3], "last100episode_action_4.png")
    
    cPickle.dump(loss_buffer, open("saved_networks/loss_buffer.pkl","wb")) 
    cPickle.dump(next_useage_rate_buffer, open("saved_networks/next_useage_rate_buffer.pkl","wb")) 
    cPickle.dump(action_buffer, open("saved_networks/action_buffer.pkl","wb")) 
    cPickle.dump(best_action_buffer, open("saved_networks/best_action_buffer.pkl","wb")) 
    # data = cPickle.load(open("saved_networks/loss_buffer.pkl","rb"))





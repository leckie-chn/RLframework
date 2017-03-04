import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Environment import Environment


def Play_Game():
    TAU = 0.001
    Learning_Rate_A = 1e-4
    Learning_Rate_C = 0.0005
    state_size = 1
    action_size = 3
    episode_count = 600000
    batch_size = 10
    epsilon = 1.0
    eps_decay = 0.000005

    np.random.seed(1234)

    sess = tf.Session()
    K.set_session(sess)

    # init actor & Critic
    actor = ActorNetwork(sess, state_size, action_size, TAU, Learning_Rate_A)
    critic = CriticNetwork(sess, state_size, action_size, TAU, Learning_Rate_C)
    env = Environment()

    batch_count = 0
    state_batch = np.zeros((batch_size, state_size))
    action_batch = np.zeros((batch_size, action_size))
    reward_batch = np.zeros((batch_size))

    reward_history = np.empty(episode_count)
    critic_loss_history = np.empty(episode_count / batch_size)

    for episode in range(episode_count):
        state = env.get_state()
        action = actor.model.predict(np.array(state).reshape(1, state_size))[0]
        if epsilon > 0:
            epsilon -= eps_decay
            noise = np.random.rand(action_size) 
            action = (1 - epsilon) * action + epsilon * noise
        # print "action ==> ", action
        reward = env.take_action(action)
        reward_history[episode] = reward
        # gather batch data
        state_batch[batch_count] = state
        action_batch[batch_count] = action
        reward_batch[batch_count] = reward
        batch_count += 1

        # train on batch
        if batch_count >= batch_size:
            batch_count = 0
            critic_loss = critic.model.train_on_batch([state_batch, action_batch], reward_batch)
            action_batch_for_grad = actor.model.predict(state_batch)
            action_grads_of_critic = critic.gradients(state_batch, action_batch_for_grad)
            actor.train(state_batch, action_grads_of_critic)
            actor.train_target_network()
            critic.train_target_network()
            critic_loss_history[episode / batch_size] = critic_loss
    actor.model.save_weights('saved_networks/actor_model.h5')
    critic.model.save_weights('saved_networks/critic_model.h5')
    print "actor target model weights:", actor.target_model.get_weights()
    print "critic model weights:", critic.model.get_weights()

    plt.title('Training Reward')
    plt.plot(np.arange(episode_count), reward_history, 'o')
    plt.xlabel('Iteration')
    plt.savefig('reward.png')
    plt.close()

    plt.title('Critic Loss')
    plt.plot(np.arange(episode_count / batch_size), np.log10(critic_loss_history), 'o')
    plt.xlabel('Batch Iteration')
    plt.savefig('loss.png')
    plt.close()

    test_episode = 100
    for c in xrange(test_episode):
        state = env.get_state()
        print "state ==>", state
        action = actor.model.predict(np.array(state).reshape(1, state_size))[0]
        print "action ==>", action
        reward = env.take_action(action)
        print "reward ==>", reward

if __name__ == "__main__":
    Play_Game()

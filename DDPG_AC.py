import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

import tensorflow as tf
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from Environment import Environment
from ReplayBuffer import ReplayBuffer


def Play_Game():
    TAU = 0.01
    GAMMA = 0.99
    Buffer_Size = 500
    Batch_Size = 32
    Learning_Rate_A = 1e-4
    Learning_Rate_C = 1e-4
    state_size = 4
    action_size = 3
    episode_count = 10000
    epsilon = 1.0
    eps_decay = 0.000001

    np.random.seed(1234)

    sess = tf.Session()
    K.set_session(sess)

    # init actor & Critic
    actor = ActorNetwork(sess, state_size, action_size, TAU, Learning_Rate_A)
    critic = CriticNetwork(sess, state_size, action_size, TAU, Learning_Rate_C)
    replay_buffer = ReplayBuffer(Buffer_Size)

    reward_history = np.empty(episode_count)
    critic_loss_history = np.empty(episode_count)

    firstQconverge = False

    for episode in xrange(episode_count):
        env = Environment()
        state = env.get_state()
        isTerminal = False
        total_reward = 0.0
        total_loss = 0.0
        step_count = 0
        while isTerminal is False:
            action = actor.model.predict(state.reshape(1, state_size))[0]
            if epsilon > 0:
                epsilon -= eps_decay
                noise = np.random.rand(action_size)
                noise = noise / np.sum(noise)
                action = (1 - epsilon) * action + epsilon * noise
            next_state, reward, isTerminal = env.take_action(action)
            replay_buffer.add(state, action, reward, next_state, isTerminal)
            state = next_state
            total_reward += reward

            # train models
            batch = replay_buffer.get_batch(Batch_Size)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            isTerminals = np.asarray([1 if e[4] is True else 0 for e in batch])

            targetQ = critic.target_model.predict([new_states, actor.target_model.predict(new_states)]).reshape(
                rewards.shape)
            targetY = rewards + GAMMA * isTerminals * targetQ
            total_loss += critic.model.train_on_batch([states, actions], targetY)
            action_for_grad = actor.model.predict(states)
            actor_grads = critic.gradients(states, action_for_grad)
            actor.train(states, actor_grads)
            actor.train_target_network()
            critic.train_target_network()
            step_count += 1
        reward_history[episode] = total_reward / step_count
        critic_loss_history[episode] = total_loss / step_count
        if critic_loss_history[episode] < 1e-6 and firstQconverge is False:
            epsilon = 1.0
            eps_decay = 1e-4
            print "Detect Q Network converge under 1e-6"
            firstQconverge = True
        print "average reward ==> {0}, average critic loss ==> {1}, epsilon ==> {2}".format(reward_history[episode],
                                                                           critic_loss_history[episode], epsilon)
        # test every 10 episode
        if episode % 10 == 0:
            env = Environment()
            state = env.get_state()
            isTerminal = False
            while isTerminal is False:
                action = actor.model.predict(state.reshape(1, state_size))[0]
                next_state, reward, isTerminal = env.take_action(action)
                print "state ==> {0}, action ==> {1}, reward ==> {2}".format(state, action, reward)
                state = next_state

    # save model
    actor.model.save_weights('saved_networks/actor_model.h5')
    critic.model.save_weights('saved_networks/critic_model.h5')
    print "actor target model weights:", actor.target_model.get_weights()
    print "critic model weights:", critic.model.get_weights()

    # print figures
    plt.title('Training Reward')
    plt.plot(np.arange(episode_count), reward_history, 'o')
    plt.xlabel('Iteration')
    plt.savefig('reward.png')
    plt.close()

    plt.title('Critic Loss')
    plt.plot(np.arange(episode_count), np.log10(critic_loss_history), 'o')
    plt.xlabel('Iteration')
    plt.savefig('loss.png')
    plt.close()


if __name__ == "__main__":
    Play_Game()

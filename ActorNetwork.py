
from keras.models import Model
from keras.initializations import normal
from keras.layers import Dense, Input, merge

import tensorflow as tf
from keras import backend as K



class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, TAU, LEARNING_RATE):
        self.sess = sess
        self.SD = state_dim
        self.AD = action_dim
        self.TAU = TAU
        
        K.set_session(sess)
        self.model, self.weights, self.state = self.create_actor_network()   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network() 
        self.action_gradient = tf.placeholder(tf.float32, [None, action_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
    
    def train(self, state_batch, action_grad_batch_of_critic):
        self.sess.run(self.optimize, feed_dict={
            self.state: state_batch,
            self.action_gradient: action_grad_batch_of_critic
        })
    
    def create_actor_network(self):
        print("create_actor_network")
        state = Input(shape=[self.SD])
        h1 = Dense(128, activation="relu")(state)
        h2 = Dense(64, activation="relu")(h1)
        h3_for_action_1 = Dense(32, activation="relu")(h2)
        h3_for_action_2 = Dense(32, activation="relu")(h2)
        action_1_prob = Dense(2, activation='softmax', init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3_for_action_1)
        action_2_prob = Dense(2, activation='softmax', init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3_for_action_2)
        action_prob = merge([action_1_prob, action_2_prob], mode='concat')
        model = Model(input=state, output=action_prob)
        return model, model.trainable_weights, state
    
    def train_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

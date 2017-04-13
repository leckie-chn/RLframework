
from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization

import tensorflow as tf
from keras import backend as K


# TODO polymorphism on Network Architecture

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, LEARNING_RATE):
        self.sess = sess
        self.SD = state_dim
        self.AD = action_dim
        # self.TAU = TAU

        K.set_session(sess)
        self.model, self.weights, self.state = self.create_network()
        # self.target_model, self.target_weights, self.target_state = self.create_network()
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

    def create_network(self):
        print("create_actor_network")
        state = Input(shape=[self.SD])
        h1 = Dense(64, activation="relu")(state)
        h3 = Dense(32, activation="relu")(h1)
        action = Dense(self.AD, activation='softmax', init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3)
        model = Model(input=state, output=action)
        return model, model.trainable_weights, state

    # def train_target_network(self):
    #     critic_weights = self.model.get_weights()
    #     critic_target_weights = self.target_model.get_weights()
    #     for i in xrange(len(critic_weights)):
    #         critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
    #     self.target_model.set_weights(critic_target_weights)

class ActorNewtorkV1(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, LEARNING_RATE):
        super(ActorNewtorkV1, self).__init__(sess, state_dim, action_dim, LEARNING_RATE)

    def create_network(self):
        # state = Input(shape=[self.SD])
        pass

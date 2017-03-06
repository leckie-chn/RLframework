
import tensorflow as tf
from keras import backend as K
from keras.initializations import normal
from keras.layers import Dense, Input
from keras.models import Model

HIDDEN_NOTE_COUNT = 8


class ActorNetwork(object):

    def __init__(self, sess, state_size, action_size, TAU, LEARNING_RATE):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.TAU = TAU

        K.set_session(sess)

        # create the model
        self.model, self.weights, self.state = self.create_actor_network()
        self.target_model, self.target_weights, self.target_state = self.create_actor_network()
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
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
        state = Input(shape=[self.state_size])
        h1 = Dense(10, activation='linear', init=lambda shape, name: normal(shape, scale=1.0, name=name))(state)
        action = Dense(self.action_size, activation='softmax', init=lambda shape,
                       name: normal(shape, scale=1.0, name=name))(h1)
        model = Model(input=state, output=action)
        return model, model.trainable_weights, state

    def train_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

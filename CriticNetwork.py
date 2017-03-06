
import tensorflow as tf
from keras import backend as K 
from keras.layers import Dense, Input, merge
from keras.models import Model
from keras.optimizers import Adam
HIDDEN_NOTE_COUNT = 8


class CriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, TAU, LEARNING_RATE):
        self.sess = sess
        self.SD = state_dim
        self.AD = action_dim
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        self.model, self.state, self.action = self.create_critic_network()
        self.target_model, self.target_state, self.target_action = self.create_critic_network()

        self.action_grad = tf.gradients(self.model.output, self.action)  # gradients for policy(actor) update

        self.sess.run(tf.global_variables_initializer())

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_grad, feed_dict={
            self.state: state_batch,
            self.action:  action_batch
        })[0]

    def create_critic_network(self):
        print("create_critic_network")
        state = Input(shape=[self.SD])
        action = Input(shape=[self.AD])
        h1 = merge([state, action], mode='concat')
        h2 = Dense(50, activation='sigmoid')(h1)
        h3 = Dense(50, activation='sigmoid')(h2)
        h4 = Dense(20, activation='relu')(h3)
        h5 = Dense(20, activation='relu')(h4)
        h6 = Dense(20, activation='relu')(h5)
        Q_value = Dense(1, activation='linear')(h6)
        model = Model(input=[state, action], output=Q_value)
        model.compile(loss='mse', optimizer=Adam(lr=self.LEARNING_RATE))
        return model, state, action

    def train_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

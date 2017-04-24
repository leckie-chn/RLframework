import tensorflow as tf

from constants import *


class DPGModel(object):
    def __init__(self, state_size, action_size):
        self.sess = tf.Session()
        self.state_size = state_size
        self.action_size = action_size
        self._create_network()

        # training for Q function
        self.targetQ_tensor = tf.placeholder(tf.float32, [None], name="targetQ")
        self.loss_tensor = tf.nn.l2_loss(self.value_tensor - self.targetQ_tensor)
        self.event_Q_regression = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_Q, name='Adam_Critic').minimize(
            loss=self.loss_tensor)
        # gradient of Q on action
        self.event_grad_for_action = tf.gradients(self.value_tensor, self.action_tensor)

        # apply chain rule on policy function
        self.grad_for_action_tensor = tf.placeholder(tf.float32, [None, self.action_size], name="grad_for_action")
        # set policy weights
        self.policy_weights = []
        with tf.variable_scope('state_h1', reuse=True):
            self.policy_weights.extend([
                tf.get_variable('kernel'),
                tf.get_variable('bias'),
            ])
        with tf.variable_scope('state_h2', reuse=True):
            self.policy_weights.extend([
                tf.get_variable('kernel'),
                tf.get_variable('bias'),
            ])
        with tf.variable_scope('policy', reuse=True):
            self.policy_weights.extend([
                tf.get_variable('kernel'),
                tf.get_variable('bias'),
            ])
        self.policy_grads = tf.gradients(self.policy_tensor, self.policy_weights,
                                         grad_ys=-self.grad_for_action_tensor)
        self.event_policy_gradient = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_Pi,
                                                            name='Adam_Actor').apply_gradients(
            zip(self.policy_grads, self.policy_weights))

        # tf summary & metrics
        self.summary_Q_loss = tf.summary.scalar('Q-learning loss', self.loss_tensor)
        self._create_abs_error_summary()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def _create_network(self):
        self.state_tensor = tf.placeholder(tf.float32, [None, self.state_size], name="state")
        self.action_tensor = tf.placeholder(tf.float32, [None, self.action_size], name="action")

        # create policy network
        state_h1 = tf.layers.dense(inputs=self.state_tensor, units=64, activation=tf.nn.relu, name="state_h1",
                                   reuse=None)
        state_h2 = tf.layers.dense(inputs=state_h1, units=32, activation=tf.nn.relu, name="state_h2", reuse=None)
        self.policy_tensor = tf.layers.dense(inputs=state_h2, units=self.action_size, activation=tf.nn.softmax,
                                             name="policy")

        # create critic network
        state_h1 = tf.layers.dense(inputs=self.state_tensor, units=64, activation=tf.nn.relu, name="state_h1",
                                   reuse=True)
        state_h2 = tf.layers.dense(inputs=state_h1, units=32, activation=tf.nn.relu, name="state_h2", reuse=True)
        action_h1 = tf.layers.dense(inputs=self.action_tensor, units=64, activation=tf.nn.relu, name="action_h1")
        action_h2 = tf.layers.dense(inputs=action_h1, units=32, activation=tf.nn.relu, name="action_h2")
        fc = tf.layers.dense(inputs=tf.concat([state_h2, action_h2], axis=1), units=32, activation=tf.nn.relu,
                             name="fully_connected")
        self.value_tensor = tf.layers.dense(inputs=fc, units=1, activation=None, name="value")

    def _create_abs_error_summary(self):
        self.metric_graph = tf.Graph()
        self.correct_action_tensor = tf.placeholder(tf.float32, [None, self.action_size], name="correct_action")
        self.input_action_tensor = tf.placeholder(tf.float32, [None, self.action_size], name="test_action")
        with self.metric_graph.as_default():
            self.metric_action, _ = tf.metrics.mean_absolute_error(self.correct_action_tensor,self.input_action_tensor)
            self.summary_action_error = tf.summary.scalar('Absolute Action Error', self.metric_action)

    def train_Q(self, state_batch, action_batch, targetQ_batch):
        """train value function Q"""
        _, loss = self.sess.run([self.event_Q_regression, self.summary_Q_loss], feed_dict={
            self.state_tensor: state_batch,
            self.action_tensor: action_batch,
            self.targetQ_tensor: targetQ_batch,
        })
        return loss

    def train_Pi(self, state_batch, action_batch):
        """train policy function pi"""
        grad_for_action_batch = self.sess.run(self.event_grad_for_action, feed_dict={
            self.state_tensor: state_batch,
            self.action_tensor: action_batch,
        })[0]
        self.sess.run(self.event_policy_gradient, feed_dict={
            self.state_tensor: state_batch,
            self.grad_for_action_tensor: grad_for_action_batch,
        })

    def predict_Q(self, state_batch, action_batch):
        return self.sess.run(self.value_tensor, feed_dict={
            self.state_tensor: state_batch,
            self.action_tensor: action_batch,
        })

    def predict_Pi(self, state_batch):
        """return behavior action given states"""
        return self.sess.run(self.policy_tensor, feed_dict={
            self.state_tensor: state_batch,
        })

    def metric_Pi(self, action_batch, correct_action_batch):
        """test behavior action given states & correct actions"""
        with self.metric_graph.as_default():
            return self.sess.run(self.summary_action_error, feed_dict={
                self.input_action_tensor: action_batch,
                self.correct_action_tensor: correct_action_batch,
            })

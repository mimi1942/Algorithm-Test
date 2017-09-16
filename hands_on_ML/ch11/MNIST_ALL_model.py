import tensorflow as tf
from MNIST_ALL_reader import reader

class dnn(object):
    def __init__(self):
        # Network Parameters
        n_input = 28  # MNIST data input (img shape: 28*28)
        n_steps = 28  # timesteps
        self.n_classes = 10  # MNIST total classes (0-9 digits)

        # Define model's input as tf.placeholder
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_input])
        # self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_classes])
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, self.n_classes])

        size_of_hidden = 100
        # Define parameters for 3-layers dnn model
        he_init = initializer=tf.contrib.layers.xavier_initializer()
        self.w_1, self.b_1 = tf.get_variable(name="w_1", shape=[n_input * n_input, size_of_hidden], initializer=he_init), tf.get_variable(name="b_1", shape=[size_of_hidden], initializer=he_init)
        self.w_2, self.b_2 = tf.get_variable(name="w_2", shape=[size_of_hidden, size_of_hidden], initializer=he_init), tf.get_variable(name="b_2", shape=[size_of_hidden], initializer=he_init)
        self.w_3, self.b_3 = tf.get_variable(name="w_3", shape=[size_of_hidden, size_of_hidden], initializer=he_init), tf.get_variable(name="b_3", shape=[size_of_hidden], initializer=he_init)
        self.w_4, self.b_4 = tf.get_variable(name="w_4", shape=[size_of_hidden, size_of_hidden], initializer=he_init), tf.get_variable(name="b_4", shape=[size_of_hidden], initializer=he_init)
        self.w_5, self.b_5 = tf.get_variable(name="w_5", shape=[size_of_hidden, self.n_classes], initializer=he_init), tf.get_variable(name="b_5", shape=[self.n_classes], initializer=he_init)

        # build graph for forward & backward propagation
        self.build_graph()

    def build_graph(self):
        x_v = tf.reshape(self.x, [-1, 28*28])
        # h1.shape = (batch, 100)
        h1 = tf.matmul(x_v, self.w_1) + self.b_1
        h1 = tf.nn.elu(h1)

        # h2.shape = (100, 100)
        h2 = tf.matmul(h1, self.w_2) + self.b_2
        h2 = tf.nn.elu(h2)

        # h3.shape = (100, 100)
        h3 = tf.matmul(h2, self.w_3) + self.b_3
        h3 = tf.nn.elu(h3)

        # h4.shape = (100, 100)
        h4 = tf.matmul(h3, self.w_4) + self.b_4
        h4 = tf.nn.elu(h4)

        # h5.shape = (100, 10)
        epsilon = tf.constant(value=0.0000001, shape=[self.n_classes])
        logits = tf.matmul(h4, self.w_5) + self.b_5 + epsilon
        prediction = tf.nn.softmax(logits, dim=-1)


        # We use L2 loss as cost function and average the batch's loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        self.train_op = optimizer.minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



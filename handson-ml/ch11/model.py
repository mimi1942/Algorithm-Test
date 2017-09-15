import tensorflow as tf

class dnn(object):
    def __init__(self):
        # Network Parameters
        n_input = 28 # MNIST data input (img shape: 28*28)
        n_steps = 28 # timesteps
        n_classes = 10 # MNIST total classes (0-9 digits)


        self.x = tf.placeholder(dtype=tf.float32, shape=(None, n_steps, n_input))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, n_classes))

        self.x_v = tf.reshape(self.x, [-1, 28*28])
        size_of_hidden = 100
        he_init = tf.contrib.layers.xavier_initializer()
        self.w1 = tf.get_variable(name="w1", shape=(n_input * n_input, size_of_hidden), initializer=he_init)
        self.w2 = tf.get_variable(name="w2", shape=(size_of_hidden, size_of_hidden), initializer=he_init)
        self.w3 = tf.get_variable(name="w3", shape=(size_of_hidden, size_of_hidden), initializer=he_init)
        self.w4 = tf.get_variable(name="w4", shape=(size_of_hidden, size_of_hidden), initializer=he_init)
        self.w5 = tf.get_variable(name="w5", shape=(size_of_hidden, n_classes), initializer=he_init)


        self.b1 = tf.get_variable(name="b1", shape=(size_of_hidden), initializer=he_init)
        self.b2 = tf.get_variable(name="b2", shape=(size_of_hidden), initializer=he_init)
        self.b3 = tf.get_variable(name="b3", shape=(size_of_hidden), initializer=he_init)
        self.b4 = tf.get_variable(name="b4", shape=(size_of_hidden), initializer=he_init)
        self.b5 = tf.get_variable(name="b5", shape=(n_classes), initializer=he_init)

        self.build_graph()

    def build_graph(self):
        h1 = tf.matmul(self.x_v, self.w1) + self.b1
        h1 = tf.nn.elu(h1)

        h2 = tf.matmul(h1,self.w2) + self.b2
        h2 = tf.nn.elu(h2)

        h3 = tf.matmul(h2,self.w3) + self.b3
        h3 = tf.nn.elu(h3)

        h4 = tf.matmul(h3,self.w4) + self.b4
        h4 = tf.nn.elu(h4)

        logits = tf.matmul(h4,self.w5) + self.b5
        prediction = tf.nn.softmax(logits = logits, dim=-1)


        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        self.train_op = optimizer.minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(prediction, dimension=1), tf.argmax(self.y, dimension=1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

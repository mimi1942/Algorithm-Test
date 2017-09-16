import tensorflow as tf
from model import dnn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle

# from train import save_path
with open('./save/last_save_path.dmp', 'rb') as loadf :
  save_path = pickle.load(loadf)
print(save_path)




# Loading DNN graph at Practice(3)
model = dnn()

# Define Session for running graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.all_variables())
saver.restore(sess, save_path)

mnist = input_data.read_data_sets("./MNIST_data/")
x_test = mnist.test.images[mnist.test.labels < 5].reshape([-1, 28, 28])
y_test = mnist.test.labels[mnist.test.labels < 5]

feed = {model.x: x_test, model.y: y_test}
test_acc, loss= sess.run([model.acc, model.loss], feed_dict=feed)

print("Test accuracy : %.3lf, Loss : %.3lf" % (test_acc, loss))
sess.close()
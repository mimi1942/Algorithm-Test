import tensorflow as tf
from MNIST_ALL_reader import reader
from MNIST_ALL_model import dnn
import numpy as np


# Loading data reader at Practice(2)
data_reader = reader()

# Loading DNN graph at Practice(3)
model = dnn()

# Define Session for running graph
# and initialize model's parameters
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
save_path = 0
val_queue, val_weightedAvg = [],[]
count, init_flag, best_val = 0, 0, 0

batch_size = 16
max_steps = 100000
for i in range(max_steps):

  # For each iteration, first we get batch x&y data
  x_train, y_train, _, _, _ = data_reader.next_batch(16, train=1)

  # Next, construct feed for model's placeholder
  # feed is dictionary whose key is placeholder, and value is feeded value(numpy array)
  feed = {model.x: x_train, model.y: y_train}

  # Go training via running train_op with feeded data!
  # We run simultaneously train_op(backprop) and loss value
  _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

  # print loss stat every 100 iterations
  if i%100 == 0:
    print("| steps %07d | loss: %.3lf" % (i, loss))


  # running validation process every 100 iterations
  if i%100 == 0:
    x_val, y_val, start_index, batch_indices, end_index = data_reader.next_batch(100, train=0)
    feed_val = {model.x: x_val, model.y: y_val}

    # validation_acc = sess.run(model.acc, feed_dict=feed_val)
    validation_acc= sess.run(model.acc, feed_dict=feed_val)

    if count < 9 :
      val_queue.append(validation_acc)
      count += 1
    else :
      val_queue.append(validation_acc)
      weightedAvg = np.mean(val_queue)
      if init_flag == 0 :
        best_val = weightedAvg
        save_path = saver.save(sess, "./MNIST_ALL_save/model{}.ckpt".format(count))
        init_flag = 1
      elif best_val < weightedAvg :
        best_val = weightedAvg
        save_path = saver.save(sess, "./MNIST_ALL_save/model{}.ckpt".format(count))
      else :
        best_val = best_val
      val_queue.pop(0)
      count += 1

    print("| steps %07d | Validation Accuracy: %.3lf" % (i, validation_acc))
    print("start_index {}, length of batch_indices {}, end_index {}".format(start_index, len(batch_indices), end_index))









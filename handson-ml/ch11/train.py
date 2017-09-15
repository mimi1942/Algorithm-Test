import tensorflow as tf
from reader import reader
from model import dnn
import numpy as np

data_reader = reader()
model = dnn()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

val_queue, val_weightedAvg = [], []
count, init_flag, best_val = 0, 0, 0

batch_size = 16
max_steps = 100000
for i in range(max_steps) :
     bTrain_X, bTrain_y = data_reader.next_batch(16, tOrV = "t")

     feed = {model.x: bTrain_X, model.y: bTrain_y}

     _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

     if i%100 == 0:
         bValidation_X, bValidation_y = data_reader.next_batch(100, tOrV="v")
         feed_val = {model.x: bValidation_X, model.y: bValidation_y}
         acc = sess.run(model.acc, feed_dict=feed_val)

         if count < 9 :
             val_queue.append(acc)
             count += 1
         elif init_flag == 0:
             val_queue.append(acc)
             val_weightedAvg = np.mean(val_queue)
             best_val = val_weightedAvg
             val_queue.pop(0)
             init_flag = 1
             count += 1
         else :
             val_queue.append(acc)
             val_weightedAvg = np.mean(val_queue)
             if best_val < val_weightedAvg :
                 best_val = val_weightedAvg
                 save_path = saver.save(sess, "./save/model{}.ckpt".format(count))
             else :
                 best_val = best_val
             val_queue.pop(0)
             count += 1

         print("| stops %07d | loss: %.3f | acc : %.2f" % (i, loss, acc))

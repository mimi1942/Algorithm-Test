import numpy as np
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
class reader(object):
    def __init__(self):
        mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
        self.x_train = mnist.train.images.reshape([-1, 28, 28])
        self.y_train = mnist.train.labels
        self.x_validation = mnist.test.images.reshape([-1, 28, 28])
        self.y_validation = mnist.test.labels

        print("y_train" , self.y_train.shape)

        self.start_index_train = 0
        self.start_index_val = 0

    def next_batch(self, batch_size, train):
        if train == 1:
            num_examples = self.x_train.shape[0]
            shuffle_indices = list(range(num_examples))

            if self.start_index_train == 0:
                np.random.shuffle(shuffle_indices)

            end_index = min([num_examples, self.start_index_train + batch_size])
            batch_indices = [shuffle_indices[idx] for idx in range(self.start_index_train, end_index)]

            batch_x = self.x_train[batch_indices]
            batch_y = self.y_train[batch_indices]

            if end_index == num_examples:
                self.start_index_train = 0
            else:
                self.start_index_train = end_index

            return batch_x, batch_y, self.start_index_train, batch_indices, end_index


        elif train == 0 :
            num_examples = self.y_validation.shape[0]
            shuffle_indices_val = list(range(num_examples))

            if self.start_index_val == 0:
                np.random.shuffle(shuffle_indices_val)

            end_index = min([num_examples, self.start_index_val + batch_size])
            batch_indices = [shuffle_indices_val[idx] for idx in range(self.start_index_val, end_index)]

            batch_x = self.x_train[batch_indices]
            batch_y = self.y_train[batch_indices]

            if end_index == num_examples:
                self.start_index_val = 0
            else:
                self.start_index_val = end_index

            return batch_x, batch_y, self.start_index_val, batch_indices, end_index

        else :
            print("train has only 1 or 0")





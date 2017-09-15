import csv
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class reader(object):
    def __init__(self):
        mnist = input_data.read_data_sets("data/", one_hot=True)
        self.train_X =  mnist.train.images.reshape([-1,28,28])
        self.train_y = mnist.train.labels
        self.validation_X = mnist.test.images.reshape([-1,28,28])
        self.validation_y = mnist.test.labels

        self.start_index_train = 0
        self.start_index_val = 0


    def next_batch(self, batch_size, tOrV):
        if tOrV == "t":
            num_examples = len(self.train_X)
            batchData_randIdx = range(num_examples)

            if self.start_index_train == 0:
                np.random.shuffle(batchData_randIdx)

            end_index = min([num_examples, self.start_index_train + batch_size])
            batch_indices = [ batchData_randIdx[idx] for idx in range(self.start_index_train, end_index)]

            batch_x = self.train_X[ batch_indices ]
            batch_y = self.train_y[ batch_indices ]

            if end_index == num_examples :
                self.start_index_train = 0
            else : self.start_index_train = end_index

            return batch_x, batch_y

        elif tOrV == "v" :
            num_examples = len(self.validation_X)
            batchData_randIdx = range(num_examples)

            if self.start_index_val == 0:
                np.random.shuffle(batchData_randIdx)

            end_index = min([num_examples, self.start_index_val + batch_size])
            batch_indices = [ batchData_randIdx[idx] for idx in range(self.start_index_val, end_index)]

            batch_x = self.train_X[ batch_indices ]
            batch_y = self.train_y[ batch_indices ]

            if end_index == num_examples :
                self.start_index_val = 0
            else : self.start_index_val = end_index

            return batch_x, batch_y

        else :
            print("tOrV is not 't' or 'v'")




# t = reader()

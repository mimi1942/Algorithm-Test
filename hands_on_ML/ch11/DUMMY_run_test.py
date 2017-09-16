from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import tensorflow as tf
import numpy as np

he_init = tf.contrib.layers.xavier_initializer()

class TESTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_param=0, str_param='go', random_state=None):
        """Initialize the DNNClassifier by simply storing all the hyperparameters."""
        self.num_param = num_param
        self.str_param = str_param
        self.random_state = random_state

        def _ho(self):
            print(ho)

        def fit((self, x, y, X_valid=None, y_valid=None):

            if x == 0 :




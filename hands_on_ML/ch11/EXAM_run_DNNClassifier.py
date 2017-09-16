from EXAM_DNNClassifier import DNNClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score

mnist = input_data.read_data_sets("./MNIST_data/")
x_train = mnist.train.images
y_train = mnist.train.labels
x_validation = mnist.validation.images
y_validation = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels

dnn_clf = DNNClassifier(random_state=42)
dnn_clf.fit(x_train, y_train, n_epochs=1000, X_valid=x_validation, y_valid=y_validation)

y_pred = dnn_clf.predict(x_test)
accuracy_score(y_test, y_pred)


from sklearn.model_selection import RandomizedSearchCV

def leaky_relu(alpha=0.01):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)
    return parametrized_leaky_relu

param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
    # you could also try exploring different numbers of hidden layers, different optimizers, etc.
    #"n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #"optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95)],
}

rnd_search = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,
                                fit_params={"X_valid": x_validation, "y_valid": y_validation, "n_epochs": 1000},
                                random_state=42, verbose=2)
rnd_search.fit(x_train, y_train)

import pickle
with open('./rnd_search.dmp', 'wb') as f :
  data = rnd_search
  pickle.dump(data, f)
  print("save rnd_search")
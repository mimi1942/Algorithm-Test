import numpy as np
class Perceptron(object):
    """
    파라미터 설명
    eta : 학습률(learning rate) ; float
    n_iter : Passes over the training dataset
             epoch인듯? ; int

    에트리뷰트 설명
    w_ : fitting 후의 가중치 ; 1d array
    errors_ : 매 epoch마다 오분류치 ; list
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            erros = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                erros += int(update != 0.0)
            self.errors_.append(erros)
        return self
    def net_input(self,X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

import numpy as np


class TestModel:

    def __init__(self):
        self.resolution = (100,100)

    def fit(self, X, y):
        assert X.shape[1:3] == self.resolution
        pass

    def predict_proba(self, X) -> float:
        assert X.shape[2:4] == self.resolution
        assert X.shape[1]==2, f'should get n pairs, but second dimension is {X.shape[1]}'
        return np.random.random((len(X), 2))
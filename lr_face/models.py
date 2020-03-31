import time
import numpy as np
import functools

from scipy import spatial
from lr_face.utils import resize_and_normalize


class DummyModel:
    """
    dummy model that return random scores
    """

    def __init__(self, resolution=(100, 100)):
        self.resolution = resolution

    def fit(self, X, y):
        assert X.shape[1:3] == self.resolution
        pass

    def predict_proba(self, X, ids=None):
        # assert X.shape[2:4] == self.resolution
        assert len(X[0]) == 2, f'should get n pairs, but second dimension is {X.shape[1]}'
        return np.random.random((len(X), 2))

    def __str__(self):
        return 'Dummy'


class Deepface_Lib_Model:
    """
    deepface/Face model
    """

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def predict_proba(self, X, ids):
        assert len(X)==len(ids)
        scores = []
        for id, pair in zip(ids, X):
            if id in self.cache:
                score = self.cache[id]
            else:
                score = self.score_for_pair(pair)
                self.cache[id]=score
            scores.append([score, 1-score])

        return np.asarray(scores)

    def score_for_pair(self, pair):
        img1 = resize_and_normalize(pair[0], self.model.input_shape[1:3])
        img2 = resize_and_normalize(pair[1], self.model.input_shape[1:3])
        img1_representation = self.model.predict(img1)[0, :]
        img2_representation = self.model.predict(img2)[0, :]
        score = spatial.distance.cosine(img1_representation, img2_representation)
        return score

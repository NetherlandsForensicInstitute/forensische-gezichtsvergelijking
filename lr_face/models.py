import numpy as np
import cv2

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

    def predict_proba(self, X):
        # assert X.shape[2:4] == self.resolution
        assert X.shape[1] == 2, f'should get n pairs, but second dimension is {X.shape[1]}'
        return np.random.random((len(X), 2))


class Deepface_Lib_Model:
    """
    deepface/Face model
    """

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        scores = []
        for pair in X:
            img1 = resize_and_normalize(pair[0], self.model.input_shape[1:3])
            img2 = resize_and_normalize(pair[1], self.model.input_shape[1:3])

            img1_representation = self.model.predict(img1)[0, :]
            img2_representation = self.model.predict(img2)[0, :]

            score = spatial.distance.cosine(img1_representation, img2_representation)
            scores.append([score, 1-score])

        return np.asarray(scores)
    
class InsightFace_Model:
    """
    InsightFace model. Model name : FaceModel
    
    """

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        scores = []
        for pair in X:
            img1 = cv2.imread(pair[0])
            img2 = cv2.imread(pair[1])
            img1_adapted = self.model.get_input(img1)
            img2_adapted = self.model.get_input(img2)
            img1_representation = self.model.get_feature(img1_adapted)
            img2_representation = self.model.get_feature(img2_adapted)
            score_cos = np.dot(img1_representation, img2_representation.T)
            score_eucl =np.sum(np.square(img1_representation-img2_representation))
            scores.append([score_cos, score_eucl])

        return np.asarray(scores) 

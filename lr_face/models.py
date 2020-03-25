import numpy as np

from deepface.deepface.basemodels import Facenet, FbDeepFace, VGGFace, OpenFace
from scipy import spatial
from lr_face.utils import resize


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
        assert X.shape[2:4] == self.resolution
        assert X.shape[1] == 2, f'should get n pairs, but second dimension is {X.shape[1]}'
        return np.random.random((len(X),2))


class FacenetModel:
    """
    deepface/Face model
    """

    def __init__(self, resolution=(160, 160)):
        self.resolution = resolution

    def predict_proba(self, X):
        model = Facenet.loadModel() #ergens anders laden waar hij blijft bestaan?

        scores = []
        for pair in X:
            img1 = resize(pair[0], self.resolution)
            img2 = resize(pair[1], self.resolution)

            img1_representation = model.predict(img1)[0, :]
            img2_representation = model.predict(img2)[0, :]

            score = spatial.distance.cosine(img1_representation, img2_representation)
            scores.append([1-score, score])

        return np.asarray(scores)


class OpenFaceModel:
    """
    deepface/OpenFace model
    """

    def __init__(self, resolution=(96, 96)):
        self.resolution = resolution

    def predict_proba(self, X):
        model =  OpenFace.loadModel() #ergens anders laden waar hij blijft bestaan?

        scores = []
        for pair in X:
            img1 = resize(pair[0], self.resolution)
            img2 = resize(pair[1], self.resolution)

            img1_representation = model.predict(img1)[0, :]
            img2_representation = model.predict(img2)[0, :]

            score = spatial.distance.cosine(img1_representation, img2_representation)
            scores.append([1-score, score])

        return np.asarray(scores)


class VGGFaceModel:
    """
    deepface/VVGFace model
    """

    def __init__(self, resolution=(224, 224)):
        self.resolution = resolution

    def predict_proba(self, X):
        model = VGGFace.loadModel() #ergens anders laden waar hij blijft bestaan?

        scores = []
        for pair in X:
            img1 = resize(pair[0], self.resolution)
            img2 = resize(pair[1], self.resolution)

            img1_representation = model.predict(img1)[0, :]
            img2_representation = model.predict(img2)[0, :]

            score = spatial.distance.cosine(img1_representation, img2_representation)
            scores.append([1-score, score])

        return np.asarray(scores)


class FbDeepFaceModel:
    """
    deepface/FbDeepFace
    """
    def __init__(self, resolution=(152, 152)):
        self.resolution = resolution

    def predict_proba(self, X):
        model = FbDeepFace.loadModel() #ergens anders laden waar hij blijft bestaan?

        scores = []
        for pair in X:
            img1 = resize(pair[0], self.resolution)
            img2 = resize(pair[1], self.resolution)

            img1_representation = model.predict(img1)[0, :]
            img2_representation = model.predict(img2)[0, :]

            score = spatial.distance.cosine(img1_representation, img2_representation)
            scores.append([1-score, score])

        return np.asarray(scores)
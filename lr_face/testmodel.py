import numpy as np

from deepface import DeepFace
from deepface.commons import functions
from scipy import spatial
from lr_face.utils import detectFace
from PIL import Image


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


class OpenFace:
    """
    deepface/OpenFace model
    """

    def __init__(self, resolution=(96, 96)):
        self.resolution = resolution

    def predict_proba(self, X):
        model = DeepFace.OpenFace.Model()

        scores = []

        for pair in X:
            i1 = (pair[0] * 255).astype(np.uint8)
            i2 = (pair[1] * 255).astype(np.uint8)

            # i1 = pair[0]
            # i2 = pair[1]

            # img = Image.fromarray(i1, 'RGB')
            # img.save('my.png')
            # img.show()

            # img1 = detectFace(i1, self.resolution)
            # img2 = detectFace(i2, self.resolution)

            img1_representation = model.predict(i1)[0, :]
            img2_representation = model.predict(i2)[0, :]

            score = spatial.distance.cosine(img1_representation, img2_representation)
            scores.append([1-score, score])

        return np.asarray(scores)

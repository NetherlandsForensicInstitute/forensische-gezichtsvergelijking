import numpy as np


class TestModel:
    """
    dummy model that return random scores
    """

    def __init__(self, resolution=(100, 100)):
        self.resolution = resolution

    def fit(self, X, y):
        assert X.shape[1:3] == self.resolution
        pass

    def predict_proba(self, X) -> float:
        assert X.shape[2:4] == self.resolution
        assert X.shape[1] == 2, f
        'should get n pairs, but second dimension is {X.shape[1]}'
        return np.random.random((len(X), 2))


class Insightface:
    """
    insightface model
    """

    def __init__(self, resolution=(112, 112)):
        self.resolution = resolution

    def predict_proba(self, X):
        model = insightface.face_model.FaceModel(args)

        scores = []

        for pair in X:
            img = cv2.imread(img1)
            img = model.get_input(img)
            f1 = model.get_feature(img)

            img = cv2.imread(img2)
            img = model.get_input(img)
            f2 = model.get_feature(img)

            dist12 = np.sum(np.square(f1 - f2))
            sim12 = np.dot(f1, f2.T)

        return np.asarray(scores)
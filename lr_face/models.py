import numpy as np
import tensorflow as tf
from scipy import spatial

from lr_face.utils import resize_and_normalize


class DummyModel:
    """
    Dummy model that returns random scores.
    """

    def __init__(self, resolution=(100, 100)):
        self.resolution = resolution

    def fit(self, X, y):
        assert X.shape[1:3] == self.resolution
        pass

    def predict_proba(self, X, ids=None):
        # assert X.shape[2:4] == self.resolution
        if X.shape[1] != 2:
            raise ValueError(
                f'Should get n pairs, but second dimension is {X.shape[1]}')
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
        assert len(X) == len(ids)
        scores = []
        for id, pair in zip(ids, X):
            if id in self.cache:
                score = self.cache[id]
            else:
                score = self.score_for_pair(pair)
                self.cache[id] = score
            scores.append([score, 1 - score])

        return np.asarray(scores)

    def score_for_pair(self, pair):
        img1 = resize_and_normalize(pair[0], self.model.input_shape[1:3])
        img2 = resize_and_normalize(pair[1], self.model.input_shape[1:3])
        img1_representation = self.model.predict(img1)[0, :]
        img2_representation = self.model.predict(img2)[0, :]
        score = spatial.distance.cosine(img1_representation,
                                        img2_representation)
        return score


class FinetuneModel(tf.keras.Model):
    """
    A subclass of tf.keras.Model that can be used to finetune the pre-trained
    embedding models. This new training model takes 3 inputs, namely:

        anchor: A 4D tensor containing a batch of anchor images with shape
            `(batch_size, height, width, num_channels)`.
        positive: A 4D tensor containing a batch of images of the same identity
            as the anchor image with shape `(batch_size, height, width,
            num_channels)`.
        positive: A 4D tensor containing a batch of images of a different
            identity than the anchor image with shape `(batch_size, height,
            width, num_channels)`.

    It outputs embeddings for each of the images and returns them as a single
    3D tensor of shape `(batch_size, 3, embedding_size)`, where the second
    axis represents the anchor, positive and negative images, respectively.
    The reason for returning the results as a single tensor instead of 3
    separate outputs is because all 3 are required for computing a single loss.
    """

    def __init__(self, embedding_layer: tf.keras.Model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_layer = embedding_layer
        # batch_shape, *input_shape = self.embedding_layer.input_shape

    def call(self, inputs, training=None, mask=None):
        anchor_input, positive_input, negative_input = inputs

        anchor_output = self.embedding_layer(anchor_input)
        positive_output = self.embedding_layer(positive_input)
        negative_output = self.embedding_layer(negative_input)

        return tf.stack([
            anchor_output,
            positive_output,
            negative_output
        ], axis=1)

    def save_weights(self, filepath, overwrite=True, save_format=None):
        """
        We override the `save_weights()` method of the parent class because
        whenever we want to save the weights of this finetune model we really
        only want to save the newly updated weights of the `embedding_layer`.
        This allows us to load back the saved weights directly into the
        original embedding/inference model.

        Example:
        ```
            embedding_model = ...  # Load the embedding/inference model here.
            finetune_model = FinetuneModel(embedding_model)
            finetune_model.compile(...)
            finetune_model.fit(...)
            finetune_model.save_weights('weights.h5')
            embedding_model.load_weights('weights.h5')
        ```
        """
        self.embedding_layer.save_weights(
            filepath, overwrite=overwrite, save_format=save_format)

    def load_weights(self, filepath, by_name=False):
        return self.embedding_layer.load_weights(filepath, by_name=by_name)

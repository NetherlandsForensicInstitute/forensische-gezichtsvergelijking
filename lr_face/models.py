import importlib
from enum import Enum
from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy import spatial

from lr_face.utils import resize_and_normalize, cache


class DummyScorerModel:
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
        array = np.array(X)
        if array.shape[1] != 2:
            raise ValueError(
                f'Should get n pairs, but 2nd dimension is {array.shape[1]}')
        return np.random.random(array.shape)

    def __str__(self):
        return 'Dummy'


class ScorerModel:
    def __init__(self, embedding_model: tf.keras.Model, name: str):
        self.embedding_model = embedding_model
        self.name = name
        self.cache = dict()

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
        # TODO assumes resizing is necessary
        img1 = resize_and_normalize(
            pair[0], self.embedding_model.input_shape[1:3])
        img2 = resize_and_normalize(
            pair[1], self.embedding_model.input_shape[1:3])
        # TODO assumes deepface like predict
        img1_representation = self.embedding_model.predict(img1)[0, :]
        img2_representation = self.embedding_model.predict(img2)[0, :]
        score = spatial.distance.cosine(img1_representation,
                                        img2_representation)
        return score

    def __str__(self):
        return self.name


class TripletEmbeddingModel(tf.keras.Model):
    """
    A subclass of tf.keras.Model that can be used to finetune an existing, pre-
    trained embedding model using a triplet loss.

    ```python
    embedding_model = ...
    triplet_embedding_model = TripletEmbeddingModel(embedding_model)
    triplet_embedding_model.compile(loss=TripletLoss(...))
    triplet_embedding_model.fit(...)
    ```

    When called, the `TripletEmbeddingModel` takes 3 inputs, namely:

        anchor: A 4D tensor containing a batch of anchor images with shape
            `(batch_size, height, width, num_channels)`.
        positive: A 4D tensor containing a batch of images of the same identity
            as the anchor image with shape `(batch_size, height, width,
            num_channels)`.
        negative: A 4D tensor containing a batch of images of a different
            identity than the anchor image with shape `(batch_size, height,
            width, num_channels)`.

    It outputs embeddings for each of the images and returns them as a single
    3D tensor of shape `(batch_size, 3, embedding_size)`, where the second
    axis represents the anchor, positive and negative images, respectively.
    The reason for returning the results as a single tensor instead of 3
    separate outputs is because all 3 are required for computing a single loss.
    """

    def __init__(self, embedding_model: tf.keras.Model, *args, **kwargs):
        """
        Arguments:
            embedding_model: A tf.keras.Model instance that given a batch of
                images computes their embeddings. It should accept 4D tensors
                with shape `(batch_size, height, width, num_channels)` as input
                and output 2D tensors of shape `(batch_size, embedding_size)`.
        """
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model

    def call(self, inputs, training=None, **kwargs):
        """
        Arguments:
            inputs: A tuple of 3 tensors, representing a batch of anchor,
                positive and negative images, respectively. Each of these 3
                tensors has shape `(batch_size, height, width, num_channels)`.
            training: An optional boolean flag whether the model is currently
                being trained. For a `TripletEmbeddingModel` instance this will
                almost always be True, except for maybe some test cases.

        Returns:
            A 3D tensor with the embeddings of all anchor, positive and
            negative images, with shape `(batch_size, 3, embedding_size)`.
        """
        anchor_input, positive_input, negative_input = inputs

        anchor_output = self.embedding_model(anchor_input, training)
        positive_output = self.embedding_model(positive_input, training)
        negative_output = self.embedding_model(negative_input, training)

        return tf.stack([
            anchor_output,
            positive_output,
            negative_output
        ], axis=1)

    def save_weights(self, filepath, overwrite=True, save_format=None):
        """
        We override the `save_weights()` method of the parent class because
        whenever we want to save the weights of this finetune model we really
        only want to save the newly updated weights of the `embedding_model`.
        This allows us to load back the saved weights directly into the
        original embedding/embedding model.

        Example:

        ```python
        embedding_model = ...  # Load the embedding model here.
        triplet_embedding_model = TripletEmbeddingModel(embedding_model)
        triplet_embedding_model.compile(...)
        triplet_embedding_model.fit(...)
        triplet_embedding_model.save_weights('weights.h5')
        embedding_model.load_weights('weights.h5')
        ```

        For a practical dummy example of how this works, check out the unit
        tests located at `tests/test_finetuning.py`.
        """
        self.embedding_model.save_weights(filepath, overwrite, save_format)

    def load_weights(self, filepath, by_name=False):
        """
        Since we override `save_weights()` we also have to override
        `load_weights()` to make the two compatible again.
        """
        return self.embedding_model.load_weights(filepath, by_name)


class Architecture(Enum):
    """
    This Enum can be used to define all base model architectures that we
    currently support, and to build appropriate Python objects to apply those
    models. This abstracts away the individual implementations of various
    models so that there is one standard way of loading them.

    To load the embedding model for VGGFace for example, you would use:

    ```python
    embedding_model = Architecture.VGGFACE.get_embedding_model()`
    ```

    Similarly, to load a triplet embedder model, you would use:

    ```python
    triplet_embedding_model = \
        Architecture.VGGFACE.get_triplet_embedding_model()`
    ```

    Finally, to load a scorer model, you would use:

    ```python
    scorer_model = Architecture.VGGFACE.get_scorer_model()
    ```
    """
    VGGFACE = 'VGGFace'
    FACENET = 'Facenet'
    FBDEEPFACE = 'FbDeepFace'
    OPENFACE = 'OpenFace'

    @cache
    def get_embedding_model(self) -> tf.keras.Model:
        if self.source == 'deepface':
            module_name = f'deepface.basemodels.{self.value}'
            module = importlib.import_module(module_name)
            model = module.loadModel()
            # TODO: once we start finetuning, load the finetuned weights.
            return model
        raise ValueError("Unable to load embedding model.")

    def get_triplet_embedding_model(self) -> TripletEmbeddingModel:
        return TripletEmbeddingModel(self.get_embedding_model())

    def get_scorer_model(self) -> ScorerModel:
        return ScorerModel(self.get_embedding_model(), self.value)

    @property
    def resolution(self) -> Tuple[int, int]:
        """
        Returns the expected spatial dimensions of the input image as a
        `(height, width)` tuple.

        :return: Tuple[int, int]
        """
        return self.get_embedding_model().input_shape[1:3]

    @property
    def embedding_size(self) -> int:
        """
        Returns the dimensionality of the embeddings for this architecture.

        :return: int
        """
        return self.get_embedding_model().output_shape[1]

    @property
    def source(self) -> str:
        """
        Returns a textual description of where the model comes from.

        :return: str
        """
        deepface_models = [self.VGGFACE,
                           self.FACENET,
                           self.FBDEEPFACE,
                           self.OPENFACE]
        if self in deepface_models:
            return 'deepface'
        raise ValueError("Unknown model source.")

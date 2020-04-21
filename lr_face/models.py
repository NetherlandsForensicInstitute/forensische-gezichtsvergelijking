from __future__ import annotations

import hashlib
import importlib
import os
import pickle
from enum import Enum
from typing import Tuple, List, Optional, Union

import numpy as np
import tensorflow as tf
from scipy import spatial

from lr_face.data import FaceImage, FaceTriplet, to_array
from lr_face.utils import cache
from lr_face.versioning import Version


class DummyScorerModel:
    """
    Dummy model that returns random scores.
    """

    def __init__(self, resolution=(100, 100)):
        self.resolution = resolution

    def fit(self, X, y):
        assert X.shape[1:3] == self.resolution
        pass

    def predict_proba(self, X: List['FacePair']):
        return np.random.random(size=(len(X), 2))

    def __str__(self):
        return 'Dummy'


class ScorerModel:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def predict_proba(self, X: List['FacePair']) -> np.ndarray:
        """
        Takes a list of face pairs as an argument and computes similarity
        scores between all pairs. To conform to the sklearn interface we
        return a 2D array of shape `(num_pairs, 2)`, where the first column
        is effectively ignored. The similarity scores are thus stored in the
        second column.

        :param X: List[FacePair]
        :return np.ndarray
        """
        scores = []
        cache_dir = 'embeddings'  # TODO: make dynamic?
        for pair in X:
            embedding1 = self.embedding_model.embed(pair.first, cache_dir)
            embedding2 = self.embedding_model.embed(pair.second, cache_dir)
            score = spatial.distance.cosine(embedding1, embedding2)
            scores.append([score, 1 - score])
        return np.asarray(scores)

    def __str__(self) -> str:
        return f'{self.embedding_model.name}Scorer'


class EmbeddingModel:
    def __init__(self,
                 base_model: tf.keras.Model,
                 version: Optional[Version],
                 resolution: Tuple[int, int],
                 model_dir: str,
                 name: str):
        self.base_model = base_model
        self.current_version = version
        self.resolution = resolution
        self.model_dir = model_dir
        self.name = name
        if version:
            self.load_weights(version)

    @cache
    def embed(self,
              image: FaceImage,
              cache_dir: Optional[str] = None) -> np.ndarray:
        """
        Computes an embedding of the `image`. Returns a 1D array of shape
        `(embedding_size)`.

        Optionally, a `cache_dir` may be specified where the embedding should
        be stored on disk. It can then be quickly loaded from disk later, which
        is typically faster than recomputing the embedding.

        :param image: FaceImage
        :param cache_dir: Optional[str]
        :return: np.ndarray
        """
        x = image.get_image(self.resolution, normalize=True)
        x = np.expand_dims(x, axis=0)
        if cache_dir:
            output_path = os.path.join(
                cache_dir,
                str(self),
                image.source if image.source else '_',
                f'{hashlib.md5(image.path.encode()).hexdigest()}.obj'
            )

            # If the embedding has been cached before, load and return it.
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    return pickle.load(f)

            # If the embedding has not been cached to disk yet: compute the
            # embedding, cache it afterwards and then return the result.
            embedding = self.base_model.predict(x)[0]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(embedding, f)
            return embedding

        # If no `output_dir` is specified, we simply compute the embedding.
        return self.base_model.predict(x)[0]

    def load_weights(self, version: Version):
        weights_path = self.get_weights_path(version)
        if not os.path.exists(weights_path):
            raise ValueError(f"Unable to load weights for version {version}: "
                             f"Could not find weights at {weights_path}")
        self.base_model.load_weights(weights_path)
        self.current_version = version

    def save_weights(self, version: Version):
        weights_path = self.get_weights_path(version)
        self.base_model.save_weights(weights_path, overwrite=False)
        self.current_version = version
        print(f"Saved weights for version {version} to {weights_path}")

    def get_weights_path(self, version: Version):
        filename = version.append_to_filename('weights.h5')
        return os.path.join(self.model_dir, filename)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and self.name == other.name \
               and self.current_version == other.current_version

    def __str__(self):
        if self.current_version:
            return f'{self.name}_{self.current_version}'
        return self.name


class TripletEmbeddingModel(EmbeddingModel):
    """
    A subclass of EmbeddingModel that can be used to finetune an existing,
    pre-trained embedding model using a triplet loss.

    ```python
    triplet_embedding_model = TripletEmbeddingModel(...)
    triplet_embedding_model.keras().compile(loss=TripletLoss(...))
    triplet_embedding_model.keras().fit(...)
    triplet_embedding_model.save()
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

    def train(self,
              triplets: List[FaceTriplet],
              batch_size: int,
              num_epochs: int,
              optimizer: tf.keras.optimizers.Optimizer,
              loss: tf.keras.losses.Loss):
        trainable_model = self._build_trainable_model()
        trainable_model.compile(optimizer, loss)

        anchors, positives, negatives = to_array(
            triplets,
            resolution=self.resolution,
            normalize=True
        )

        # The triplet loss that is used to train the model actually does not need
        # any ground truth labels, since it simply aims to maximize the difference
        # in distances to the anchor embedding between positive and negative query
        # images. However, Keras' Loss interface still needs a `y_true` that has
        # the same first dimension as the `y_pred` output by the model. That's why
        # we create a dummy "ground truth" of the same length.
        inputs = [anchors, positives, negatives]
        y = np.zeros(shape=(anchors.shape[0], 1))
        trainable_model.fit(
            x=inputs,
            y=y,
            batch_size=batch_size,
            epochs=num_epochs
        )

    def _build_trainable_model(self) -> tf.keras.Model:
        input_shape = (*self.resolution, 3)
        anchors = tf.keras.layers.Input(input_shape)
        positives = tf.keras.layers.Input(input_shape)
        negatives = tf.keras.layers.Input(input_shape)

        anchor_embeddings = self.base_model(anchors)
        positive_embeddings = self.base_model(positives)
        negative_embeddings = self.base_model(negatives)

        output = tf.stack([
            anchor_embeddings,
            positive_embeddings,
            negative_embeddings
        ], axis=1)

        return tf.keras.Model([anchors, positives, negatives], output)


class Architecture(Enum):
    """
    This Enum can be used to define all base model architectures that we
    currently support, and to build appropriate Python objects to apply those
    models. This abstracts away the individual implementations of various
    models so that there is one standard way of loading them.

    To load the embedding model for VGGFace for example, you would use:

    ```python
    embedding_model = Architecture.VGGFACE.get_embedding_model(version)`
    ```

    Similarly, to load a triplet embedder model, you would use:

    ```python
    triplet_embedding_model = \
        Architecture.VGGFACE.get_triplet_embedding_model(version)`
    ```

    Finally, to load a scorer model, you would use:

    ```python
    scorer_model = Architecture.VGGFACE.get_scorer_model(version)
    ```
    """
    VGGFACE = 'VGGFace'
    FACENET = 'Facenet'
    FBDEEPFACE = 'FbDeepFace'
    OPENFACE = 'OpenFace'

    @cache
    def get_base_model(self):
        if self.source == 'deepface':
            module_name = f'deepface.basemodels.{self.value}'
            module = importlib.import_module(module_name)
            return module.loadModel()
        raise ValueError("Unable to load base model")

    def get_embedding_model(self,
                            version: Optional[Union[str, Version]] = None,
                            use_triplets: bool = False) -> EmbeddingModel:
        base_model = self.get_base_model()
        os.makedirs(self.model_dir, exist_ok=True)
        cls = TripletEmbeddingModel if use_triplets else EmbeddingModel
        if isinstance(version, str):
            version = Version.from_string(version)
        embedding_model = cls(
            base_model,
            version,
            self.resolution,
            self.model_dir,
            name=self.value
        )
        return embedding_model

    def get_triplet_embedding_model(
            self,
            version: Optional[Union[str, Version]] = None
    ) -> TripletEmbeddingModel:
        embedding_model = self.get_embedding_model(version, use_triplets=True)
        if not isinstance(embedding_model, TripletEmbeddingModel):
            raise ValueError('')
        return embedding_model

    def get_scorer_model(
            self,
            version: Optional[Union[str, Version]] = None
    ) -> ScorerModel:
        embedding_model = self.get_embedding_model(version, use_triplets=False)
        return ScorerModel(embedding_model)

    def get_latest_version(self) -> Version:
        try:
            model_files = os.listdir(self.model_dir)
        except FileNotFoundError:
            model_files = []
        if not model_files:
            raise ValueError(
                f'No {self.value} models have been saved yet')
        return max(map(Version.from_filename, model_files))

    @property
    def model_dir(self):
        """
        Returns the directory where models for this architecture are stored.

        TODO: make dynamic? (optional)

        :return: str
        """
        return os.path.join('models', self.value)

    @property
    def resolution(self) -> Tuple[int, int]:
        """
        Returns the expected spatial dimensions of the input image as a
        `(height, width)` tuple.

        :return: Tuple[int, int]
        """
        return self.get_base_model().input_shape[1:3]

    @property
    def embedding_size(self) -> int:
        """
        Returns the dimensionality of the embeddings for this architecture.

        :return: int
        """
        return self.get_base_model().output_shape[1]

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

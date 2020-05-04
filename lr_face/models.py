from __future__ import annotations

import hashlib
import importlib
import math
import os
import pickle
import random
import re
from enum import Enum
from typing import Tuple, List, Optional, Union, Callable

import numpy as np
import tensorflow as tf
from scipy import spatial
from tensorflow.python.keras.layers import Flatten, Dense, Input

from lr_face.data import FaceImage, FacePair, FaceTriplet, to_array
from lr_face.losses import TripletLoss
from lr_face.utils import cache
from lr_face.versioning import Tag

EMBEDDINGS_DIR = 'embeddings'
WEIGHTS_DIR = 'weights'


class DummyModel(tf.keras.Sequential):
    """
    A dummy model that takes RGB images with dimensions 100x100 as input and
    outputs random embeddings with dimensionality 100.
    """

    def __init__(self):
        super().__init__([Input(shape=(100, 100, 3)), Flatten(), Dense(100)])


class ScorerModel:
    """
    A wrapper around an `EmbeddingModel` that converts the embeddings of image
    pairs into (dis)similarity scores.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def predict_proba(self, X: List[FacePair]) -> np.ndarray:
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
        cache_dir = EMBEDDINGS_DIR
        for pair in X:
            embedding1 = self.embedding_model.embed(pair.first, cache_dir)
            embedding2 = self.embedding_model.embed(pair.second, cache_dir)
            score = spatial.distance.cosine(embedding1, embedding2)
            scores.append([score, 1 - score])
        return np.asarray(scores)

    def __str__(self) -> str:
        name = self.embedding_model.name
        tag = self.embedding_model.tag
        if tag:
            return f'{name}Scorer_{tag}'
        return name


class EmbeddingModel:
    def __init__(self,
                 model: tf.keras.Model,
                 tag: Optional[Tag],
                 resolution: Tuple[int, int],
                 model_dir: str,
                 name: str):
        self.model = model
        self.tag = tag
        self.resolution = resolution
        self.model_dir = model_dir
        self.name = name
        if tag:
            self.load_weights(tag)

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
                str(self).replace(':', '-'),  # Windows compatibility
                image.source or '_',
                f'{hashlib.md5(image.path.encode()).hexdigest()}.obj'
            )

            # If the embedding has been cached before, load and return it.
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    return pickle.load(f)

            # If the embedding has not been cached to disk yet: compute the
            # embedding, cache it afterwards and then return the result.
            embedding = self.model.predict(x)[0]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(embedding, f)
            return embedding

        # If no `cache_dir` is specified, we simply compute the embedding.
        return self.model.predict(x)[0]

    def load_weights(self, tag: Tag):
        weights_path = self.get_weights_path(tag)
        if not os.path.exists(weights_path):
            raise ValueError(f"Unable to load weights for {tag}: "
                             f"Could not find weights at {weights_path}")
        self.model.load_weights(weights_path)
        self.tag = tag

    def save_weights(self, tag: Tag):
        weights_path = self.get_weights_path(tag)
        self.model.save_weights(weights_path, overwrite=False)
        self.tag = tag
        print(f"Saved weights for {tag} to {weights_path}")

    def get_weights_path(self, tag: Tag):
        filename = tag.append_to_filename('weights.h5')
        return os.path.join(self.model_dir, filename)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and self.name == other.name \
               and self.tag == other.tag

    def __str__(self):
        if self.tag:
            return f'{self.name}_{self.tag}'
        return self.name


class TripletEmbeddingModel(EmbeddingModel):
    """
    A subclass of EmbeddingModel that can be used to finetune an existing,
    pre-trained embedding model using a triplet loss.
    """

    def train(self,
              triplets: List[FaceTriplet],
              batch_size: int,
              num_epochs: int,
              optimizer: tf.keras.optimizers.Optimizer,
              loss: TripletLoss,
              augmenter: Optional[Callable[[np.ndarray], np.ndarray]] = None):

        def generator():
            while True:
                data = random.sample(triplets, len(triplets))
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    inputs = to_array(
                        batch,
                        resolution=self.resolution,
                        normalize=True,
                        augmenter=augmenter
                    )
                    y = np.zeros(shape=(len(batch), 1))
                    yield inputs, y

        trainable_model = self.build_trainable_model()
        trainable_model.compile(optimizer, loss)

        steps_per_epoch = int(math.ceil(len(triplets) / batch_size))
        trainable_model.fit_generator(
            generator=generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            workers=0  # Without this we get segmentation faults or OOM errors.
        )

    def build_trainable_model(self) -> tf.keras.Model:
        input_shape = (*self.resolution, 3)
        anchors = tf.keras.layers.Input(input_shape)
        positives = tf.keras.layers.Input(input_shape)
        negatives = tf.keras.layers.Input(input_shape)

        anchor_embeddings = self.model(anchors)
        positive_embeddings = self.model(positives)
        negative_embeddings = self.model(negatives)

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
    embedding_model = Architecture.VGGFACE.get_embedding_model("0.0.1")`
    ```

    Similarly, to load a triplet embedder model, you would use:

    ```python
    triplet_embedding_model = \
        Architecture.VGGFACE.get_triplet_embedding_model("0.0.1")`
    ```

    Finally, to load a scorer model, you would use:

    ```python
    scorer_model = Architecture.VGGFACE.get_scorer_model("0.0.1")
    ```
    """
    DUMMY = 'Dummy'
    VGGFACE = 'VGGFace'
    FACENET = 'Facenet'
    FBDEEPFACE = 'FbDeepFace'
    OPENFACE = 'OpenFace'
    ARCFACE = 'ArcFace'

    @cache
    def get_model(self):
        # TODO: unify cases
        if self.source == 'deepface':
            module_name = f'deepface.basemodels.{self.value}'
            module = importlib.import_module(module_name)
            return module.loadModel()
        elif self.source == 'insightface':
            module_name = f'insightface.{self.value}'
            module = importlib.import_module(module_name)
            return module.loadModel()
        if self == self.DUMMY:
            return DummyModel()
        raise ValueError("Unable to load base model")

    def get_embedding_model(self,
                            tag: Optional[Union[str, Tag]] = None,
                            use_triplets: bool = False) -> EmbeddingModel:
        if isinstance(tag, str):
            tag = Tag(tag)
        base_model = self.get_model()
        os.makedirs(self.model_dir, exist_ok=True)
        cls = TripletEmbeddingModel if use_triplets else EmbeddingModel
        return cls(
            base_model,
            tag,
            self.resolution,
            self.model_dir,
            name=self.value
        )

    def get_triplet_embedding_model(
            self,
            tag: Optional[Union[str, Tag]] = None
    ) -> TripletEmbeddingModel:
        embedding_model = self.get_embedding_model(tag, use_triplets=True)
        if not isinstance(embedding_model, TripletEmbeddingModel):
            raise ValueError(f'Expected `TripletEmbeddingModel`, '
                             f'but got {type(embedding_model)}')
        return embedding_model

    def get_scorer_model(
            self,
            tag: Optional[Union[str, Tag]] = None
    ) -> ScorerModel:
        embedding_model = self.get_embedding_model(tag, use_triplets=False)
        return ScorerModel(embedding_model)

    def get_latest_version(self, tag: Union[str, Tag]) -> int:
        if isinstance(tag, str):
            tag = Tag(tag)
        try:
            def filter_func(filename):
                return bool(re.search(rf'{tag.name}-\d+\.\w+$', filename))

            model_files = list(filter(filter_func, os.listdir(self.model_dir)))
        except FileNotFoundError:
            model_files = []
        if not model_files:
            raise ValueError(f'No {self.value} weights have been saved yet')
        return max(map(Tag.get_version_from_filename, model_files))

    @property
    def model_dir(self):
        """
        Returns the directory where models for this architecture are stored.

        :return: str
        """
        return os.path.join(WEIGHTS_DIR, self.value)

    @property
    def resolution(self) -> Tuple[int, int]:
        """
        Returns the expected spatial dimensions of the input image as a
        `(height, width)` tuple.

        :return: Tuple[int, int]
        """
        return self.get_model().input_shape[1:3]

    @property
    def embedding_size(self) -> int:
        """
        Returns the dimensionality of the embeddings for this architecture.

        :return: int
        """
        return self.get_model().output_shape[1]

    @property
    def source(self) -> Optional[str]:
        """
        Returns a textual description of where the model comes from, or None if
        no source can be determined.

        :return: Optional[str]
        """
        deepface_models = [self.VGGFACE,
                           self.FACENET,
                           self.FBDEEPFACE,
                           self.OPENFACE]
        insightface_models = [self.ARCFACE]

        if self in deepface_models:
            return 'deepface'
        if self in insightface_models:
            return 'insightface'
        return None

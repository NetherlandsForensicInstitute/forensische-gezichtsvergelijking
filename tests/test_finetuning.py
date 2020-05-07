from typing import List

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from lr_face.data import FaceImage, DummyFaceImage, FacePair, make_pairs, \
    FaceTriplet, make_triplets
from lr_face.losses import TripletLoss
from lr_face.models import TripletEmbeddingModel, EmbeddingModel, Architecture
from lr_face.utils import fix_tensorflow_rtx
from lr_face.versioning import Tag
from tests.src.util import scratch_dir, get_tests_path

fix_tensorflow_rtx()

SCRATCH_DIR = get_tests_path('scratch/finetuning')


def get_dummy_base_model(input_shape) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
    ])
    return model


def get_dummy_embedding_model(
        base_model: tf.keras.Model,
        use_triplets: bool = False
) -> EmbeddingModel:
    cls = TripletEmbeddingModel if use_triplets else EmbeddingModel
    return cls(
        base_model,
        tag=None,
        resolution=base_model.input_shape[1:3],
        model_dir=SCRATCH_DIR,
        name='DUMMY-TRIPLET-EMBEDDING-MODEL'
    )


def get_dummy_triplet_embedding_model(
        base_model: tf.keras.Model
) -> TripletEmbeddingModel:
    embedding_model = get_dummy_embedding_model(base_model, use_triplets=True)
    if not isinstance(embedding_model, TripletEmbeddingModel):
        raise ValueError(
            f'Expected `TripletEmbeddingModel` instance, '
            f'but got {type(embedding_model)}'
        )
    return embedding_model


@pytest.fixture()
def scratch():
    yield from scratch_dir(SCRATCH_DIR)


@pytest.fixture
def dummy_images() -> List[FaceImage]:
    ids = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    return [DummyFaceImage(path='', identity=f'TEST-{idx}') for idx in ids]


@pytest.fixture
def dummy_pairs() -> List[FacePair]:
    ids = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    images = [DummyFaceImage(path='', identity=f'TEST-{idx}') for idx in ids]
    return make_pairs(images)


@pytest.fixture
def dummy_triplets() -> List[FaceTriplet]:
    ids = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    images = [DummyFaceImage(path='', identity=f'TEST-{idx}') for idx in ids]
    return make_triplets(images)


def finetune_and_embed(
        triplet_embedding_model: TripletEmbeddingModel,
        triplets: List[FaceTriplet]
):
    dummy_image = triplets[0].anchor
    y_original = triplet_embedding_model.embed(dummy_image)

    batch_size = 1
    num_epochs = 1
    optimizer = Adam(learning_rate=3e-4)
    loss = TripletLoss(alpha=0.5, force_normalization=True)
    triplet_embedding_model.train(
        triplets,
        batch_size,
        num_epochs,
        optimizer,
        loss
    )

    tag = Tag('tag:1')
    triplet_embedding_model.save_weights(tag)
    y_trained = triplet_embedding_model.embed(dummy_image)

    triplet_embedding_model.load_weights(tag)
    y_restored = triplet_embedding_model.embed(dummy_image)
    return y_original, y_trained, y_restored


def test_can_load_weights_from_training_model_into_embedding_model(
        dummy_triplets,
        scratch
):
    input_shape = (10, 10, 3)
    base_model = get_dummy_base_model(input_shape)
    triplet_embedding_model = get_dummy_triplet_embedding_model(base_model)

    y_original, y_trained, y_restored = finetune_and_embed(
        triplet_embedding_model,
        dummy_triplets
    )

    assert not np.all(y_original == y_trained)
    assert np.all(y_trained == y_restored)


def test_can_finetune_vggface(
        dummy_triplets,
        scratch
):
    tem = Architecture.VGGFACE.get_triplet_embedding_model()
    tem.model_dir = scratch
    y_original, y_trained, y_restored = finetune_and_embed(tem, dummy_triplets)

    assert not np.all(y_original == y_trained)
    assert np.all(y_trained == y_restored)


def test_can_finetune_openface(
        dummy_triplets,
        scratch
):
    tem = Architecture.OPENFACE.get_triplet_embedding_model()
    tem.model_dir = scratch
    y_original, y_trained, y_restored = finetune_and_embed(tem, dummy_triplets)

    assert not np.all(y_original == y_trained)
    assert np.all(y_trained == y_restored)


def test_can_finetune_fbdeepface(
        dummy_triplets,
        scratch
):
    tem = Architecture.FBDEEPFACE.get_triplet_embedding_model()
    tem.model_dir = scratch
    y_original, y_trained, y_restored = finetune_and_embed(tem, dummy_triplets)

    assert not np.all(y_original == y_trained)
    assert np.all(y_trained == y_restored)


def test_can_finetune_facenet(
        dummy_triplets,
        scratch
):
    tem = Architecture.FACENET.get_triplet_embedding_model()
    tem.model_dir = scratch
    y_original, y_trained, y_restored = finetune_and_embed(tem, dummy_triplets)

    assert not np.all(y_original == y_trained)
    assert np.all(y_trained == y_restored)


def test_can_finetune_arcface(
        dummy_triplets,
        scratch
):
    tem = Architecture.ARCFACE.get_triplet_embedding_model()
    tem.model_dir = scratch
    y_original, y_trained, y_restored = finetune_and_embed(tem, dummy_triplets)

    assert not np.all(y_original == y_trained)
    assert np.all(y_trained == y_restored)

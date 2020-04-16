import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from lr_face.losses import TripletLoss
from lr_face.models import TripletEmbeddingModel
from tests.src.util import scratch_dir


def get_dummy_embedding_model(batch_input_shape) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
    ])
    model.build(input_shape=batch_input_shape)
    return model


def get_dummy_triplet_embedding_model(embedding_model: tf.keras.Model) -> \
        TripletEmbeddingModel:
    triplet_embedding_model = TripletEmbeddingModel(embedding_model)
    triplet_embedding_model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=TripletLoss(alpha=0.5, force_normalization=True)
    )
    return triplet_embedding_model


@pytest.fixture()
def scratch():
    yield from scratch_dir('scratch/finetuning')


def test_can_load_weights_from_training_model_into_embedding_model(scratch):
    """
    This methods tests whether or not it is possible to save the weights of a
    TripletEmbeddingModel and restore them into an embedding model.

    """
    batch_size = 1
    batch_input_shape = (batch_size, 10, 10, 3)
    embedding_model = get_dummy_embedding_model(batch_input_shape)
    triplet_embedding_model = get_dummy_triplet_embedding_model(embedding_model)
    x = [np.random.normal(size=(batch_size, 10, 10, 3)),
         np.random.normal(size=(batch_size, 10, 10, 3)),
         np.random.normal(size=(batch_size, 10, 10, 3))]
    triplet_embedding_model.fit(x=x,
                         y=np.zeros(shape=(batch_size,)),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=0)
    weights_path = os.path.join(scratch, 'weights.h5')
    triplet_embedding_model.save_weights(weights_path)

    y1 = embedding_model.predict(x[0])

    # Reload a blank embedding model and see if we can load our weights into it
    embedding_model = get_dummy_embedding_model(batch_input_shape)
    embedding_model.load_weights(weights_path)
    y2 = embedding_model.predict(x[0])

    assert np.all(y1 == y2)

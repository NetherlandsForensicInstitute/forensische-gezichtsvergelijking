import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from lr_face.losses import TripletLoss
from lr_face.models import FinetuneModel
from tests.src.util import scratch_dir


def get_dummy_embedding_model(batch_input_shape) -> tf.keras.Model:
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu')])
    model.build(input_shape=batch_input_shape)
    return model


def get_dummy_finetune_model(embedding_model: tf.keras.Model) -> FinetuneModel:
    finetune_model = FinetuneModel(embedding_model)
    finetune_model.compile(optimizer=Adam(learning_rate=3e-4),
                           loss=TripletLoss(alpha=0.5))
    return finetune_model


@pytest.fixture()
def scratch():
    yield from scratch_dir('scratch/finetuning')


def test_can_load_weights_from_training_model_into_embedding_model(scratch):
    """
    This methods tests whether or not it is possible to save the weights of a
    FinetuneModel and restore them into an embedding model.

    """
    batch_size = 1
    batch_input_shape = (batch_size, 10, 10, 3)
    embedding_model = get_dummy_embedding_model(batch_input_shape)
    finetune_model = get_dummy_finetune_model(embedding_model)
    x = [np.random.normal(size=(batch_size, 10, 10, 3)),
         np.random.normal(size=(batch_size, 10, 10, 3)),
         np.random.normal(size=(batch_size, 10, 10, 3))]
    finetune_model.fit(x=x,
                       y=np.zeros(shape=(batch_size,)),
                       batch_size=batch_size,
                       epochs=1,
                       verbose=0)
    weights_path = os.path.join(scratch, 'weights.h5')
    finetune_model.save_weights(weights_path)

    y1 = embedding_model.predict(x[0])

    # Reload a blank embedding model and see if we can load our weights into it
    embedding_model = get_dummy_embedding_model(batch_input_shape)
    embedding_model.load_weights(weights_path)
    y2 = embedding_model.predict(x[0])

    assert np.all(y1 == y2)

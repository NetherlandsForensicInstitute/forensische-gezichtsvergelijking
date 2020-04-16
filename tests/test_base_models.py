import numpy as np
import pytest
import tensorflow as tf

from lr_face.models import BaseModel


@pytest.mark.skip(reason="Fails on Github because model weights don't exist")
def test_get_embedding_models():
    """
    Tests whether no exceptions are thrown when we try to load the embedding
    models for each defined BaseModel.
    """
    for base_model in BaseModel:
        base_model.get_embedding_model()


@pytest.mark.skip(reason="Not all BaseModel outputs have been normalized yet")
def test_embedding_models_return_normalized_embeddings():
    """
    Tests whether the embedding model of each `BaseModel` returns embeddings
    that are L2-normalized.
    """
    for base_model in BaseModel:
        embedding_model: tf.keras.Model = base_model.get_embedding_model()
        batch_input_shape = embedding_model.input_shape
        x = np.random.normal(size=(2, *batch_input_shape[1:]))
        embedding = embedding_model.predict(x)
        squared_sum = np.sum(np.square(embedding), axis=1)
        assert np.all((0.999 < squared_sum) & (squared_sum < 1.001)), \
            f"{base_model.value}'s embeddings are not properly L2-normalized"


@pytest.mark.skip(reason="Fails on Github because model weights don't exist")
def test_get_triplet_embedding_models():
    """
    Tests whether no exceptions are thrown when we try to load the finetune
    models for each defined BaseModel.
    """
    for base_model in BaseModel:
        base_model.get_triplet_embedding_model()

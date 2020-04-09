import numpy as np
import pytest
import tensorflow as tf

from lr_face.losses import TripletLoss


def test_triplet_loss_with_degenerate_embeddings():
    alpha = 0.5
    loss_func = TripletLoss(alpha, force_normalization=False)

    # Create an artificial batch of 2 instances with an embedding size of 4
    # where all images have been mapped to the same embedding. This should
    # cause our loss to be equal to `alpha`.
    y_pred = np.array([
        [
            [0.1, 0.1, 0.1, 0.1],  # Anchor
            [0.1, 0.1, 0.1, 0.1],  # Positive
            [0.1, 0.1, 0.1, 0.1],  # Negative
        ],
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
        ]
    ])
    loss = loss_func.call(None, y_pred)
    assert loss.shape == (2,)
    assert all(loss == alpha)


def test_triplet_loss_with_good_embeddings():
    alpha = 0.5
    loss_func = TripletLoss(alpha, force_normalization=False)

    # Create an artificial batch of 2 instances with an embedding size of 4
    # where the distance between the anchor and positive embeddings is much
    # smaller than the distance between the anchor and negative embeddings.
    y_pred = np.array([
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.5],
        ],
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.5],
        ]
    ])
    loss = loss_func.call(None, y_pred)
    assert loss.shape == (2,)
    assert all(loss == 0)


def test_triplet_loss_with_unnormalized_embeddings_raises_exception():
    loss_func = TripletLoss(0.5, force_normalization=True)
    y_pred = np.array([
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5, 0.5],
        ],
    ])
    with pytest.raises(tf.errors.InvalidArgumentError):
        loss_func.call(None, y_pred)

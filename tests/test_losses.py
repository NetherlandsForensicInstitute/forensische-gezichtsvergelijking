import numpy as np

from lr_face.losses import TripletLoss


def test_triplet_loss_with_degenerate_embeddings():
    alpha = 0.5
    loss_func = TripletLoss(alpha)

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
    assert loss.shape == (2, )
    assert all(loss == alpha)


def test_triplet_loss_with_good_embeddings():
    alpha = 0.5
    loss_func = TripletLoss(alpha)

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
    assert loss.shape == (2, )
    assert all(loss == 0)

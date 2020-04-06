from functools import partial

import tensorflow as tf
from tensorflow.keras.losses import Loss


class TripletLoss(Loss):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        # TODO: make the distance function dynamic.
        self.distance_func = partial(tf.norm, ord='euclidean')

    def call(self, y_true, y_pred):
        """
        See https://en.wikipedia.org/wiki/Triplet_loss for more info.

        Arguments:
            y_true: Ignored
            y_pred: A 3D tensor of shape `(batch_size, embedding_size, 3)`.
                It contains the embeddings for each combination of anchor,
                positive and negative image in the batch, respectively, which
                is represented by the last axis of size 3. Make sure that each
                embedding vector is L2-normalized for proper convergence.
        """
        anchor, positive, negative = tf.split(y_pred, 3, axis=-1)
        positive_distance = self.distance_func(anchor - positive)
        negative_distance = self.distance_func(anchor - negative)
        return tf.maximum(
            positive_distance - negative_distance + self.alpha, 0)

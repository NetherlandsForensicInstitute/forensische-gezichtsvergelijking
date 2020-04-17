import tensorflow as tf
from tensorflow.keras.losses import Loss


class TripletLoss(Loss):
    def __init__(self,
                 alpha: float,
                 force_normalization: bool = True,
                 **kwargs):
        """
        Instantiate a triplet loss instance.

        Arguments:
            alpha: A margin between positive and negative pairs. Mathematically
                for the loss to be 0, the following must hold:

                    `dist(anchor, negative) - dist(anchor, positive) > alpha`

                Where `dist(a, b)` is a distance measure between two embeddings
                `a` and `b`. When `alpha` is 0, the loss can theoretically be
                minimized by mapping every image to the same embedding.
            force_normalization: A boolean flag indicating whether the `call()`
                method should explicitly enforce that its input embeddings are
                properly l2-normalized.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.force_normalization = force_normalization

    def call(self, y_true, y_pred):
        """
        Keras implementation of the triplet loss. If `self.force_normalization`
        is True, a `tf.errors.InvalidArgumentError` is raised if some of the
        embeddings are not properly L2-normalized.

        See https://en.wikipedia.org/wiki/Triplet_loss for more info.

        Arguments:
            y_true: Ignored, but required due to `Loss.call()` interface.
            y_pred: A 3D tensor of shape `(batch_size, 3, embedding_size)`.
                It contains the embeddings for each combination of anchor,
                positive and negative image in the batch, respectively, which
                is represented by the last axis of size 3. Make sure that each
                embedding vector is L2-normalized for proper convergence.

        Returns:
            A 1D tensor with shape `(batch_size)` containing the loss for each
            triplet in the batch.
        """

        if self.force_normalization:
            squared_sum = tf.reduce_sum(tf.square(y_pred), axis=2)
            check = tf.reduce_all(
                tf.logical_and(squared_sum > 0.999, squared_sum < 1.001))
            tf.assert_equal(check, True)

        anchor, positive, negative = tf.split(y_pred, 3, axis=1)

        # Squeeze the 2nd dimension of each of the split embedding
        # tensors. Resulting shape: `(batch_size, embedding_size)`.
        anchor = tf.squeeze(anchor, axis=1)
        positive = tf.squeeze(positive, axis=1)
        negative = tf.squeeze(negative, axis=1)

        # Compute the euclidean distance between the anchor and the two
        # query images. Resulting shape: `(batch_size)`.
        positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        # The loss is then given by the difference in distance with a minimum
        # of 0.
        loss = tf.maximum(
            positive_distance - negative_distance + self.alpha, 0.)
        return loss

    def get_config(self):
        return {**super().get_config(), 'alpha': self.alpha}

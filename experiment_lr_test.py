import random
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from experiment_finetuning import make_triplets_v2, LfwDataset
from finetuning import BaseModel
from lr_face.data_providers import Triplet
from lr_face.losses import TripletLoss
from lr_face.models import TripletEmbedder
from lr_face.utils import fix_tensorflow_rtx

# Needed to make TensorFlow 2.x work with RTX Nvidia cards.
fix_tensorflow_rtx()


def get_lr(step: int, step_size: float, initial_lr: float) -> float:
    """
    Computes the learning rate for the given `step`.

    :param step: int
    :param step_size: float
    :param initial_lr: float
    :return: float
    """
    return initial_lr * (step_size ** step)


def lr_test(model: TripletEmbedder, triplets: List[Triplet]):
    batch_size = 4

    def generator():
        idx = 0
        while True:
            start = idx * batch_size
            end = start + batch_size
            if end > len(triplets):
                random.shuffle(triplets)
                idx = 0
                continue
            anchors, positives, negatives = zip(*[(
                triplet.anchor,
                triplet.positive,
                triplet.negative
            ) for triplet in triplets[start:end]])
            x = [np.stack(anchors), np.stack(positives), np.stack(negatives)]
            y = np.zeros(shape=(batch_size, 1))
            yield x, y
            idx += 1

    initial_lr = 1e-8

    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss=TripletLoss(alpha=0.5, force_normalization=False),
    )

    schedule = partial(get_lr, step_size=10 ** 0.1, initial_lr=initial_lr)

    epochs = 100
    callback = tf.keras.callbacks.LearningRateScheduler(schedule)
    history = model.fit_generator(generator(),
                                  steps_per_epoch=1,
                                  epochs=epochs,
                                  callbacks=[callback])

    plt.plot(list(map(schedule, range(epochs))), history.history['loss'])
    plt.xscale('log')
    plt.show()


def main():
    base_model = BaseModel.VGGFACE
    triplet_embedder = base_model.load_triplet_embedder()
    dataset = LfwDataset()
    triplets = make_triplets_v2(dataset.data)
    lr_test(triplet_embedder, triplets)


if __name__ == '__main__':
    main()

import math
import random
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from lr_face.data import FaceTriplet, make_triplets, to_array, \
    EnfsiDataset
from lr_face.losses import TripletLoss
from lr_face.models import TripletEmbeddingModel, BaseModel
from lr_face.utils import fix_tensorflow_rtx

# Needed to make TensorFlow 2.x work with RTX Nvidia cards.
fix_tensorflow_rtx()

BATCH_SIZE = 4
INITIAL_LEARNING_RATE = 1e-8
STEP_SIZE = 10 ** 0.1
MAX_LEARNING_RATE = .1
BASE_MODEL = BaseModel.VGGFACE
DATASET = EnfsiDataset(years=[2011, 2012, 2013, 2017])
OPTIMIZER = Adam
FOCAL_LOSS_ALPHA = 0.5


def get_learning_rate(step: int, step_size: float, initial: float) -> float:
    """
    Computes the learning rate for the given `step`.

    :param step: int
    :param step_size: float
    :param initial: float
    :return: float
    """
    return initial * (step_size ** step)


def lr_test(model: TripletEmbeddingModel, triplets: List[FaceTriplet]):
    def batches():
        random.shuffle(triplets)
        for i in range(0, len(triplets), BATCH_SIZE):
            batch = triplets[i:i + BATCH_SIZE]
            if len(batch) != BATCH_SIZE:
                break
            x = to_array(batch, resolution=BASE_MODEL.resolution)
            y = np.zeros(shape=(BATCH_SIZE, 1))
            yield x, y

    def generator():
        while True:
            yield from batches()

    # Compile the model.
    optimizer = OPTIMIZER(learning_rate=INITIAL_LEARNING_RATE)
    loss = TripletLoss(alpha=FOCAL_LOSS_ALPHA)
    model.compile(optimizer, loss)

    # Create the learning rate scheduler.
    schedule = partial(get_learning_rate,
                       step_size=STEP_SIZE,
                       initial=INITIAL_LEARNING_RATE)
    callback = tf.keras.callbacks.LearningRateScheduler(schedule)

    # Compute the number of epochs, i.e. the number of iterations it takes for
    # the learning rate to go from its initial value to the maximum specified
    # value
    epochs = int(
        math.log(MAX_LEARNING_RATE / INITIAL_LEARNING_RATE, STEP_SIZE))

    # Start training, saving the loss after each "epoch".
    history = model.fit_generator(generator(),
                                  steps_per_epoch=1,
                                  epochs=epochs,
                                  callbacks=[callback])

    plot_path = f'scratch/learning_rate_test-{DATASET}.jpg'
    plt.plot(list(map(schedule, range(epochs))), history.history['loss'])
    plt.xscale('log')
    plt.savefig(plot_path)
    print(f'Saved plot to {plot_path}')
    plt.show()


def main():
    triplet_embedding_model = BASE_MODEL.get_triplet_embedding_model()
    triplets = make_triplets(DATASET)
    lr_test(triplet_embedding_model, triplets)


if __name__ == '__main__':
    main()

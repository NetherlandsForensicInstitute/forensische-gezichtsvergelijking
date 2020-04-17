import argparse
import os
import time

import numpy as np
from tensorflow.keras.optimizers import Adam

from lr_face.data import EnfsiDataset, to_array
from lr_face.losses import TripletLoss
from lr_face.models import TripletEmbeddingModel, Architecture
from lr_face.utils import fix_tensorflow_rtx

# Needed to make TensorFlow 2.x work with RTX Nvidia cards.
fix_tensorflow_rtx()


def finetune(model: TripletEmbeddingModel,
             anchors: np.ndarray,
             positives: np.ndarray,
             negatives: np.ndarray):
    """
    Fine-tunes a Tensorflow model.

    Arguments:
        model: A `TripletEmbeddingModel` instance.
        anchors: A 4D array containing a batch of anchor images with shape
            `(batch_size, height, width, num_channels)`.
        positives: A 4D array containing a batch of images of the same identity
            as the anchor image with shape `(batch_size, height, width,
            num_channels)`.
        negatives: A 4D array containing a batch of images of a different
            identity than the anchor image with shape `(batch_size, height,
            width, num_channels)`.
    """

    model.compile(
        optimizer=Adam(learning_rate=3e-5),
        loss=TripletLoss(alpha=.2),  # TODO: better value for alpha?
    )

    # The triplet loss that is used to train the model actually does not need
    # any ground truth labels, since it simply aims to maximize the difference
    # in distances to the anchor embedding between positive and negative query
    # images. However, Keras' Loss interface still needs a `y_true` that has
    # the same first dimension as the `y_pred` output by the model. That's why
    # we create a dummy "ground truth" of the same length.
    inputs = [anchors, positives, negatives]
    y = np.zeros(shape=(anchors.shape[0], 1))
    model.fit(
        x=inputs,
        y=y,
        batch_size=2,  # TODO: make dynamic
        epochs=100  # TODO: make dynamic
    )


def main(model_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    architecture = Architecture[model_name.upper()]
    triplet_embedding_model = architecture.get_triplet_embedding_model()
    dataset = EnfsiDataset(years=[2011, 2012, 2013, 2017])

    anchors, positives, negatives = to_array(
        dataset.triplets, resolution=architecture.resolution, normalize=True)
    try:
        finetune(triplet_embedding_model, anchors, positives, negatives)
    except KeyboardInterrupt:
        # Allow user to manually interrupt training and still save checkpoint.
        pass

    weights_name = f"{dataset}-{time.strftime('%Y_%m_%d-%H_%M_%S')}"
    weights_path = os.path.join(output_dir, f'{weights_name}.h5')
    triplet_embedding_model.save_weights(weights_path, overwrite=True)


if __name__ == '__main__':
    """
    Example usage: 
    
    ```
    python finetuning.py -m vggface -o scratch
    ``` 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',
        '-m',
        required=True,
        type=str,
        help='Should match one of the constants in the `Architecture` Enum'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        required=True,
        type=str,
        help='Path to the directory in which the weights should be stored'
    )
    args = parser.parse_args()
    main(**vars(args))

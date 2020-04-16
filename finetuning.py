import argparse
import os

import numpy as np
from tensorflow.keras.optimizers import Adam

from lr_face.data import make_triplets, EnfsiDataset, to_array
from lr_face.losses import TripletLoss
from lr_face.models import TripletEmbedder, BaseModel
from lr_face.utils import fix_tensorflow_rtx

# Needed to make TensorFlow 2.x work with RTX Nvidia cards.
fix_tensorflow_rtx()


def finetune(model: TripletEmbedder,
             anchors: np.ndarray,
             positives: np.ndarray,
             negatives: np.ndarray):
    """
    Fine-tunes a Tensorflow model.

    Arguments:
        model: A BaseModel instance that is suitable for training, i.e. whose
            output is compatible with the triplet loss function. See
            `BaseModel.load_training_model()` for more information.
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
        optimizer=Adam(learning_rate=3e-5),  # TODO: default choice
        loss=TripletLoss(alpha=1.),  # TODO: better value for alpha?
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
    base_model: BaseModel = BaseModel[model_name.upper()]
    triplet_embedder = base_model.load_triplet_embedder()
    dataset = EnfsiDataset(years=[2011, 2012, 2013, 2017])
    triplets = make_triplets(dataset)
    x = to_array(triplets, (224, 224))  # TODO: Make resolution dynamic
    anchors, positives, negatives = np.split(x, 3, axis=1)
    try:
        finetune(triplet_embedder,
                 np.squeeze(anchors, axis=1),
                 np.squeeze(positives, axis=1),
                 np.squeeze(negatives, axis=1))
    except KeyboardInterrupt:
        # Allow user to manually interrupt training and still save checkpoint.
        pass

    weights_path = os.path.join(output_dir, 'weights.h5')
    triplet_embedder.save_weights(weights_path, overwrite=True)


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
        help='Should match one of the constants in the `BaseModel` Enum'
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

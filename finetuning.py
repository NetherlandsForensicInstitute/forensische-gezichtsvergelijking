import argparse

import cv2
from tensorflow.keras.optimizers import Adam

from lr_face.data import LfwDevDataset
from lr_face.losses import TripletLoss
from lr_face.models import Architecture
from lr_face.utils import fix_tensorflow_rtx
from lr_face.versioning import Tag

fix_tensorflow_rtx()

AUGMENT_RESOLUTION = (50, 50)
DATASET = LfwDevDataset(training=True)
OPTIMIZER = Adam(learning_rate=3e-5)
LOSS = TripletLoss(alpha=.2)  # TODO: better value for alpha?
BATCH_SIZE = 4  # TODO: make dynamic
NUM_EPOCHS = 100  # TODO: make dynamic


def augmenter(image):
    return cv2.resize(image, AUGMENT_RESOLUTION)


def main(architecture: str, tag: str):
    architecture: Architecture = Architecture[architecture.upper()]
    triplet_embedding_model = architecture.get_triplet_embedding_model()
    tag = Tag(tag)

    # Determine under which tag to save the fine-tuned weights if none was
    # explicitly specified.
    if not tag.version:
        try:
            tag.version = architecture.get_latest_version(tag) + 1
        except ValueError:
            tag.version = 1

    try:
        triplet_embedding_model.train(DATASET.triplets,
                                      BATCH_SIZE,
                                      NUM_EPOCHS,
                                      OPTIMIZER,
                                      LOSS,
                                      # Apply augmenter to anchors only.
                                      augmenter=(augmenter, None, None))
    # Allow user to manually interrupt training while still saving weights.
    except KeyboardInterrupt:
        pass
    triplet_embedding_model.save_weights(tag)


if __name__ == '__main__':
    """
    Example usage: 
    
    ```
    python finetuning.py -a vggface -t no_augmentation
    ``` 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--architecture',
        '-a',
        required=True,
        type=str,
        help='Should match one of the constants in the `Architecture` Enum'
    )
    parser.add_argument(
        '--tag',
        '-t',
        required=True,
        type=str,
        help='The name used for saving the finetuned weights'
    )
    args = parser.parse_args()
    main(**vars(args))

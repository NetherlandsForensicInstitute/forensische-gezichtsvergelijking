import argparse

from tensorflow.keras.optimizers import Adam

from lr_face.data import EnfsiDataset
from lr_face.losses import TripletLoss
from lr_face.models import Architecture
from lr_face.utils import fix_tensorflow_rtx
from lr_face.versioning import Tag

fix_tensorflow_rtx()


def main(architecture: str, tag: str):
    architecture: Architecture = Architecture[architecture.upper()]
    try:
        version = architecture.get_latest_version(tag) + 1
    except ValueError:
        version = 1
    tag = Tag(tag, version)
    triplet_embedding_model = architecture.get_triplet_embedding_model()
    dataset = EnfsiDataset(years=[2011, 2012, 2013, 2017])
    optimizer = Adam(learning_rate=3e-5)
    loss = TripletLoss(alpha=.2)  # TODO: better value for alpha?
    batch_size = 2  # TODO: make dynamic
    num_epochs = 100  # TODO: make dynamic

    try:
        triplet_embedding_model.train(dataset.triplets,
                                      batch_size,
                                      num_epochs,
                                      optimizer,
                                      loss)
    except KeyboardInterrupt:
        # Allow user to manually interrupt training while still saving weights.
        pass
    triplet_embedding_model.save_weights(tag)


if __name__ == '__main__':
    """
    Example usage: 
    
    ```
<<<<<<< HEAD
    python finetuning.py -m vggface -v 0.0.1
=======
    python finetuning.py -a vggface -t no_augmentation
>>>>>>> origin/feature/versioning
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

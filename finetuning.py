import argparse
from typing import Optional

from tensorflow.keras.optimizers import Adam

from lr_face.data import EnfsiDataset
from lr_face.losses import TripletLoss
from lr_face.models import Architecture
from lr_face.utils import fix_tensorflow_rtx
from lr_face.versioning import Version

fix_tensorflow_rtx()


def determine_version(version: Optional[str],
                      architecture: Architecture) -> Version:
    if version:
        return Version.from_string(version)
    else:
        try:
            return architecture.get_latest_version().increment()
        except ValueError:
            return Version(0, 0, 1)


def main(model_name: str, version: Optional[str] = None):
    architecture: Architecture = Architecture[model_name.upper()]
    version = determine_version(version, architecture)
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
    triplet_embedding_model.save_weights(version)


if __name__ == '__main__':
    """
    Example usage: 
    
    ```
    python finetuning.py -m vggface -v 0.0.1
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
        '--version',
        '-v',
        required=False,
        default=None,
        type=str,
        help='A new version number. Defaults to an increment of latest version'
    )
    args = parser.parse_args()
    main(**vars(args))

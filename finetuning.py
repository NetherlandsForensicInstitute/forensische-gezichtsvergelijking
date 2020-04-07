import argparse
import importlib
import os
from enum import Enum, auto
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from lr_face.data_providers import Triplet, test_data, make_triplets
from lr_face.losses import TripletLoss
from lr_face.models import FinetuneModel
from lr_face.utils import fix_tensorflow_rtx

# Needed to make TensorFlow 2.x work with RTX Nvidia cards.
fix_tensorflow_rtx()


class BaseModel(Enum):
    """
    This Enum can be used to define all base model architectures that we
    currently support, and to retrieve embedding and finetune models. This
    abstracts away the individual implementations of various models so that
    there is one unified way of loading models.

    TODO: currently only supports Tensorflow models.

    To load the embedding model for VGGFace for example, you would use:

        `BaseModel.VGGFace.load_embedding_model()`

    Similarly, to load a finetune model, you would use:

        `BaseModel.VGGFace.load_finetune_model()`
    """
    VGGFace = auto()
    Facenet = auto()
    FbDeepFace = auto()
    OpenFace = auto()

    def load_embedding_model(self) -> tf.keras.Model:
        if self.source == 'deepface':
            # TODO: could be nicer, but works with current deepface API.
            module_name = f'deepface.basemodels.{self.name}'
            module = importlib.import_module(module_name)
            return module.loadModel()
        raise ValueError("Unable to load embedding model.")

    def load_finetune_model(self) -> FinetuneModel:
        return FinetuneModel(self.load_embedding_model())

    @property
    def source(self) -> str:
        """
        Returns a textual description of where the model comes from.

        :return: str
        """
        deepface_models = [self.VGGFace,
                           self.Facenet,
                           self.FbDeepFace,
                           self.OpenFace]
        if self in deepface_models:
            return 'deepface'
        raise ValueError("Unknown model source.")


def finetune(model: FinetuneModel, triplets: List[Triplet]):
    """
    Fine-tunes a model.

    TODO: currently only supports Tensorflow models.

    Arguments:
        model: A BaseModel instance that is suitable for training, i.e. whose
            output is compatible with the triplet loss function. See
            `BaseModel.load_training_model()` for more information.
        triplets: A list of `Triplet` instances that will be used for training.
    """

    anchors, positives, negatives = zip(*[(
        triplet.anchor,
        triplet.positive,
        triplet.negative
    ) for triplet in triplets])

    model.compile(
        optimizer=Adam(learning_rate=3e-4),  # TODO: default choice
        loss=TripletLoss(alpha=0.5),  # TODO: better value for alpha?
    )

    x = [np.stack(anchors), np.stack(positives), np.stack(negatives)]

    # The triplet loss that is used to train the model actually does not need
    # any ground truth labels, since it simply aims to maximize the difference
    # in distances to the anchor embedding between positive and negative query
    # images. However, Keras' Loss interface still needs a `y_true` that has
    # the same first dimension as the `y_pred` output by the model. That's why
    # we create a dummy "ground truth" of the same length.
    y = np.zeros(shape=(len(triplets), 1))
    model.fit(
        x=x,
        y=y,
        batch_size=2,  # TODO: make dynamic
        epochs=1  # TODO: make dynamic
    )


def main(model_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    base_model: BaseModel = BaseModel[model_name]
    finetune_model = base_model.load_finetune_model()
    data = test_data(resolution=(224, 224))  # TODO: make dynamic
    triplets = make_triplets(data)
    finetune(finetune_model, triplets)
    weights_path = os.path.join(output_dir, 'weights.h5')
    finetune_model.save_weights(weights_path, overwrite=True)


if __name__ == '__main__':
    """
    Example usage: 
    
    ```
    python finetuning.py -m VGGFace -o scratch
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

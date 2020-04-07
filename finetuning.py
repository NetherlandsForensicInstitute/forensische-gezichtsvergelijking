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
    VGGFace = auto()
    Facenet = auto()
    FbDeepFace = auto()
    OpenFace = auto()

    def load_inference_model(self) -> tf.keras.Model:
        if self.source == 'deepface':
            module_name = f'deepface.basemodels.{self.name}'
            module = importlib.import_module(module_name)
            return module.loadModel()
        raise ValueError("Unable to load inference model.")

    def load_training_model(self) -> tf.keras.Model:
        return FinetuneModel(self.load_inference_model())

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


def finetune_model(model: tf.keras.Model, triplets: List[Triplet]):
    """
    Fine-tunes a model.

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
        loss=TripletLoss(alpha=0.5),  # TODO: optimize alpha
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
    training_model = base_model.load_training_model()
    data = test_data(resolution=(224, 224))  # Load data based on
    triplets = make_triplets(data)
    finetune_model(training_model, triplets)
    weights_path = os.path.join(output_dir, 'weights.h5')
    training_model.save_weights(weights_path)


if __name__ == '__main__':
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

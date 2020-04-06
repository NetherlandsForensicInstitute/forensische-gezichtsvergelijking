import argparse
import importlib
from enum import Enum, auto
from typing import List

import numpy as np
from tensorflow.keras.optimizers import Adam

import deepface.basemodels
from lr_face.data_providers import Triplet, test_data, make_triplets
from lr_face.losses import TripletLoss
from lr_face.utils import fix_tensorflow_rtx

fix_tensorflow_rtx()


class BaseModel(Enum):
    VGGFace = auto()
    Facenet = auto()
    FbDeepFace = auto()
    OpenFace = auto()

    def load_inference_model(self):
        if self.source == 'deepface':
            module_name = f'deepface.basemodels.{self.name}'
            module = importlib.import_module(module_name)
            return module.loadModel()
        raise ValueError("Unable to load inference model.")

    def load_training_model(self):
        if self.source == 'deepface':
            module_name = f'deepface.basemodels.{self.name}'
            module = importlib.import_module(module_name)
            return module.load_training_model()
        raise ValueError("Unable to load training model.")

    @property
    def source(self):
        deepface_models = [self.VGGFace,
                           self.Facenet,
                           self.FbDeepFace,
                           self.OpenFace]
        if self in deepface_models:
            return 'deepface'
        raise ValueError("Unknown model source.")


def finetune_model(model: BaseModel, triplets: List[Triplet]):
    """
    Fine-tunes a model.

    Arguments:
        model: A BaseModel instance that is suitable for training,
            i.e. whose output is compatible with the triplet loss
            function.
        triplets: A list of `Triplet` instances that will be used
            for training.
    """

    # TODO: ensure that all embeddings in triplet are normalized.
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
    print(x[0].shape)
    model.fit(
        x=x,
        y=np.zeros(shape=(len(triplets), 1)),  # Unused, but has to match `x`
        batch_size=2,  # TODO: make dynamic
        epochs=1  # TODO: make dynamic
    )


def main(model_name: str):
    base_model = getattr(BaseModel, model_name)
    model = base_model.load_training_model()
    data = test_data(resolution=(224, 224))
    triplets = make_triplets(data)
    finetune_model(model, triplets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', '-m', required=True, type=str)
    args = parser.parse_args()
    main(**vars(args))

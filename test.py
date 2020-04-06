import numpy as np
from tensorflow.keras.optimizers import Adam

from deepface.basemodels import VGGFace
from lr_face.data_providers import make_triplets, ImageWithIds
from lr_face.losses import TripletLoss
from lr_face.utils import fix_tensorflow_rtx

fix_tensorflow_rtx()


def dummy():
    n = 40
    data = ImageWithIds(
        images=[np.random.normal(size=(224, 224, 3)) for _ in range(n)],
        person_ids=[i // 4 for i in range(n)],
        image_ids=list(map(str, range(n))))

    return make_triplets(data)


def main():
    triplets = dummy()
    anchors, positives, negatives = zip(*[(
        triplet.anchor,
        triplet.positive, triplet.negative
    ) for triplet in triplets])

    model = VGGFace.load_training_model()
    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=TripletLoss(alpha=0.5),
    )

    model.fit(
        x=[np.stack(anchors), np.stack(positives), np.stack(negatives)],
        y=np.zeros(shape=(len(triplets), 1)),
        batch_size=2,
    )


if __name__ == '__main__':
    main()

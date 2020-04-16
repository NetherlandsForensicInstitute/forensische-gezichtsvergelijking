from typing import List

import numpy as np

from lr_face.data_providers import Triplet


def evaluate_accuracy(model, triplets: List[Triplet]):
    def classify(emb1, emb2):
        return np.linalg.norm(emb1 - emb2) < 1.2

    num_correct = 0
    num_incorrect = 0
    for triplet in triplets:
        anchor = model.predict(triplet.anchor[None, :, :, :])
        positive = model.predict(triplet.positive[None, :, :, :])
        negative = model.predict(triplet.negative[None, :, :, :])

        if classify(anchor, positive):
            num_correct += 1
        else:
            num_incorrect += 1

        if classify(anchor, negative):
            num_incorrect += 1
        else:
            num_correct += 1

    return num_correct / (num_correct + num_incorrect)


if __name__ == '__main__':
    dataset = EnfsiDataset([2011])
    print('\n'.join(map(str, [(x.path, x.identity) for x in dataset.data])))

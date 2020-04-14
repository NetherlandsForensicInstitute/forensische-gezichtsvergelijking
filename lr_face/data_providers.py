import os
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from typing import Tuple, List, Optional, Dict

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from typing_extensions import Protocol


@dataclass
class ImagePairs:
    y_calibrate: List[int]
    X_calibrate: List[List[np.ndarray]]
    ids_calibrate: List[str]
    y_test: List[int]
    X_test: List[List[np.ndarray]]
    ids_test: List[str]


@dataclass
class PairsWithIds:
    pairs: List[Tuple]
    is_same_source: List[int]
    pair_ids: List


class PairProvider(Protocol):
    def __call__(self, *args, **kwargs) -> PairsWithIds:
        pass


@dataclass
class ImageWithIds:
    images: List
    person_ids: List
    image_ids: List[str]


class ImageProvider(Protocol):
    def __call__(self, resolution: Tuple[int, int], *args,
                 **kwargs) -> ImageWithIds:
        pass


@dataclass
class Triplet:
    # Anchor image, shape `(height, width, num_channels)`.
    anchor: np.ndarray

    # Image of a face with the same identity as the face depicted on the anchor
    # image. Shape `(height, width, num_channels)`.
    positive: np.ndarray

    # Image of a face with a different identity from the face depicted on the
    # anchor image. Shape `(height, width, num_channels)`.
    negative: np.ndarray


@dataclass
class DataFunctions:
    pair_provider: Optional[PairProvider]
    image_provider: Optional[ImageProvider]

    def __str__(self):
        desc = ''
        if self.pair_provider:
            desc+=str(self.pair_provider)
        if self.image_provider:
            desc+=str(self.image_provider)
        return desc


class TestData:
    def __call__(self, resolution=(100,100)):
        """
        Return some random numbers in the right structure to test the pipeline with.
        """
        n = 11
        return ImageWithIds(
            images=list(np.random.random([n, resolution[0], resolution[1], 3])),
            person_ids=[1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5],
            image_ids=list(range(n)))

    def __str__(self):
        return 'test data'


class EnfsiData:
    def __init__(self, years=(2011, 2012, 2013, 2017)):
        self.years = years

    def __call__(self, resolution) -> PairsWithIds:
        sets=[]
        for year in self.years:
            folder = os.path.join('resources', 'enfsi', str(year))
            files = os.listdir(folder)
            n_pairs = (len(files) - 1) // 2
            X = [[-1, -1] for _ in range(n_pairs)]
            y = [-1] * n_pairs
            ids = [-1] * n_pairs
            df = pd.read_csv(os.path.join(folder, 'truth.csv')).set_index('id')
            for file in files:
                if not file.endswith('csv'):
                    # TODO check RGB/GRB ordering/colous usage on models.
                    img = cv2.imread(os.path.join(folder, file), cv2.COLOR_BGR2RGB)
                    questioned_ref = None
                    if year == 2011:
                        cls = int(file[:3])
                        questioned_ref = file[3]
                    elif year == 2012:
                        cls = int(file[:2])
                        questioned_ref = file[2]
                    elif year == 2013:
                        cls = int(file[1:3])
                        questioned_ref = file[0]
                    elif year == 2017:
                        cls = int(file[1:3])
                        questioned_ref = file[0]
                    else:
                        raise ValueError(f'Unknown ENFSI year {year}')
                    y[cls - 1] = int(df.loc[cls]['same'] == 1)
                    ids[
                        cls - 1] = f"enfsi_{year}_{cls}_{int(df.loc[cls]['same'] == 1)}"
                    if questioned_ref == 'q':
                        X[cls - 1][0] = img
                    elif questioned_ref == 'r':
                        X[cls - 1][1] = img
                    else:
                        raise ValueError(f'unknown questioned/ref: {questioned_ref}')
        sets.append(PairsWithIds(pairs=X, is_same_source=y, pair_ids=ids))
        return combine_paired_data(sets)

    def __str__(self):
        if len(self.years) == 4:
            return 'ENFSI'
        return 'ENFSI '+', '.join(map(lambda x: str(x-2000), self.years))


def combine_unpaired_data(image_providers: List[ImageProvider],
                          resolution) -> ImageWithIds:
    """
    Gets the X and y for all data in the callables, and returns the total set.
    """
    X = []
    y = np.array([])
    ids = []
    max_class = 0
    for dataset_callable in image_providers:
        this_X, this_y, this_ids = dataset_callable(resolution)
        y = np.append(y, this_y + max_class)
        max_class = max(y)
        X += this_X
        ids += this_ids
    return ImageWithIds(images=X, person_ids=list(y.astype(int)),
                        image_ids=ids)


def combine_paired_data(pair_lists: List[PairsWithIds]) -> PairsWithIds:
    """
    Appends the datasets
    """
    X = []
    y = []
    ids = []
    for pairs in pair_lists:
        assert type(pairs) == PairsWithIds
        y += pairs.is_same_source
        X += pairs.pairs
        ids = ids + pairs.pair_ids
    # we are assuming the ids are already unique - let's check
    assert len(ids) == len(set(ids))
    if X:
        assert len(X[0]) == 2
    return PairsWithIds(pairs=X, is_same_source=y, pair_ids=ids)


def get_data(datasets: DataFunctions, resolution=(100, 100), fraction_test=0.2,
             **kwargs) -> ImagePairs:
    """
    Takes a function that can return both pairs of images or unpaired images (with person identities).
    Returns a dataset with all data combined into pairs and split into the right datasets.
    """
    X_calibrate = []
    y_calibrate = []
    ids_calibrate = []

    X_test = []
    y_test = []
    ids_test = []

    if datasets.image_provider:
        images = datasets.image_provider(resolution)
        X, y, ids = images.images, images.person_ids, images.image_ids
        # TODO for now we will let the model resize, in future we should enforce the right resolution to come from preprocessing
        # assert this_X.shape[1:3] == resolution, f'resolution should be {resolution}, not {this_X.shape[:2]}'
        assert len(X) == len(y), f'y and X should have same length'

        # split on identities, not on samples (so same person does not appear in both test and train
        images_calibrate, images_test = split_data_on_groups(X, fraction_test,
                                                             y, ids)
        assert len(images_calibrate.image_ids + images_test.image_ids) == \
               len(set(images_calibrate.image_ids + images_test.image_ids))

        # make pairs per set
        pairs_test = make_pairs(images_test)
        pairs_calibrate = make_pairs(images_calibrate)

        X_calibrate += pairs_calibrate.pairs
        y_calibrate += pairs_calibrate.is_same_source
        ids_calibrate += pairs_calibrate.pair_ids

        X_test += pairs_test.pairs
        y_test += pairs_test.is_same_source
        ids_test += pairs_test.pair_ids

    if datasets.pair_provider:
        pairs = datasets.pair_provider(resolution=resolution)
        res = train_test_split(pairs.pairs, pairs.is_same_source,
                               pairs.pair_ids,
                               test_size=fraction_test,
                               stratify=pairs.is_same_source)

        X_calibrate += res[0]
        y_calibrate += res[2]
        ids_calibrate += res[4]

        X_test += res[1]
        y_test += res[3]
        ids_test += res[5]

    return ImagePairs(y_test=y_test, X_test=X_test, ids_test=ids_test,
                      y_calibrate=y_calibrate,
                      X_calibrate=X_calibrate, ids_calibrate=ids_calibrate)


def split_data_on_groups(X, fraction2, y, ids) -> Tuple[ImageWithIds]:
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction2, random_state=42)
    for train_idx, test_idx in gss.split(X, y, y):
        return (ImageWithIds(*zip(*[(X[t], y[t], ids[t]) for t in train_idx])),
                ImageWithIds(*zip(*[(X[t], y[t], ids[t]) for t in test_idx])))


def make_pairs(data: ImageWithIds) -> PairsWithIds:
    """
    Takes images and returns pairs.

    Example
    [x1, .., x9], [1,2,3,4,5,6,7,8,9]
    ->
    [[x1,x2], [x1,x3], [x1,x2], [x1, x4], [x1, x5], ...], [1, 1, 1, 0, 0, ...]

    Currently makes different sources only by pairing class n with n+1 rather than taking all ~N^2 possible pairs,
    to keep data sets limited.
    """
    person_ids = np.unique(data.person_ids)
    pairs = []
    ids = []
    same_different_source = []
    for i_person_id, person_id in enumerate(person_ids):
        idx = data.person_ids == person_id
        nidx = sum(idx)
        # all images of this person
        imgs = np.array(data.images)[idx]
        img_ids = np.array(data.image_ids)[idx]
        for i in range(nidx):
            for j in range(i + 1, nidx):
                # make same-person pairs by pairing all images of the person
                pairs.append((imgs[i], imgs[j]))
                same_different_source.append(1)
                ids.append(tuple(sorted([img_ids[i], img_ids[j]])))
            if i_person_id > 0:
                # make different-person pairs by pairing person i with person i-1
                for j in range(len(imgs_prev)):
                    pairs.append((imgs[i], imgs_prev[j]))
                    same_different_source.append(0)
                    ids.append(tuple(sorted([img_ids[i], img_ids_prev[j]])))

        imgs_prev = imgs
        img_ids_prev = img_ids
    return PairsWithIds(pairs=pairs, is_same_source=same_different_source,
                        pair_ids=ids)


def make_triplets(data: ImageWithIds) -> List[Triplet]:
    unique_identities = set(data.person_ids)
    if len(unique_identities) < 2:
        raise ValueError(
            "Can't make triplets if there are fewer than 2 unique identities.")

    # Pair the images with their identity.
    images_with_identity: List[Tuple[np.ndarray, int]] = \
        list(zip(data.images, data.person_ids))

    # Convert data to a more pleasant format: a mapping from identities to a
    # list of all images with that identity.
    images_grouped_by_identity = defaultdict(list)
    for image, identity in images_with_identity:
        images_grouped_by_identity[identity].append(image)

    triplets = []
    for identity, images in images_grouped_by_identity.items():
        for i, anchor in enumerate(images):
            for positive in images[i + 1:]:
                # TODO: better negative sampling, currently random.
                negative_identity = random.choice(
                    tuple(unique_identities - {identity}))
                negative = random.choice(
                    images_grouped_by_identity[negative_identity])
                triplets.append(Triplet(anchor, positive, negative))
    return triplets
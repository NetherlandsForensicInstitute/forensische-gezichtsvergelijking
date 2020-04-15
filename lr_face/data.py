import csv
import os
import random
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional, Union, Iterator

import cv2
import numpy as np


@dataclass
class FaceImage:
    """
    A simple data structure that can be used throughout the application to
    handle annotated images in a unified way. All datasets should preferably
    be a wrapper around handling lists of `FaceImage` instances.

    TODO: we could cache `read()` internally: decentralized and hidden away.
    """

    # The path to the image file.
    path: str

    # A globally unique identifier for the person depicted on the image, only
    # shared with other images that depict the same person.
    identity: str

    # An optional miscellaneous dictionary where any potentially relevant
    # metadata about the image can be stored.
    meta: Dict[str, Any] = None

    def read(self,
             resolution: Optional[Tuple[int, int]] = None,
             normalize: bool = False) -> np.ndarray:
        """
        Returns a 3D array of shape `(height, width, num_channels)`. Optionally
        a `resolution` may be specified as a `(width, height)` tuple to resize
        the image to those dimensions. If `normalize` is True, the returned
        array will contain values scaled between [0, 1] to be compatible with
        the input format expected by models.

        :param resolution: Optional[Tuple[int, int]]
        :param normalize: bool
        :return: np.ndarray
        """
        res = cv2.imread(self.path)
        if res is None:
            raise ValueError(f'Reading {self.path} resulted in None')
        if res.shape[-1] != 3:
            raise ValueError(f'Expected 3 channels, got {res.shape[-1]}')
        if resolution:
            res = cv2.resize(res, resolution)
        if normalize:
            res = res / 255.  # Normalize input between [0, 1]
        return res

    def __post_init__(self):
        if not self.meta:
            self.meta = dict()


@dataclass
class FacePair:
    first: FaceImage
    second: FaceImage

    @property
    def same(self) -> bool:
        """
        Returns whether or not the two images in this pair share the same
        identity or not.

        :return: bool
        """
        return self.first.identity == self.second.identity

    def __iter__(self):
        """
        A simple iterator implementation that allows unpacking a `FacePair`
        instance, i.e.

        ```python
        pairs: List[FacePair] = ...
        for first, second in pairs:
            ...
        ```
        """
        return iter([self.first, self.second])


@dataclass
class FaceTriplet:
    anchor: FaceImage
    positive: FaceImage
    negative: FaceImage

    def __iter__(self):
        """
        A simple iterator implementation that allows unpacking a `FaceTriplet`
        instance, i.e.

        ```python
        triplets: List[FaceTriplet] = ...
        for anchor, positive, negative in triplets:
            ...
        ```
        """
        return iter([self.anchor, self.positive, self.negative])

    def __post_init__(self):
        """
        Validates whether or not the triplet is valid, i.e. makes sure that
        `anchor` and `positive` share the same identity, and `anchor` and
        `negative` do not.
        """
        if self.anchor.identity != self.positive.identity:
            raise ValueError(
                'Anchor and positive image have different identities')

        if self.anchor.identity == self.negative.identity:
            raise ValueError(
                'Anchor and negative image have the same identity')


class DummyFaceImage(FaceImage):
    """
    A dummy class that can be used in place of a real `FaceImage` for testing.
    """

    def read(self,
             resolution: Optional[Tuple[int, int]] = None,
             normalize: bool = False) -> np.ndarray:
        """
        Since dummy instances don't have a real path, we override the `read()`
        method to just return random pixel data.
        """
        return np.random.random(size=(resolution[1], resolution[0], 3))


class Dataset:
    def __init__(self):
        self._data = None

    @property
    def images(self) -> List[FaceImage]:
        """
        Returns a list of all `FaceImage` instances that make up this dataset.

        :return: List[FaceImage]
        """
        if not self._data:
            self._data = self.load_data()
        return self._data

    @property
    def pairs(self) -> List[FacePair]:
        """
        Returns a list of `FacePair` instances from the images stored in this
        dataset. Subclasses can override this method if the dataset has a
        specific set of pairs associated with it (where not all images are used
        for example).

        :return: List[FacePair]
        """
        return make_pairs(self.images)

    @property
    def triplets(self) -> List[FaceTriplet]:
        """
        Returns a list of `FaceTriplet` instances from the images stored in
        this dataset. Subclasses can override this method if the dataset has a
        specific set of triplets associated with it (where not all images are
        used for example).

        :return: List[FaceTriplet]
        """
        return make_triplets(self.images)

    @property
    def num_identities(self) -> int:
        """
        Returns the unique number of identities in the dataset.

        :return: int
        """
        return len(set(x.identity for x in self.images))

    @abstractmethod
    def load_data(self) -> List[FaceImage]:
        """
        This abstract method is meant to be implemented by each subclass. It is
        used to lazily load the data only when it is necessary. To access the
        images inside a `Dataset` instance, always use the `images` property.

        :return: List[FaceImage]
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[FaceImage]:
        """
        Makes the dataset iterable by returning an iterator over all images.

        :return: Iterator[FaceImage]
        """
        return iter(self.images)

    def __len__(self) -> int:
        """
        Returns the number of images in this dataset.

        :return: int
        """
        return len(self.images)

    def __str__(self) -> str:
        return self.__class__.__name__


class TestDataset(Dataset):
    def load_data(self) -> List[FaceImage]:
        ids = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
        return [DummyFaceImage(path='', identity=f'TEST-{idx}') for idx in ids]


class ForenFaceDataset(Dataset):
    ROOT = os.path.join('resources', 'forenface')

    def __init__(self, max_num_images: int):
        self.max_num_images = max_num_images
        super().__init__()

    def load_data(self) -> List[FaceImage]:
        files = os.listdir(self.ROOT)
        if self.max_num_images > len(files):
            files = random.sample(files, self.max_num_images)

        data = []
        for file in files:
            path = os.path.join(self.ROOT, file)
            identity = f'FORENFACE-{file[:3]}'
            data.append(FaceImage(path, identity, {
                'source': str(self)
            }))
        return data


class LfwDataset(Dataset):
    ROOT = os.path.join('resources', 'lfw')

    def __init__(self):
        super().__init__()
        self._pairs = None

    @property
    def pairs(self) -> List[FacePair]:
        if not self._pairs:
            pairs = []
            with open(os.path.join(self.ROOT, 'pairs.txt'), 'r') as f:
                # Skip the first line.
                _, *lines = [line.strip() for line in f.readlines()]

                # The first half of the remaining lines consists of positive
                # pairs (images with the same identity).
                positive_lines = lines[:len(lines) // 2]
                negative_lines = lines[len(lines) // 2:]
                for line in positive_lines:
                    person, idx1, idx2 = line.split('\t')
                    pairs.append(FacePair(
                        self._create_face_image(person, int(idx1)),
                        self._create_face_image(person, int(idx2))
                    ))

                # The second half consists of negative pairs (images with
                # different identities).
                for line in negative_lines:
                    person1, idx1, person2, idx2 = line.split('\t')
                    pairs.append(FacePair(
                        self._create_face_image(person1, int(idx1)),
                        self._create_face_image(person2, int(idx2))
                    ))
            self._pairs = pairs
        return self._pairs

    def _create_face_image(self, person: str, idx: int) -> FaceImage:
        return FaceImage(
            path=self._get_path(person, idx),
            identity=self._create_identity(person),
            meta={
                'source': str(self)
            }
        )

    def load_data(self) -> List[FaceImage]:
        data = []
        for person in os.listdir(self.ROOT):
            identity = self._create_identity(person)
            person_dir = os.path.join(self.ROOT, person)
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                data.append(FaceImage(image_path, identity, {
                    'source': str(self)
                }))
        return data

    @staticmethod
    def _create_identity(person: str) -> str:
        return f'LFW-{person}'

    def _get_path(self, person: str, idx: int) -> str:
        """
        Get the full path to the image file corresponding to the given `person`
        and `idx`.

        :param person: str
        :param idx: int
        :return: str
        """
        return os.path.join(self.ROOT, f'{person}_{idx:04}.jpg')


class EnfsiDataset(Dataset):
    ROOT = os.path.join('resources', 'enfsi_cropped')  # TODO: change back

    def __init__(self, years: List[int]):
        self.years = years
        super().__init__()

    def load_data(self) -> List[FaceImage]:
        data = []
        for year in self.years:
            folder = os.path.join(self.ROOT, str(year))
            with open(os.path.join(folder, 'truth.csv')) as f:
                reader = csv.DictReader(f)
                for line in reader:
                    idx = int(line['id'])
                    same = line['same'] == '1'
                    query, reference = self._get_query_and_reference(year, idx)
                    reference_id = self._create_reference_id(year, idx)
                    query_id = self._create_query_id(year, idx, same)

                    # Create a record for the reference image.
                    path = os.path.join(folder, reference)
                    data.append(FaceImage(path, reference_id, {
                        'source': str(self),
                        'year': year
                    }))

                    # Create a record for the query image.
                    path = os.path.join(folder, query)
                    data.append(FaceImage(path, query_id, {
                        'source': str(self),
                        'year': year
                    }))
        return data

    @staticmethod
    def _create_reference_id(year: int, idx: int) -> str:
        """
        Creates a unique ID for the person in the reference image defined by
        the given `idx` and `year`

        :param year: int, the year the relevant dataset was published
        :param idx: int, the index of the image in the given `year`
        :return: str
        """
        return f'ENFSI-{year}-{idx}'

    @classmethod
    def _create_query_id(cls, year: int, idx: int, same: bool) -> str:
        """
        Creates a unique ID for the person in the query image. If the person
        depicted on the query image is the same as the person depicted on the
        corresponding reference image, their IDs will also be the same.
        Otherwise, they are sure to be different.

        :param year: int, the year the relevant dataset was published
        :param idx: int, the index of the image in the given `year`
        :param same: bool, whether the query and reference id are the same
        :return: str
        """
        reference_id = cls._create_reference_id(year, idx)
        if same:
            return reference_id
        return f'{reference_id}-unknown'

    @staticmethod
    def _get_query_and_reference(year: int, idx: int) -> Tuple[str, str]:
        """
        Returns the file names for the query and reference images corresponding
        to the given `year` and `idx`. We need a separate function for this
        because for some reason the format of the filenames in the ENFSI
        dataset seems to change just about every year.

        :param year: int, the year the relevant dataset was published
        :param idx: int, the index of the image in the given `year`
        :return: Tuple[str, str]
        """
        if year < 2013:
            pad_length = 3 if year == 2011 else 2
            query = f'{str(idx).zfill(pad_length)}questioned.jpg'
            reference = f'{str(idx).zfill(pad_length)}reference.jpg'
        else:
            query = f'q{str(idx).zfill(2)}.jpg'
            reference = f'r{str(idx).zfill(2)}.jpg'
        return query, reference

    def __str__(self):
        """
        Override the `__str__()` method to also include the years.

        :return: str
        """
        return f'{super().__str__()}[{"-".join(map(str, self.years))}]'


def make_pairs(data: Union[Dataset, List[FaceImage]],
               same: Optional[bool] = None,
               n: Optional[int] = None) -> List[FacePair]:
    """
    Takes a `Dataset` or a list of `FaceImage` instances and pairs them up.

    Arguments:
        data: The data to convert into pairs.
        same: An optional boolean flag. When set to True, only pairs of images
            that depict the same person are returned. When set to False, only
            pairs of images that depict different people are returned. When
            omitted, an equal number of positive and negative pairs are
            returned. In this case, the negative pairs may contain duplicates.
        n: The number of pairs to return. When omitted, as many pairs as
            possible are returned. This number then depends on `same`:
                - If `same` is True: all possible positive pairs are returned.
                - If `same` is False: all possible negative pairs are returned.
                    Beware: this number may be huge, so use with caution.
                - If `same` is omitted: all possible positive pairs and an
                    equal number of negative pairs are returned.

    Returns:
        A list of `FacePair` instances.
    """
    images_by_identity = defaultdict(list)
    for x in data:
        images_by_identity[x.identity].append(x)

    res = []
    identities = set(x.identity for x in data)

    # First we handle the case when `same` is False, meaning we don't have to
    # make any positive pairs, just negative pairs.
    if same is False:
        if n is None:
            # Again, omitting `n` when `same` is False can lead to an explosion
            # in the number of negative pairs that have to be made. Just look
            # at how many nested for loops we need below. This combination of
            # arguments should only be used when you know that the number of
            # negative pairs will remain manageable.
            for identity, images in images_by_identity.items():
                for image in images:
                    for negative_id in identities - {identity}:
                        for negative in images_by_identity[negative_id]:
                            res.append(FacePair(image, negative))
        else:
            # If `n` is not omitted we simply create `n` random negative pairs.
            for _ in range(n):
                a, b = random.sample(identities, 2)
                res.append(
                    FacePair(
                        random.choice(images_by_identity[a]),
                        random.choice(images_by_identity[b])
                    )
                )

    else:
        # If `same` is either True or omitted (None) we are going to need the
        # positive pairs.
        for identity, images in images_by_identity.items():
            for i, image in enumerate(images):
                for positive in images[i + 1:]:
                    res.append(FacePair(image, positive))

        # If `same` is omitted (None), it means we need to generate an equal
        # number of negative pairs.
        if same is None:
            # Loop over and unpack all positive pairs, then create matching
            # negative pairs that contain at least one of the images from each
            # positive pair and a randomly chosen other image with a different
            # identity.
            for first, second in res:
                identity = first.identity
                negative_id = random.choice(tuple(identities - {identity}))
                negative = random.choice(images_by_identity[negative_id])
                res.append(FacePair(random.choice([first, second]), negative))

        if n:
            res = random.sample(res, min(len(res), n))
    return res


def make_triplets(data: Union[Dataset, List[FaceImage]]) -> List[FaceTriplet]:
    identities = set(x.identity for x in data)
    if len(identities) < 2:
        raise ValueError(
            "Can't make triplets if there are fewer than 2 unique identities.")

    images_by_identity = defaultdict(list)
    for x in data:
        images_by_identity[x.identity].append(x)

    triplets = []
    for identity, images in images_by_identity.items():
        for i, anchor in enumerate(images):
            for positive in images[i + 1:]:
                # TODO: requires better negative sampling, currently random.
                negative_id = random.choice(tuple(identities - {identity}))
                negative = random.choice(images_by_identity[negative_id])
                triplets.append(FaceTriplet(anchor, positive, negative))
    return triplets


def to_array(data: Union[Dataset,
                         List[FaceImage],
                         List[FacePair],
                         List[FaceTriplet]],
             resolution: Tuple[int, int]) -> np.ndarray:
    """
    Converts the `data` to a numpy array of the appropriate shape. This method
    accepts a variety of data types, which influence the shape of the returned
    array:
        - If `data` is a `Dataset` or a list of `FaceImage` instances: Returns
            a 4D array of shape `(num_images, height, width, num_channels)`.
        - If `data` is a list of `FacePair` instances: Returns a 5D array of
            shape `(num_pairs, 2, height, width, num_channels)`.
        - If `data` is a list of `FaceTriplet` instances: Returns a 5D array of
            shape `(num_pairs, 3, height, width, num_channels)`.
    To ensure all images have the same spatial dimensions, a `resolution`
    should be provided as a `(width, height)` tuple. All images will be resized
    to these dimensions.

    :param data:
    :param resolution: Tuple[int, int]
    :return: np.ndarray
    """

    # When `data` is a `Dataset` or a list of `FaceImage` instances.
    if isinstance(data, Dataset) or all(
            isinstance(x, FaceImage) for x in data):
        return np.array([x.read(resolution) for x in data])

    # When `data` is a list of `FacePair` instances.
    if all(isinstance(x, FacePair) for x in data):
        return np.array([[
            first.read(resolution),
            second.read(resolution)
        ] for first, second in data])

    # When `data` is a list of `FaceTriplet` instances.
    if all(isinstance(x, FaceTriplet) for x in data):
        return np.array([[
            anchor.read(resolution),
            positive.read(resolution),
            negative.read(resolution)
        ] for anchor, positive, negative in data])

    # If we haven't returned something by now it means an invalid data type
    # was passed along, so we let the user know about that.
    raise ValueError(f'Invalid data type: {type(data)}')

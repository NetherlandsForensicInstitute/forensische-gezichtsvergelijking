import hashlib
import os
import pickle
from functools import wraps
from typing import List

import cv2
import pytest

from lr_face.data import (FaceImage,
                          FacePair,
                          FaceTriplet,
                          DummyFaceImage,
                          Dataset,
                          EnfsiDataset,
                          ForenFaceDataset,
                          LfwDataset,
                          make_pairs,
                          make_triplets,
                          to_array, split_by_identity)
from lr_face.models import Architecture
from lr_face.utils import fix_tensorflow_rtx
from tests.src.util import get_project_path, scratch_dir

fix_tensorflow_rtx()


def dataset_testable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        dataset = func(*args, **kwargs)
        if hasattr(dataset, 'RESOURCE_FOLDER'):
            dataset.RESOURCE_FOLDER = get_project_path(dataset.RESOURCE_FOLDER)
        return dataset

    return wrapper


def skip_if_missing_dataset(dataset: Dataset):
    if hasattr(dataset, 'RESOURCE_FOLDER'):
        root = get_project_path(dataset.RESOURCE_FOLDER)
        return pytest.mark.skipif(
            not os.path.exists(root),
            reason=f'{dataset} does not exist'
        )
    raise ValueError(
        'Cannot check for missing dataset without `RESOURCE_FOLDER` attr')


@pytest.fixture
def dummy_images() -> List[FaceImage]:
    ids = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    return [DummyFaceImage(path='', identity=f'TEST-{idx}') for idx in ids]


@pytest.fixture
def dummy_pairs() -> List[FacePair]:
    ids = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    images = [DummyFaceImage(path='', identity=f'TEST-{idx}') for idx in ids]
    return make_pairs(images)


@pytest.fixture
def dummy_triplets() -> List[FaceTriplet]:
    ids = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    images = [DummyFaceImage(path='', identity=f'TEST-{idx}') for idx in ids]
    return make_triplets(images)


@pytest.fixture
@dataset_testable
def enfsi_2011():
    return EnfsiDataset(years=[2011])


@pytest.fixture
@dataset_testable
def enfsi_all():
    return EnfsiDataset(years=[2011, 2012, 2013, 2017])


@pytest.fixture
@dataset_testable
def forenface():
    return ForenFaceDataset()


@pytest.fixture
@dataset_testable
def lfw():
    return LfwDataset()


@pytest.fixture()
def scratch():
    yield from scratch_dir('scratch/test_data')


###############
# `FaceImage` #
###############

def test_get_image(dummy_images, scratch):
    width = 100
    height = 50
    resolution = (height, width)
    image = dummy_images[0].get_image(resolution, normalize=False)
    image_path = os.path.join(scratch, 'tmp.jpg')
    cv2.imwrite(image_path, image)
    face_image = FaceImage(image_path, dummy_images[0].identity)
    reloaded_image = face_image.get_image(resolution)
    assert reloaded_image.shape == (*resolution, 3)


def test_get_vggface_embedding(dummy_images):
    architecture = Architecture.VGGFACE
    embedding = dummy_images[0].get_embedding(architecture)
    assert embedding.shape == (architecture.embedding_size,)


def test_get_vggface_embedding_is_deterministic(dummy_images, scratch):
    architecture = Architecture.VGGFACE
    image = dummy_images[0].get_image()
    embeddings = []

    # By saving and reloading the FaceImage 3 times with a different file name
    # we make sure to bypass any caching mechanisms.
    for i in range(3):
        image_path = os.path.join(scratch, f'tmp_{i}.jpg')
        cv2.imwrite(image_path, image)
        face_image = FaceImage(image_path, dummy_images[0].identity)
        embeddings.append(face_image.get_embedding(architecture))

    assert all(embeddings[0] == embeddings[1])
    assert all(embeddings[0] == embeddings[2])


def test_get_vggface_embedding_with_filesystem_caching(dummy_images, scratch):
    dummy_image = dummy_images[0]
    architecture = Architecture.VGGFACE
    output_path = os.path.join(
        scratch,
        architecture.name,
        f'{hashlib.md5(dummy_image.path.encode()).hexdigest()}.obj'
    )
    assert not os.path.exists(output_path)
    embedding = dummy_image.get_embedding(architecture, output_dir=scratch)
    assert os.path.exists(output_path)
    with open(output_path, 'rb') as f:
        cached_embedding = pickle.load(f)
    assert all(embedding == cached_embedding)


#################
# `FaceTriplet` #
#################

def test_face_triplet_raises_exception_on_wrong_identities():
    anchor = FaceImage('', identity='A')
    false_positive = FaceImage('', identity='B')
    true_positive = FaceImage('', identity='A')
    false_negative = FaceImage('', identity='A')
    true_negative = FaceImage('', identity='B')
    with pytest.raises(ValueError):
        FaceTriplet(anchor, false_positive, true_negative)
    with pytest.raises(ValueError):
        FaceTriplet(anchor, true_positive, false_negative)
    with pytest.raises(ValueError):
        FaceTriplet(anchor, false_positive, false_negative)


################
# `LfwDataset` #
################

@skip_if_missing_dataset(LfwDataset)
def test_lfw_dataset_has_correct_num_images(lfw):
    assert len(lfw.images) == 13233


@skip_if_missing_dataset(LfwDataset)
def test_lfw_dataset_has_correct_num_pairs(lfw):
    assert len(lfw.pairs) == 6000


##################
# `EnfsiDataset` #
##################

@skip_if_missing_dataset(EnfsiDataset)
def test_enfsi_dataset_has_correct_num_images(enfsi_all):
    assert len(enfsi_all.images) == 270


@skip_if_missing_dataset(EnfsiDataset)
def test_enfsi_dataset_has_correct_num_pairs(enfsi_all):
    assert len(enfsi_all.pairs) == 135
    assert all([a.meta['idx'] == b.meta['idx'] for a, b in enfsi_all.pairs])


###############
# `ForenFace` #
###############

@skip_if_missing_dataset(ForenFaceDataset)
def test_forenface_dataset_has_correct_num_images(forenface):
    assert len(forenface.images) == 2476


@skip_if_missing_dataset(ForenFaceDataset)
def test_forenface_dataset_has_correct_num_pairs(forenface):
    assert len(forenface.pairs) == 60798


##################
# `make_pairs()` #
##################

def test_make_pairs_positive_and_negative_no_n(dummy_images):
    pairs = make_pairs(dummy_images, same=None, n=None)
    positive_pairs = [p for p in pairs if p.same_identity]
    negative_pairs = [p for p in pairs if not p.same_identity]
    assert len(positive_pairs) == len(negative_pairs) == 8
    assert positive_pairs[0].first.identity == 'TEST-1'
    assert positive_pairs[1].first.identity == 'TEST-1'
    assert positive_pairs[2].first.identity == 'TEST-1'
    assert positive_pairs[3].first.identity == 'TEST-2'
    assert positive_pairs[4].first.identity == 'TEST-2'
    assert positive_pairs[5].first.identity == 'TEST-2'
    assert positive_pairs[6].first.identity == 'TEST-3'
    assert positive_pairs[7].first.identity == 'TEST-4'


def test_make_pairs_positive_and_negative_fixed_n(dummy_images):
    n = 8
    pairs = make_pairs(dummy_images, same=None, n=n)
    assert len(pairs) == n


def test_make_pairs_positive_and_negative_fixed_n_odd(dummy_images):
    n = 7
    pairs = make_pairs(dummy_images, same=None, n=n)
    assert len(pairs) == n


def test_make_pairs_positive_and_negative_large_n(dummy_images):
    n = 10000
    pairs = make_pairs(dummy_images, same=None, n=n)
    assert len(pairs) == 16


def test_make_pairs_positive_only_no_n(dummy_images):
    pairs = make_pairs(dummy_images, same=True, n=None)
    positive_pairs = [p for p in pairs if p.same_identity]
    assert len(positive_pairs) == len(pairs) == 8
    assert positive_pairs[0].first.identity == 'TEST-1'
    assert positive_pairs[1].first.identity == 'TEST-1'
    assert positive_pairs[2].first.identity == 'TEST-1'
    assert positive_pairs[3].first.identity == 'TEST-2'
    assert positive_pairs[4].first.identity == 'TEST-2'
    assert positive_pairs[5].first.identity == 'TEST-2'
    assert positive_pairs[6].first.identity == 'TEST-3'
    assert positive_pairs[7].first.identity == 'TEST-4'


def test_make_pairs_positive_only_fixed_n(dummy_images):
    n = 8
    pairs = make_pairs(dummy_images, same=True, n=n)
    positive_pairs = [p for p in pairs if p.same_identity]
    assert len(positive_pairs) == len(pairs) == 8


def test_make_pairs_positive_only_fixed_n_odd(dummy_images):
    n = 7
    pairs = make_pairs(dummy_images, same=True, n=n)
    positive_pairs = [p for p in pairs if p.same_identity]
    assert len(positive_pairs) == len(pairs) == 7


def test_make_pairs_positive_only_large_n(dummy_images):
    n = 10000
    pairs = make_pairs(dummy_images, same=True, n=n)
    positive_pairs = [p for p in pairs if p.same_identity]
    assert len(positive_pairs) == len(pairs) == 8


def test_make_pairs_negative_only_no_n(dummy_images):
    pairs = make_pairs(dummy_images, same=False, n=None)
    negative_pairs = [p for p in pairs if not p.same_identity]
    assert len(negative_pairs) == len(pairs) == 94
    assert negative_pairs[0].first.identity == 'TEST-1'
    assert negative_pairs[24].first.identity == 'TEST-2'
    assert negative_pairs[48].first.identity == 'TEST-3'
    assert negative_pairs[66].first.identity == 'TEST-4'
    assert negative_pairs[84].first.identity == 'TEST-5'


def test_make_pairs_negative_only_fixed_n(dummy_images):
    n = 48
    pairs = make_pairs(dummy_images, same=False, n=n)
    negative_pairs = [p for p in pairs if not p.same_identity]
    assert len(negative_pairs) == len(pairs) == n


def test_make_pairs_negative_only_fixed_n_odd(dummy_images):
    n = 47
    pairs = make_pairs(dummy_images, same=False, n=n)
    negative_pairs = [p for p in pairs if not p.same_identity]
    assert len(negative_pairs) == len(pairs) == n


def test_make_pairs_negative_only_large_n(dummy_images):
    n = 10000
    pairs = make_pairs(dummy_images, same=False, n=n)
    negative_pairs = [p for p in pairs if not p.same_identity]
    assert len(negative_pairs) == len(pairs) == n


##################
# `EnfsiDataset` #
##################

@skip_if_missing_dataset(EnfsiDataset)
def test_enfsi_dataset_has_correct_num_images(enfsi_all):
    assert len(enfsi_all.images) == 270


@skip_if_missing_dataset(EnfsiDataset)
def test_enfsi_dataset_has_correct_num_pairs(enfsi_all):
    assert len(enfsi_all.pairs) == 135
    assert all([a.meta['idx'] == b.meta['idx'] for a, b in enfsi_all.pairs])


###############
# `ForenFace` #
###############

@skip_if_missing_dataset(ForenFaceDataset)
def test_forenface_dataset_has_correct_num_images(forenface):
    assert len(forenface.images) == 2476


#####################
# `make_triplets()` #
#####################

def test_make_triplets_one_pair_per_identity():
    n = 10
    data = [DummyFaceImage('', str(i // 2)) for i in range(n)]
    triplets = make_triplets(data)
    assert len(triplets) == 5
    for i, (anchor, positive, negative) in enumerate(triplets):
        assert anchor is data[2 * i]
        assert positive is data[2 * i + 1]


def test_make_triplets_six_pairs_per_identity():
    n = 40
    data = [DummyFaceImage('', str(i // 4)) for i in range(n)]
    triplets = make_triplets(data)
    assert len(triplets) == 60


################
# `to_array()` #
################

def test_face_images_to_array(dummy_images):
    resolution = (50, 100)
    array = to_array(dummy_images, resolution=resolution)
    assert array.shape == (len(dummy_images), *resolution, 3)


def test_face_images_to_array_with_various_resolutions(dummy_images, scratch):
    face_images = []
    for i, dummy_image in enumerate(dummy_images):
        image = dummy_image.get_image(normalize=False)
        dimensions = (50 + i, 100)
        image = cv2.resize(image, dimensions)
        image_path = os.path.join(scratch, f'tmp_{i}.jpg')
        cv2.imwrite(image_path, image)
        face_images.append(FaceImage(image_path, dummy_images[0].identity))

    # Should raise an exception, because we do not allow `to_array` to accept
    # images of various shapes.
    with pytest.raises(ValueError):
        to_array(face_images)


def test_zero_face_images_to_array():
    with pytest.raises(ValueError):
        to_array([])


def test_face_pairs_to_array(dummy_pairs):
    resolution = (50, 100)
    left, right = to_array(dummy_pairs, resolution=resolution)
    expected_shape = (len(dummy_pairs), *resolution, 3)
    assert left.shape == expected_shape
    assert right.shape == expected_shape


def test_face_triplets_to_array(dummy_triplets):
    resolution = (50, 100)
    anchors, positives, negatives = to_array(
        dummy_triplets,
        resolution=resolution
    )
    expected_shape = (len(dummy_triplets), *resolution, 3)
    assert anchors.shape == expected_shape
    assert positives.shape == expected_shape
    assert negatives.shape == expected_shape


#########################
# `split_by_identity()` #
#########################

def test_split_by_identity(dummy_images):
    """
    Tests that `split_by_identity` results in two disjoint sets of images in
    terms of their identity.
    """

    # Since the results are random we run the test 10 times.
    for _ in range(10):
        train, test = split_by_identity(
            dummy_images,
            test_size=0.2
        )

        unique_train_identities = set(x.identity for x in train)
        unique_test_identities = set(x.identity for x in test)
        assert not unique_train_identities.intersection(unique_test_identities)

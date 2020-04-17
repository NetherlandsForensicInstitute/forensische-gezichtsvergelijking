import os
from functools import wraps

import pytest

from lr_face.data import make_triplets, DummyFaceImage, make_pairs, \
    EnfsiDataset, ForenFaceDataset, Dataset
from tests.src.util import get_project_path


def dataset_testable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        dataset = func(*args, **kwargs)
        if hasattr(dataset, 'RESOURCE_FOLDER'):
            dataset.RESOURCE_FOLDER = get_project_path(dataset.RESOURCE_FOLDER)
        return dataset

    return wrapper


def skip_if_missing(dataset: Dataset):
    if hasattr(dataset, 'RESOURCE_FOLDER'):
        root = get_project_path(dataset.RESOURCE_FOLDER)
        return pytest.mark.skipif(
            not os.path.exists(root),
            reason=f'{dataset} does not exist'
        )
    raise ValueError(
        'Cannot check for missing dataset without `RESOURCE_FOLDER` attr')


@pytest.fixture
def dummy_images():
    ids = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    return [DummyFaceImage(path='', identity=f'TEST-{idx}') for idx in ids]


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


################
# `LfwDataset` #
################

# TODO: add tests for LfwDataset.


##################
# `EnfsiDataset` #
##################

@skip_if_missing(EnfsiDataset)
def test_enfsi_dataset_has_correct_num_images(enfsi_all):
    assert len(enfsi_all.images) == 270


@skip_if_missing(EnfsiDataset)
def test_enfsi_dataset_has_correct_num_pairs(enfsi_all):
    assert len(enfsi_all.pairs) == 135
    assert all([a.meta['idx'] == b.meta['idx'] for a, b in enfsi_all.pairs])


###############
# `ForenFace` #
###############

@skip_if_missing(ForenFaceDataset)
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

# TODO: add tests for `to_array()`

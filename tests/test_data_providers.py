import numpy as np

from lr_face.data_providers import ImageWithIds, make_triplets


def test_make_triplets_one_pair_per_identity():
    n = 10
    data = ImageWithIds(
        images=[np.random.normal(size=(100, 100, 3)) for _ in range(n)],
        person_ids=[i // 2 for i in range(n)],
        image_ids=list(map(str, range(n))))

    triplets = make_triplets(data)
    assert len(triplets) == 5
    for i, triplet in enumerate(triplets):
        assert triplet.anchor is data.images[2 * i]
        assert triplet.positive is data.images[2 * i + 1]


def test_make_triplets_six_pairs_per_identity():
    n = 40
    data = ImageWithIds(
        images=[np.random.normal(size=(100, 100, 3)) for _ in range(n)],
        person_ids=[i // 4 for i in range(n)],
        image_ids=list(map(str, range(n))))

    triplets = make_triplets(data)
    assert len(triplets) == 60

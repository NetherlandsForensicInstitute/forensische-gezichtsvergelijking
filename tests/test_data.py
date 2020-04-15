from lr_face.data import FaceImage, make_triplets


def test_make_triplets_one_pair_per_identity():
    n = 10
    data = [FaceImage('', str(i // 2)) for i in range(n)]
    triplets = make_triplets(data)
    assert len(triplets) == 5
    for i, (anchor, positive, negative) in enumerate(triplets):
        assert anchor is data[2 * i]
        assert positive is data[2 * i + 1]


def test_make_triplets_six_pairs_per_identity():
    n = 40
    data = [FaceImage('', str(i // 4)) for i in range(n)]
    triplets = make_triplets(data)
    assert len(triplets) == 60

import hashlib
import os
import pickle
from typing import List

import cv2
import pytest

from lr_face.data import FaceImage, DummyFaceImage
from lr_face.models import Architecture
from lr_face.utils import fix_tensorflow_rtx
from tests.src.util import scratch_dir
from tests.test_architectures import skip_on_github

fix_tensorflow_rtx()


@pytest.fixture
def dummy_images() -> List[FaceImage]:
    ids = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    return [DummyFaceImage(path='', identity=f'TEST-{idx}') for idx in ids]


@pytest.fixture()
def scratch():
    yield from scratch_dir('scratch/test_models')


@skip_on_github
def test_get_vggface_embedding(dummy_images):
    architecture = Architecture.VGGFACE
    embedding_model = architecture.get_embedding_model()
    embedding = embedding_model.embed(dummy_images[0])
    assert embedding.shape == (architecture.embedding_size,)


@skip_on_github
def test_get_vggface_embedding_is_deterministic(dummy_images, scratch):
    architecture = Architecture.VGGFACE
    embedding_model = architecture.get_embedding_model()
    image = dummy_images[0].get_image()
    embeddings = []

    # By saving and reloading the FaceImage 3 times with a different file name
    # we make sure to bypass any caching mechanisms.
    for i in range(3):
        image_path = os.path.join(scratch, f'tmp_{i}.jpg')
        cv2.imwrite(image_path, image)
        face_image = FaceImage(image_path, dummy_images[0].identity)
        embeddings.append(embedding_model.embed(face_image))

    assert all(embeddings[0] == embeddings[1])
    assert all(embeddings[0] == embeddings[2])


@skip_on_github
def test_get_vggface_embedding_with_filesystem_caching(dummy_images, scratch):
    dummy_image = dummy_images[0]
    dummy_image.source = 'test'
    architecture = Architecture.VGGFACE
    embedding_model = architecture.get_embedding_model()

    def md5(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    cache_path = os.path.join(
        scratch,
        str(embedding_model),
        dummy_image.source,
        md5(dummy_image.path),
        f'{md5(f"{str(embedding_model)}{str(dummy_image)}{str(scratch)}")}.obj'
    )
    assert not os.path.exists(cache_path)
    embedding = embedding_model.embed(dummy_image, cache_dir=scratch)
    assert os.path.exists(cache_path)
    with open(cache_path, 'rb') as f:
        cached_embedding = pickle.load(f)
    assert all(embedding == cached_embedding)

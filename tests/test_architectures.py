import numpy as np

from lr_face.models import Architecture
from lr_face.utils import fix_tensorflow_rtx
from tests.conftest import skip_on_github

fix_tensorflow_rtx()


@skip_on_github
def test_get_embedding_models():
    """
    Tests whether no exceptions are thrown when we try to load the embedding
    models for each defined Architecture.
    """
    for architecture in Architecture:
        architecture.get_embedding_model()


@skip_on_github
def test_embedding_models_return_normalized_embeddings():
    """
    Tests whether the embedding model of each `Architecture` returns embeddings
    that are L2-normalized.
    """
    for architecture in Architecture:
        model = architecture.get_model()
        batch_input_shape = model.input_shape
        x = np.random.normal(size=(2, *batch_input_shape[1:]))
        embedding = model.predict(x)
        squared_sum = np.sum(np.square(embedding), axis=1)
        assert np.all((0.999 < squared_sum) & (squared_sum < 1.001)), \
            f"{architecture.value}'s embeddings are not properly L2-normalized"


@skip_on_github
def test_get_triplet_embedding_models():
    """
    Tests whether no exceptions are thrown when we try to load the finetune
    models for each defined Architecture.
    """
    for architecture in Architecture:
        architecture.get_triplet_embedding_model()


@skip_on_github
def test_resolution_vggface():
    assert Architecture.VGGFACE.resolution == (224, 224)


@skip_on_github
def test_embedding_size_vggface():
    assert Architecture.VGGFACE.embedding_size == 4096


@skip_on_github
def test_resolution_openface():
    assert Architecture.OPENFACE.resolution == (96, 96)


@skip_on_github
def test_embedding_size_openface():
    assert Architecture.OPENFACE.embedding_size == 128


@skip_on_github
def test_resolution_fbdeepface():
    assert Architecture.FBDEEPFACE.resolution == (152, 152)


@skip_on_github
def test_embedding_size_fbdeepface():
    assert Architecture.FBDEEPFACE.embedding_size == 4096


@skip_on_github
def test_resolution_facenet():
    assert Architecture.FACENET.resolution == (160, 160)


@skip_on_github
def test_embedding_size_facenet():
    assert Architecture.FACENET.embedding_size == 128


@skip_on_github
def test_resolution_arcface():
    assert Architecture.ARCFACE.resolution == (112, 112)


@skip_on_github
def test_embedding_size_arcface():
    assert Architecture.ARCFACE.embedding_size == 512


@skip_on_github
def test_resolution_lresnet():
    assert Architecture.LRESNET.resolution == (112, 112)


@skip_on_github
def test_embedding_size_lresnet():
    assert Architecture.LRESNET.embedding_size == 512


@skip_on_github
def test_resolution_ir50m1sm():
    assert Architecture.IR50M1SM.resolution == (112, 112)


@skip_on_github
def test_embedding_size_ir50m1sm():
    assert Architecture.IR50M1SM.embedding_size == 512


@skip_on_github
def test_resolution_ir50asia():
    assert Architecture.IR50ASIA.resolution == (112, 112)


@skip_on_github
def test_embedding_size_ir50asia():
    assert Architecture.IR50ASIA.embedding_size == 512

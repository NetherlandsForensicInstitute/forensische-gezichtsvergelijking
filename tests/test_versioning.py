import pytest

from lr_face.versioning import Tag


def test_create_tag_from_string():
    tag = Tag('tag:1')
    assert tag.name == 'tag'
    assert tag.version == 1


def test_create_tag_from_string_without_version():
    tag = Tag('tag')
    assert tag.name == 'tag'
    assert tag.version is None


def test_create_tag_from_string_double_colon():
    with pytest.raises(ValueError):
        Tag('tag:2:1')


def test_get_version_from_filename():
    filename = 'myfile-mytag-2.txt'
    assert Tag.get_version_from_filename(filename) == 2


def test_get_tag_from_filename():
    filename = 'myfile-mytag_simple-2.txt'
    tag = Tag.from_filename(filename)
    assert tag.name == 'mytag_simple'
    assert tag.version == 2


def test_get_tag_from_filename_without_version():
    filename = 'myfile-mytag_simple.txt'
    tag = Tag.from_filename(filename)
    assert tag.name == 'mytag_simple'
    assert tag.version is None


def test_append_to_filename():
    filename = 'myfile.txt'
    tag = Tag('mytag_simple', 2)
    tagged_filename = tag.append_to_filename(filename)
    assert tagged_filename == 'myfile-mytag_simple-2.txt'

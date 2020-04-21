from lr_face.versioning import Version


def test_version_from_filename():
    version = Version.from_filename("file_7.5.1.jpg")
    assert version.major == 7
    assert version.minor == 5
    assert version.micro == 1


def test_version_from_string():
    version = Version.from_string("7.5.1")
    assert version.major == 7
    assert version.minor == 5
    assert version.micro == 1


def test_version_comparison_equal():
    a = Version(7, 5, 1)
    b = Version(7, 5, 1)
    assert a == b


def test_version_comparison_major_greater_than():
    a = Version(8, 0, 0)
    b = Version(7, 5, 1)
    assert a > b


def test_version_comparison_minor_greater_than():
    a = Version(7, 6, 0)
    b = Version(7, 5, 1)
    assert a > b

    a = Version(6, 6, 0)
    b = Version(7, 5, 1)
    assert a < b


def test_version_comparison_micro_greater_than():
    a = Version(7, 5, 2)
    b = Version(7, 5, 1)
    assert a > b

    a = Version(7, 4, 9)
    b = Version(7, 5, 1)
    assert a < b


def test_get_latest_version():
    versions = [Version(7, 5, 2),
                Version(1, 6, 9),
                Version(7, 8, 0),
                Version(6, 8, 3),
                Version(7, 2, 10),
                Version(0, 0, 1)]

    assert max(versions) == versions[2]

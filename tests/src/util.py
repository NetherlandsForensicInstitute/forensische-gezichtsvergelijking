import os
import shutil
from typing import Generator


def get_tests_path(relative_path: str) -> str:
    """
    Takes a resource path relative to the root `tests` directory and returns
    an absolute path to it.

    :param relative_path: str
    :return: str
    """
    tests_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(tests_path, relative_path)


def scratch_dir(path: str) -> Generator[str, None, None]:
    """
    Creates a temporary directory that can be used in a test fixture and is
    removed after the test concludes. The path to this temporary directory is
    yielded so the test knows where to find it.

    :param path: str, the path to the directory that should be created
    :return: Generator[str, None, None]
    """
    if not os.path.isabs(path):
        path = get_tests_path(path)
    try:
        os.makedirs(path, exist_ok=True)
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

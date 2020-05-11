from pathlib import Path

import pytest

from lr_face.models import Architecture
from lr_face.utils import cache

# When running our tests, cache the models to prevent OOM errors.
Architecture.get_model = cache(Architecture.get_model)


def skip_on_github(func):
    return pytest.mark.skipif(
        str(Path.home()) == '/home/runner',
        reason="Fails on Github because model weights don't exist"
    )(func)

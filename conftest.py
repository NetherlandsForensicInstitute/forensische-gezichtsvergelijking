from lr_face.models import Architecture
from lr_face.utils import cache

# When running our tests, cache the models to prevent OOM errors.
Architecture.get_model = cache(Architecture.get_model)

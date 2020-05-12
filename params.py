import numpy as np
from lir import (LogitCalibrator,
                 NormalizedCalibrator,
                 ELUBbounder,
                 KDECalibrator,
                 FractionCalibrator,
                 IsotonicCalibrator,
                 DummyCalibrator)

from lr_face.data import (TestDataset,
                          EnfsiDataset,
                          LfwDataset,
                          LfwDevDataset,
                          SCDataset, ForenFaceDataset)
from lr_face.models import Architecture
from lr_face.utils import fix_tensorflow_rtx

fix_tensorflow_rtx()

"""How often to repeat all experiments"""
TIMES = 10

""" 
New datasets can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""
DATA = {
    'current_set_up': ['lfw_sanity_check'],
    'all': {
        # Either specify a single dataset as `datasets`, in which case the
        # dataset is split into calibration and test pairs according to the
        # specified `fraction_test`, or specify a tuple of 2 datasets, in which
        # case the pairs from the first dataset are used for calibration and
        # the pairs from the second dataset are used for testing.
        'test': {
            'datasets': TestDataset(),
            'fraction_test': .5,
        },
        'enfsi': {
            'datasets': EnfsiDataset(years=[2011, 2012, 2013, 2017]),
            'fraction_test': .2,
        },
        'enfsi2011': {
            'datasets': EnfsiDataset(years=[2011]),
            'fraction_test': .2,
        },
        'enfsi2012': {
            'datasets': EnfsiDataset(years=[2012]),
            'fraction_test': .2,
        },
        'enfsi2013': {
            'datasets': EnfsiDataset(years=[2013]),
            'fraction_test': .2,
        },
        'enfsi2017': {
            'datasets': EnfsiDataset(years=[2017]),
            'fraction_test': .2,
        },
        'lfw': {
            'datasets': LfwDataset(),
            'fraction_test': .9,
        },
        'SC': {
            'datasets': SCDataset(image_types=['frontal',
                                               'rotated',
                                               'surveillance']),
            'fraction_test': .9,
        },
        'lfw_sanity_check': {
            'datasets': (LfwDevDataset(True), LfwDevDataset(False)),
            'fraction_test': None  # Can be omitted if `datasets` is a tuple.
        },
        'lfw_enfsi': {
            'datasets': (LfwDevDataset(True), EnfsiDataset(years=[2011, 2012, 2013, 2017])),
            'fraction_test': None  # Can be omitted if `datasets` is a tuple.
        },
        'forenface': {
            'datasets': ForenFaceDataset(),
            'fraction_test': .5,
        }
    }
}

"""
New models/scorers can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""

SCORERS = {
    'current_set_up': ['vggface'],
    'all': {
        # We apply lazy loading to the scorer models since they take up a lot
        # of memory. Each setup has type `Tuple[Architecture, Optional[str]]`.
        # To pin a specific version of a tag, use a colon (':') as a delimiter,
        # e.g. 'my_tag:2'. If no version is specified, the latest version is
        # used by default.
        'dummy': (Architecture.DUMMY, None),
        'openface': (Architecture.OPENFACE, None),
        'facenet': (Architecture.FACENET, None),
        'fbdeepface': (Architecture.FBDEEPFACE, None),
        'vggface': (Architecture.VGGFACE, None),
        'arcface': (Architecture.ARCFACE, None),
        'vggface_lfw_resized': (Architecture.VGGFACE, 'lfw_resized'),
    }
}

""" 
New calibrators can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""
CALIBRATORS = {
    'current_set_up': ['logit'],
    'all': {
        'logit': LogitCalibrator(),
        'logit_normalized': NormalizedCalibrator(LogitCalibrator()),
        'KDE': KDECalibrator(),
        'elub_KDE': ELUBbounder(KDECalibrator()),
        'dummy': DummyCalibrator(),
        'fraction': FractionCalibrator(),
        'isotonic': IsotonicCalibrator(add_one=True)
    }
}

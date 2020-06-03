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
TIMES = 1

"""
Parameters to be used in an experiment, different/new sets can be added under 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""
PARAMS = {
    'current_set_up': ['scenario_1'],
    'all': {
        'scenario_1': {
            'calibration_filters': [],
        },
        'scenario_2': {
            'calibration_filters': ['quality_score'],
        },
        'scenario_3': {
            'calibration_filters': ['yaw', 'pitch', 'other_occlusions', 'resolution_bin'],
        },
    }
}


""" 
New datasets can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""
DATA = {
    'current_set_up': ['enfsi'],
    'all': {
        # specify both calibration and test as a tuple of datasets
        'test': {
            'calibration': (TestDataset(),),
            'test': (TestDataset(),),
        },
        'dev': {
            'calibration': (EnfsiDataset(years=[2011, 2012]),),
            'test': (EnfsiDataset(years=[2011, 2012]),),
        },
        'forenface_enfsi_sc': {
            'calibration': (ForenFaceDataset(),
                            EnfsiDataset(years=[2011, 2012, 2013, 2017]),
                            SCDataset(image_types=['frontal',
                                                   'rotated',
                                                   'surveillance'])
                            ),
            'test': (EnfsiDataset(years=[2011, 2012, 2013, 2017]),),
        },
        'enfsi': {
            'calibration': (EnfsiDataset(years=[2011, 2012, 2013, 2017]),),
            'test': (EnfsiDataset(years=[2011, 2012, 2013, 2017]),),
        },
        'lfw': {
            'calibration': (LfwDataset(),),
            'test': (LfwDataset(),),
        },
        'SC': {
            'calibration': (SCDataset(image_types=['frontal',
                                               'rotated',
                                               'surveillance']),),
            'test': (SCDataset(image_types=['frontal',
                                               'rotated',
                                               'surveillance']),),
        },
        'lfw_sanity_check': {
            'calibration': (LfwDevDataset(True), ),
            'test': (LfwDevDataset(False),),
        },
    }
}

"""
New models/scorers can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""

SCORERS = {
    'current_set_up': ['face_recognition'],
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
        'keras_vggface': (Architecture.KERAS_VGGFACE, None),
        'keras_vggface_resnet': (Architecture.KERAS_VGGFACE_RESNET, None),  # Don't use yet, terrible performance
        'arcface': (Architecture.ARCFACE, None),
        'lresnet': (Architecture.LRESNET, None),
        'ir50m1sm': (Architecture.IR50M1SM, None),
        'ir50asia': (Architecture.IR50ASIA, None),
        'face_recognition': (Architecture.FACERECOGNITION, None),
        'lfw_sanity_check': (Architecture.VGGFACE, 'lfw_resized_50'),
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

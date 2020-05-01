import numpy as np
from lir import (LogitCalibrator,
                 NormalizedCalibrator,
                 ELUBbounder,
                 KDECalibrator,
                 FractionCalibrator,
                 IsotonicCalibrator,
                 DummyCalibrator)

from lr_face.data import TestDataset, EnfsiDataset, LfwDataset, \
    ForenFaceDataset
from lr_face.models import DummyScorerModel, Architecture
from lr_face.utils import fix_tensorflow_rtx

fix_tensorflow_rtx()

"""How often to repeat all experiments"""

TIMES = 10

"""
Parameters to be used in an experiment, different/new sets can be added under 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""
PARAMS = {

    'current_set_up': ['SET1'],
    'all': {
        'SET1': {
            'fraction_training': 0.6,
            'n_datapoints_test': 20,
            'transform_scorer_output': False,
            'train_calibration_same_data': False,
        },
        'SET2': {
            'fraction_training': 0.6,
            'n_datapoints_test': 1000,
            'transform_scorer_output': False,
            'train_calibration_same_data': False,
        },
        'calibrate_same1': {
            'fraction_training': .5,
            'n_datapoints_test': 30,
            'transform_scorer_output': False,
            'train_calibration_same_data': [True, False],
        },
        'fraction1': {
            'fraction_training': list(np.arange(0.1, 1.0, 0.1)),
            'n_datapoints_test': 50,
            'transform_scorer_output': False,
            'train_calibration_same_data': False
        },
    }
}

DATA = {
    'current_set_up': ['enfsi'],
    'all': {
        'test': {
            'datasets': [TestDataset()],
            'fraction_test': .5,
        },
        'enfsi': {
            'datasets': [EnfsiDataset(years=[2011, 2012, 2013, 2017])],
            'fraction_test': .5,
        },
        'enfsi-separate': {
            'datasets': [
                EnfsiDataset(years=[2011]),
                EnfsiDataset(years=[2012]),
                EnfsiDataset(years=[2013]),
                EnfsiDataset(years=[2017])],
            'fraction_test': .5,
        },
        'lfw': {
            'datasets': [LfwDataset()],
            'fraction_test': .9,
        },
        'forenface': {
            'datasets': [ForenFaceDataset()],
            'fraction_test': .5,
        }
    }
}

"""
New models/scorers can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""
SCORERS = {
    'current_set_up': ['dummy',
                       'openface',
                       'facenet',
                       'vggface',
                       'fbdeepface'],
    'all': {
        'dummy': DummyScorerModel(),
        # TODO: specify tags to use below.
        'openface': Architecture.OPENFACE.get_scorer_model(tag=None),
        'facenet': Architecture.FACENET.get_scorer_model(tag=None),
        'fbdeepface': Architecture.FBDEEPFACE.get_scorer_model(tag=None),
        'vggface': Architecture.VGGFACE.get_scorer_model(tag=None),
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

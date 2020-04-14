from functools import partial
import numpy as np

from lir import LogitCalibrator, NormalizedCalibrator, ELUBbounder, KDECalibrator, FractionCalibrator, \
    IsotonicCalibrator, DummyCalibrator


from deepface.basemodels import VGGFace, FbDeepFace, Facenet, OpenFace
from lr_face.models import DummyModel, Deepface_Lib_Model
from lr_face.data_providers import test_data, enfsi_data, lfw_data, combine_paired_data, DataFunctions


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
            'datasets': [DataFunctions(image_provider=test_data, pair_provider=None)],
            'fraction_test': .5,
        },
        'enfsi': {
            'datasets': [DataFunctions(image_provider=None,
                                       pair_provider=partial(combine_paired_data, pair_providers=[
                                           partial(enfsi_data, year=2011),
                                           partial(enfsi_data, year=2012),
                                           partial(enfsi_data, year=2013),
                                           partial(enfsi_data, year=2017),
                                                                          ]))],
            'fraction_test': .2,
        },
        'lfw': {
            'datasets': [DataFunctions(image_provider=None,
                                       pair_provider=partial(combine_paired_data, pair_providers=[
                                           partial(lfw_data),
                                                                          ]))],
            'fraction_test': .2,
        }
    }
}

"""
New models/scorers can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""
SCORERS = {
    'current_set_up': ['openface', 'facenet', 'vggface', 'fbdeepface', 'dummy'],
    'all': {
        'dummy': DummyModel(),
        'openface': Deepface_Lib_Model(model=OpenFace.loadModel()),
        'facenet': Deepface_Lib_Model(model=Facenet.loadModel()),
        'fbdeepface': Deepface_Lib_Model(model=FbDeepFace.loadModel()),
        'vggface': Deepface_Lib_Model(model=VGGFace.loadModel())
    }
}

""" 
New calibrators can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated.
"""
CALIBRATORS = {
    'current_set_up': ['KDE'],
    'all': {
        'logit': LogitCalibrator(),
        'logit_normalized': NormalizedCalibrator(LogitCalibrator()),
        'KDE': KDECalibrator(),
        'elub_KDE': ELUBbounder(KDECalibrator()),
        'dummy': DummyCalibrator(),
        'fraction': FractionCalibrator(),
        'isotonic': IsotonicCalibrator()
    }
}

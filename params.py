from functools import partial
import numpy as np

from lir import LogitCalibrator, NormalizedCalibrator, ELUBbounder, KDECalibrator, FractionCalibrator, \
    IsotonicCalibrator, DummyCalibrator


from deepface.deepface.basemodels import VGGFace, FbDeepFace, Facenet, OpenFace
from lr_face.models import DummyModel, Deepface_Lib_Model, InsightFace_Model #Classes
from lr_face.data_providers import test_data, enfsi_data, combine_data

from insightface.deploy import face_model_args





"""How often to repeat all experiments"""

TIMES = 10

"""
Parameters to be used in an experiment, different/new sets can be added under 'all'
For the input of an experiment the 'current_set_up' list can be updated
"""
PARAMS = {

    'current_set_up': ['calibrate_same1'],
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
    'current_set_up': ['test'],
    'all': {
        'test': {
            'dataset_callable': [test_data],
            'fraction_test': .5,
        },
        'enfsi': {
            #TODO currently every element becomes a new experiment, we probably want functionality to combine datasets
            'dataset_callable': partial(combine_data, dataset_callables=[partial(enfsi_data, year=2011),
                                 partial(enfsi_data, year=2012),
                                 partial(enfsi_data, year=2013),
                                 partial(enfsi_data, year=2017)]),
            'fraction_test': .5,
        }
    }
}

"""
New models/scorers can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated
"""
SCORERS = {
    'current_set_up': ['dummy'],
    'all': {
        'dummy': DummyModel(),
        'openface': Deepface_Lib_Model(model=OpenFace.loadModel()),
        'facenet': Deepface_Lib_Model(model=Facenet.loadModel()),
        'fbdeepface': Deepface_Lib_Model(model=FbDeepFace.loadModel()),
        'vggface': Deepface_Lib_Model(model=VGGFace.loadModel()), #anadir coma luego
        'insightface': InsightFace_Model(model=face_model_args.loadModel())
    }
}

""" 
New calibrators can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated
"""
CALIBRATORS = {
    'current_set_up': ['elub_KDE'],
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

import functools
import numpy as np

from lir import LogitCalibrator, NormalizedCalibrator, ELUBbounder, KDECalibrator, FractionCalibrator, \
    IsotonicCalibrator, DummyCalibrator
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from lr_face.data_providers import test_data
from lr_face.testmodel import DummyModel, OpenFace

"""How often to repeat all experiments"""

TIMES = 50

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
        'rna1': {
            'fraction_training': 0.6,
            'n_datapoints_test': 50,
            'transform_scorer_output': False,
            'train_calibration_same_data': False,
        },
        'calibrate_same1': {
            'fraction_training': .5,
            'n_datapoints_test': 30,
            'transform_scorer_output': False,
            'train_calibration_same_data': [True, False],
        },
        'calibrate_same2': {
            'fraction_training': .5,
            'n_datapoints_test': 1000,
            'transform_scorer_output': False,
            'train_calibration_same_data': True,
        },
        'fraction1': {
            'fraction_training': list(np.arange(0.1, 1.0, 0.1)),
            'n_datapoints_test': 50,
            'transform_scorer_output': False,
            'train_calibration_same_data': False
        },
        'fraction2': {
            'fraction_training': list(np.arange(0.1, 1.0, 0.1)),
            'n_datapoints_test': 100,
            'transform_scorer_output': False,
            'train_calibration_same_data': True
        },
        'data_mismatch1': {
            'fraction_training': 0.6,
            'n_datapoints_test': 1000,
            'transform_scorer_output': False,
            'train_calibration_same_data': False
        }
    }
}

DATA = {
    'current_set_up': ['test'],
    'all': {
        'test': {
            'dataset_callable': test_data,
            'fraction_test': .5,
        }
    }
}

"""
New models/scorers can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated
"""
SCORERS = {
    'current_set_up': ['test', 'facenet'],
    'all': {
        'test': DummyModel(),
        'facenet': OpenFace()
    }
}

""" 
New calibrators can be added to 'all'.
For the input of an experiment the 'current_set_up' list can be updated
"""
CALIBRATORS = {
    'current_set_up': ['logit', 'elub_KDE'],
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

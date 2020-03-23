from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class Images:
    y_train: np.ndarray
    X_train: np.ndarray
    y_calibrate: np.ndarray
    X_calibrate: np.ndarray
    y_test: np.ndarray
    X_test: np.ndarray


def test_data(resolution):
    """
    return some random numbers in the right structure to test the pipeline with
    """
    return np.random.random([11, resolution[0], resolution[1], 3]), np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5])


def get_data(dataset_callable, resolution=(100, 100),
             train_calibration_same_data=True, fraction_calibration=None, fraction_test=0.2) -> Images:
    """
    Takes a list of functions that return X, y, with X images and y identities. Returns a dataset with all data
    split into the right datasets

    fraction_calibration is the fraction of train data that is used for calibration,
        so fraction_calibration + fraction_test can be > 1

    """

    X = np.zeros((0, resolution[0], resolution[1], 3))
    y = np.array([])
    this_X, this_y = dataset_callable(resolution)
    assert this_X.shape[1:3] == resolution, f
    'resolution should be {resolution}, not {this_X.shape[:2]}'
    assert this_X.shape[0] == len(this_y), f
    'y and X should have same length'

    X = np.concatenate((X, this_X), axis=0)
    y = np.append(y, this_y)

    # split on identities, not on samples (so same person does not appear in both test and train
    X_train, X_test, y_train, y_test = split_data_on_groups(X, fraction_test, y)

    if train_calibration_same_data:
        X_calibrate = X_train
        y_calibrate = y_train
    else:
        X_train, X_calibrate, y_train, y_calibrate = split_data_on_groups(X_train, fraction_calibration, y_train)
    return Images(y_train=y_train, X_train=X_train, y_test=y_test, X_test=X_test, y_calibrate=y_calibrate,
                  X_calibrate=X_calibrate)


def split_data_on_groups(X, fraction2, y):
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction2, random_state=42)
    for train_idx, test_idx in gss.split(X, y, y):
        X1 = X[train_idx]
        y1 = y[train_idx]
        X2 = X[test_idx]
        y2 = y[test_idx]
    return X1, X2, y1, y2


def make_pairs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    takes images X and classes y, and returns a paired data and a vector indicating same or different source

    Example
    [x1, .., x9], [1,2,3,4,5,6,7,8,9]
    ->
    [[x1,x2], [x1,x3], [x1,x2], [x1, x4], [x1, x5], ...], [1, 1, 1, 0, 0, ...]

    Currently makes different sources only by pairing class n with n+1 rather than taking all ~N^2 possible pairs,
    to keep data sets limited.
    """
    person_ids = np.unique(y)
    pairs = []
    same_different_source = []
    for i_person_id, person_id in enumerate(person_ids):
        idx = y == person_id
        nidx = sum(idx)
        # all images of this person
        imgs = X[idx]
        for i in range(nidx):
            for j in range(i + 1, nidx):
                # make same-person pairs by pairing all images of the person
                pairs.append((imgs[i], imgs[j]))
                same_different_source.append(1)
            if i_person_id > 0:
                # make different-person pairs by pairing person i with person i-1
                for j in range(len(imgs_prev)):
                    pairs.append((imgs[i], imgs_prev[j]))
                    same_different_source.append(0)

        imgs_prev = imgs
    return np.array(pairs), np.array(same_different_source)

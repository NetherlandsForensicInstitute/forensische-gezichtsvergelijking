import csv
import os
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Tuple, List

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy import stats
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from lir import Xn_to_Xy, Xy_to_Xn
from lr_face.utils import get_test_data




def assert_enough_data(train_data, test_data, available_data):
    assert train_data + test_data <= available_data, \
        f'less data than minimum amount required {train_data} + {test_data}, got {available_data}'


def make_X_arrays(df, item_column, features):
    """
    make the X_same and X_different arrays

    :param df: data to use
    :param item_column: name of item number column
    :param features: features is df to include in arrays
    :return: X1 and X2 arrays
    """
    rows = len(df)

    combs = combinations(np.arange(0, rows), r=2)
    indices = [(c[0], c[1]) for c in combs]
    index_one = [c[0] for c in indices]
    index_two = [c[1] for c in indices]

    X = abs(df.iloc[index_one, :][features].to_numpy() - df.iloc[index_two, :][features].to_numpy())
    mask = df.iloc[index_one, :][item_column].to_numpy() == df.iloc[index_two, :][item_column].to_numpy()

    X_same = X[mask, :]
    X_different = X[~mask, :]

    return X_same, X_different


def get_rna_data(cell_type=None, **kwargs):
    """
    gets the rna data in a X, y format
    for simplicity, each replicate is treated is it's own sample
    :param classes: list containing the classes to compare
    :return: (X1,X2)
    """
    assert cell_type, 'cell_type was not defined'

    file = 'Dataset_NFI_rv.xlsx'

    rna_folder = os.path.join('resources', 'rna')
    df = pd.read_excel(os.path.join(rna_folder, file), sep='/')
    df = get_one_vs_rest(df, cell_type)

    X1, X2 = Xy_to_Xn(np.array(df.iloc[:, 1:]), np.array(df.iloc[:, 0]))

    return X1, X2


def get_event_times_samenreizen(h1_distribution):
    """
    generates even times for samenreizen data

    :param h1_distribution: Cotravel instance with h1 distribution
    :return: list of event times
    """
    event_times = []
    for i in range(2):
        event_times.append(np.cumsum(np.random.exponential(scale=24 / h1_distribution.samples_per_day,
                                                           size=h1_distribution.number_of_days * h1_distribution.samples_per_day * 4)))
    return event_times




def get_one_vs_rest(df, cell_type):
    """
    gets the main cell_type and other cell_types data from df
    :param df: pandas df
    :param cell_type:: string, indicating cell_type of main class
    :return: pandas df with main cell_type data and sampled other cell_type data
    """
    main = df[df.iloc[:, 0] == cell_type].fillna(0)
    rest = sample_rest(df, cell_type)
    return pd.concat([main, rest], axis=0)


def sample_rest(df, cell_type):
    """
    samples n_samples equal to n_samples cell_type from other cell_types in df
    :param df: pandas df
    :param cell_type: string, indicating cell_type of main class
    :param n_samples: number of samples in main class
    :return: pandas df with sampled other cell_type data, indicated as class rest
    """
    rest = df[df.iloc[:, 0] != cell_type].fillna(0)
    rest.iloc[:, 0] = 'rest'
    return rest

@dataclass
class Images:
    y_train: np.ndarray
    X_train: np.ndarray
    y_calibrate: np.ndarray
    X_calibrate: np.ndarray
    y_test: np.ndarray
    X_test: np.ndarray

def test_data(resolution):
    return np.random.random([11, resolution[0], resolution[1], 3]), np.array([1,1,1, 2, 2, 2, 3, 3, 4, 4, 5])

def get_data(dataset_callable, resolution=(100,100),
             train_calibration_same_data=True, fraction_calibration=None, fraction_test=0.2):
    """
    Takes a list of functions that return X, y, with X images and y identities. Returns a dataset with all images
    split into the right datasets

    fraction_calibration is the fraction of train data that is used for calibration

    """

    X=np.zeros((0, resolution[0], resolution[1], 3))
    y=np.array([])
    this_X, this_y = dataset_callable(resolution)
    assert this_X.shape[1:3] == resolution, f'resolution should be {resolution}, not {this_X.shape[:2]}'
    assert this_X.shape[0] == len(this_y), f'y and X should have same length'

    X= np.concatenate((X, this_X), axis=0)
    y = np.append(y, this_y)

    # split on identities, not on samples (so same person does not appear in both test and train
    X_train, X_test, y_train, y_test = split_data_on_groups(X, fraction_test, y)

    if train_calibration_same_data:
        X_calibrate = X_train
        y_calibrate = y_train
    else:
        X_train, X_calibrate, y_train, y_calibrate = split_data_on_groups(X_train, fraction_calibration, y_train)
    return Images(y_train=y_train, X_train=X_train, y_test=y_test, X_test=X_test, y_calibrate=y_calibrate, X_calibrate=X_calibrate)


def split_data_on_groups(X, fraction2, y):
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction2, random_state=42)
    for train_idx, test_idx in gss.split(X, y, y):
        X1 = X[train_idx]
        y1 = y[train_idx]
        X2 = X[test_idx]
        y2 = y[test_idx]
    return X1, X2, y1, y2


def make_pairs(X: np.ndarray, y:np.ndarray):
    """
    takes images, identities, and returns a list of pairs, 0/1 different same source
    :param X: np
    :param y:
    :return:
    """
    image_ids = np.unique(y)
    pairs = []
    same_different_source = []
    for i_id, image_id in enumerate(image_ids):
        idx = y==image_id
        nidx = sum(idx)
        imgs = X[idx]
        for i in range(nidx):
            for j in range(i+1, nidx):
                pairs.append((imgs[i], imgs[j]))
                same_different_source.append(1)
            if i_id>0:
                # make different source with previous
                for j in range(len(imgs_prev)):
                    pairs.append((imgs[i], imgs_prev[j]))
                    same_different_source.append(0)

        imgs_prev = imgs
    return np.array(pairs), np.array(same_different_source)
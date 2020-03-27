import csv
from dataclasses import dataclass
from typing import Tuple, List
import os

import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class ImagePairs:
    y_calibrate: np.ndarray
    X_calibrate: np.ndarray
    ids_calibrate: List
    y_test: np.ndarray
    X_test: np.ndarray
    ids_test: List


def test_data(resolution=(100,100)):
    """
    return some random numbers in the right structure to test the pipeline with
    """
    return np.random.random([11, resolution[0], resolution[1], 3]), np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5])


def enfsi_data(resolution, year) -> Tuple[List,List, List]:
    folder = os.path.join('resources', 'enfsi', str(year))
    files = os.listdir(folder)
    n_pairs = (len(files)-1)//2
    X = [[-1, -1] for _ in range(n_pairs)]
    y = [-1]*n_pairs
    ids=[-1]*n_pairs
    df = pd.read_csv(os.path.join(folder, 'truth.csv')).set_index('id')
    for file in files:
        if not file.endswith('csv'):
            # TODO check RGB/GRB ordering/colous usage on models.
            img = cv2.imread(os.path.join(folder, file), cv2.COLOR_BGR2RGB)
            questioned_ref=None
            if year == 2011:
                cls = int(file[:3])
                questioned_ref = file[3]
            elif year == 2012:
                cls = int(file[:2])
                questioned_ref = file[2]
            elif year == 2013:
                cls = int(file[1:3])
                questioned_ref = file[0]
            elif year == 2017:
                cls = int(file[1:3])
                questioned_ref = file[0]
            else:
                raise ValueError(f'Unknown ENFSI year {year}')
            y[cls-1]=int(df.loc[cls]['same']==1)
            ids[cls-1] = f"enfsi_{year}_{cls}_{int(df.loc[cls]['same']==1)}"
            if questioned_ref=='q':
                X[cls-1][0]=img
            elif questioned_ref=='r':
                X[cls-1][1]=img
            else:
                raise ValueError(f'unknown questioned/ref: {questioned_red}')
    return X, y, ids


def combine_unpaired_data(dataset_callables, resolution) -> Tuple[np.ndarray, np.ndarray]:
    """
    gets the X and y for all data in the callables, and returns the total set
    """
    X = np.array([])
    y = np.array([])
    max_class = 0
    for dataset_callable in dataset_callables:
        this_X, this_y = dataset_callable(resolution)
        y = np.append(y, this_y+max_class)
        max_class = max(y)
        X = np.append(X, this_X)
    return X, y.astype(int)

def combine_pairs(dataset_callables, resolution) -> Tuple[List,List, List]:
    """
    gets the pairs for all data in the callables, and returns the total set
    """
    X = []
    y = []
    ids=[]
    for dataset_callable in dataset_callables:
        this_X, this_y, this_ids = dataset_callable(resolution)
        y +=this_y
        X +=this_X
        ids = ids + this_ids
    assert len(ids) == len(set(ids))
    if X:
        assert len(X[0]) == 2
    return X, y, ids

def get_data(dataset_callable, resolution=(100, 100), fraction_test=0.2) -> ImagePairs:
    """
    Takes a function that returns X, y, with X either images or pairs of images and y identities. Returns a dataset with all data
    split into the right datasets


    """

    X, y, ids = dataset_callable(resolution=resolution)
    # TODO for now we will let the model resize, in future we should enforce the right resolution to come from preprocessing
    # assert this_X.shape[1:3] == resolution, f'resolution should be {resolution}, not {this_X.shape[:2]}'
    assert len(X) == len(y), f'y and X should have same length'


    if len(X[0])==2:
        # these are pairs
        X_calibrate, X_test, y_calibrate, y_test, ids_calibrate, ids_test = train_test_split(X, y, ids, test_size=fraction_test, stratify=y)

    else:
        # split on identities, not on samples (so same person does not appear in both test and train
        X_calibrate, X_test, y_calibrate, y_test = split_data_on_groups(X, fraction_test, y)

        # make pairs per set
        X_test, y_test, ids_test = make_pairs(X_test,y_test)
        X_calibrate, y_calibrate, ids_calibrate = make_pairs(X_calibrate,y_calibrate)
        assert len(ids_test+ids_calibrate) == len(set(ids_test+ids_calibrate))
    return ImagePairs(y_test=y_test, X_test=X_test, ids_test=ids_test, y_calibrate=y_calibrate,
                  X_calibrate=X_calibrate, ids_calibrate=ids_calibrate)


def split_data_on_groups(X, fraction2, y):
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction2, random_state=42)
    for train_idx, test_idx in gss.split(X, y, y):
        X1 = X[train_idx]
        y1 = y[train_idx]
        X2 = X[test_idx]
        y2 = y[test_idx]
    return X1, X2, y1, y2


def make_pairs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
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
    ids=[]
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
                ids.append(hash((imgs[i], imgs[j])))
            if i_person_id > 0:
                # make different-person pairs by pairing person i with person i-1
                for j in range(len(imgs_prev)):
                    pairs.append((imgs[i], imgs_prev[j]))
                    same_different_source.append(0)
                    ids.append(hash((imgs[i], imgs[j])))

        imgs_prev = imgs
    return np.array(pairs), np.array(same_different_source), ids

import argparse
import os
import re
from functools import lru_cache

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image


def write_output(df, experiment_name):
    # %H:%M:%S -> : (colon) werkt niet in windows
    output_file = os.path.join('.', 'output',
                               f'{experiment_name}_experiments_results.csv')
    with open(output_file, 'w+') as f:
        df.to_csv(f, header=True)


def parser_setup():
    """
    Function that sets the different CL flags that can be used to run the project.

    :return: parser (ArgumentParser object)
    """
    parser = argparse.ArgumentParser(
        description='Run one or more calibration experiments')

    parser.add_argument('--data', '-d',
                        help='Select the type or set of data to be used. Codes can be found in' +
                             '\'params.py\' e.g.: SIM1. Defaults to settings in \'current_set_up\'',
                        nargs='+')
    parser.add_argument('--scorer', '-s',
                        help='Select the scorer to be used. Codes can be found in \'params.py\',' +
                             'e.g.: GB. Defaults to settings in \'current_set_up\'',
                        nargs='+')
    parser.add_argument('--calibrator', '-c',
                        help='Select the calibrator to be used. Codes can be found in \'params.py\',' +
                             'e.g.: KDE. Defaults to settings in \'current_set_up\'',
                        nargs='+')
    parser.add_argument('--params', '-p',
                        help='Select the parameter set(s) to be used. Codes can be found in \'params.py\',' +
                             'e.g.: SET1. Defaults to settings in \'current_set_up\'',
                        nargs='+')
    return parser


def parse_object_string(obj_string, name_only=False):
    """
    Function to parse objectstrings in parameter dataframe such as:
    obj_string = "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)"
    If string does not have a FunctionName(flag=value) setup but a FunctionName(value) setup, the value is set to the
    key and the value is set to None (e.g. body: { value: None }
    :param obj_string:
    :param name_only:
    :return: dictionary with keys name and body
        e.g. {name: LogisticRegression, body: {class_weight: None, dual: False}}
    """
    obj_dict = {'name': None, 'body': None}
    if obj_string is not None and obj_string not in ['None', 'nan']:
        obj_string = obj_string.replace('\r', '').replace('\n', '')
        name_match = re.findall(r'[a-zA-Z]+(?=\()', obj_string)
        obj_dict["name"] = "_".join(name_match)
        if not name_only:
            body_match = re.search(r'(?<=\().+(?=\)$)', obj_string).group()
            if len(name_match) > 1:
                body_match = re.search(r'\(.*?\)', body_match).group()[1:-1]
            if body_match is not None:
                body_arr = body_match.split(',')
                obj_dict['body'] = {}
                for par in body_arr:
                    key_val = par.split('=')
                    if len(key_val) == 2:
                        obj_dict['body'][key_val[0].strip()] = key_val[
                            1].strip()
                    else:
                        obj_dict['body'][key_val[0].strip()] = None
    return obj_dict


def process_dataframe(df):
    """
    Processes the output dataframe of a series of experiments to add columns used in data exploration
    :param df: DataFrame
    :return: processed DataFrame
    """
    make_name_columns = {
        # new column name : old column name
        'scorer_name': 'scorers',
        'calibrator_name': 'calibrators',
        'h1_name': 'h1_distribution',
        'h2_name': 'h2_distribution',
        'test_distribution': 'test_set'
    }
    for new_column, old_column in make_name_columns.items():
        try:
            df[new_column] = df.apply(
                lambda row: get_function_names(row[old_column]), axis=1)
        except (KeyError, AttributeError):
            df[new_column] = None

    # Cast to string columns:
    df['fraction_training'] = round(df['fraction_training'], 1).astype(str)
    df['train_calibration_same_data'] = df[
        'train_calibration_same_data'].astype(str)

    make_parameter_columns = [
        # old column name: parameter name (new column is [old column name]_[parameter name])
        ['scorers', 'class_weight'],
        ['calibrators', 'class_weight'],
    ]
    for column, parameter in make_parameter_columns:
        new_column = column + "_" + parameter
        try:
            df[new_column] = df.apply(
                lambda row: get_parameter_value(row[column], parameter),
                axis=1)
        except (KeyError, AttributeError):
            df[new_column] = None

    make_concatenated_columns = {
        # new column name: columns to concatenate
        'lr_system': ['scorer_name', 'calibrator_name'],
        'distr': ['h1_name', 'h2_name'],
        'samedata_scorer': ['scorer_name', 'train_calibration_same_data'],
        'weighted_scorer_label': ['scorer_name', 'scorers_class_weight'],
        'weighted_calibrator_label': ['calibrator_name',
                                      'calibrators_class_weight'],
        'weighted': ['scorers_class_weight', 'calibrators_class_weight']
    }
    for new_column, column_list in make_concatenated_columns.items():
        df = concat_columns(df, column_list, new_column)

    return df


def get_function_names(row):
    """
    Parses function string in row into a dictionary and returns the function name
    e.g. for row Gaussian(dimensions=3, mean=1, sigma=1) returns Gaussian

    :param row: Row in pandas DataFrame; expects single column
    :return: string
    """
    if row != str:
        parsed = parse_object_string(str(row), name_only=True)
    return parsed['name']


def get_parameter_value(row, parameter):
    """
    Returns the parameter value passed to the function string in row
    e.g. for row Gaussian(dimensions=3, mean=1, sigma=1) and parameter dimensions, 3 is returned
    if the parameter does not exist in the function string/row, None is returned

    :param row: Row in pandas DataFrame; expects single column
    :param parameter: parameter to return
    :return: value of parameter
    """
    parsed = parse_object_string(str(row))
    value = None
    if parsed['body'] is not None and parameter in parsed['body']:
        value = parsed['body'][parameter]
    return value


def concat_columns(df, column_names, output_column_name, separator='-'):
    """
    Concatenates the (string) values in columns listed in column_names using a separator and saves output in column
    output_column_name

    :param df: DataFrame
    :param column_names: list of column names to concatenate
    :param output_column_name: name of output column
    :param separator: concatenation seperator, defaults to -
    :return: DataFrame with output column
    """
    assert len(column_names) >= 1
    df[output_column_name] = df[column_names[0]].astype(str)
    for i in range(1, len(column_names)):
        df[output_column_name] += separator + df[column_names[i]].astype(str)
    return df


def resize_and_normalize(img, target_size):
    right_size_img = cv2.resize(img, target_size)

    img_pixels = image.img_to_array(right_size_img)
    img_pixels = np.expand_dims(img_pixels, axis=0)

    # normalize input in [0, 1]
    img_pixels /= 255

    return img_pixels


def fix_tensorflow_rtx():
    """
    A fix to make tensorflow-gpu work with RTX cards (or at least the 2700).
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)


def cache(func):
    """
    A thin wrapper around `lru_cache` so we don't have to specify a `maxsize`.
    """
    # return lru_cache(maxsize=None)(func)
    return func

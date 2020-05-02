import argparse
import os
import re
from csv import writer
from functools import lru_cache
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
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
    parser.add_argument('--scorers', '-s',
                        help='Select the scorer to be used. Codes can be found in \'params.py\',' +
                             'e.g.: GB. Defaults to settings in \'current_set_up\'',
                        nargs='+')
    parser.add_argument('--calibrators', '-c',
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


def create_dataframe(experimental_setup: 'ExperimentalSetup',
                     results: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame({
        'scorers': [e.scorer for e in experimental_setup],
        'calibrators': [e.calibrator for e in experimental_setup],
        **{k: [e.params[k] for e in experimental_setup]
           for k in experimental_setup.params_keys},
        **{k: [e.data_config[k] for e in experimental_setup]
           for k in experimental_setup.data_keys}
    })
    for i, result in enumerate(results):
        for k, v in result.items():
            df.loc[i, k] = v

    df['index'] = df.index
    return df


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
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


def cache(func):
    """
    A thin wrapper around `lru_cache` so we don't have to specify a `maxsize`.
    """
    return lru_cache(maxsize=None)(func)


def save_predicted_lrs(lr_system,
                       test_pairs,
                       lr_predicted,
                       make_plots_and_save_as):
    output_file = f'{make_plots_and_save_as}_lr_results.csv'

    # TODO: dataset toevoegen als dit leesbaar is
    field_names = ['scorers', 'calibrators', 'experiment_id', 'pair_id', 'LR']

    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            csv_writer = writer(f, delimiter=',')
            csv_writer.writerow(field_names)

    experiment_id = os.path.split(make_plots_and_save_as)[-1]
    with open(output_file, 'a+', newline='') as f:
        csv_writer = writer(f, delimiter=',')
        for i in range(len(lr_predicted)):
            first, second = test_pairs[i]
            # check if a test_pair is a proper ENFSI pair:
            # TODO: should this be in our generic pipeline if it's ENFSI specific?
            if first.identity[0:5] == 'ENFSI' \
                    and first.meta['year'] == second.meta['year'] \
                    and first.meta['idx'] == second.meta['idx']:
                csv_writer.writerow([lr_system.scorer,
                                     lr_system.calibrator,
                                     experiment_id,
                                     f"enfsi_{first.meta['year']}_{first.meta['idx']}",
                                     lr_predicted[i],
                                     ])


def get_enfsi_lrs():
    enfsi_data = {
        '2011': {
            'header_row': 1,
            'no_of_pictures': 30,
            'no_of_participants': 17
        },
        '2012': {
            'header_row': 1,
            'no_of_pictures': 30,
            'no_of_participants': 9
        },
        '2013': {
            'header_row': 0,
            'no_of_pictures': 40,
            'no_of_participants': 23
        },
        '2017': {
            'header_row': 1,
            'no_of_pictures': 35,
            'no_of_participants': 25
        },
    }

    columns_df = ['Groundtruth', 'pictures', 'pair_id']
    df_enfsi = pd.DataFrame(columns=columns_df)
    columns_df.extend([n for n in range(1, 40 + 1)])

    for year in ['2011', '2012', '2013', '2017']:
        df_temp = pd.read_excel(os.path.join('resources', 'enfsi',
                                             'Proficiency_test.xlsx'),
                                sheet_name=year,
                                header=enfsi_data[year]['header_row'])

        columns = ['Groundtruth', 'pictures']
        columns.extend([n for n in range(1, enfsi_data[year][
            'no_of_participants'] + 1)])
        df_temp = df_temp[columns]
        df_temp = df_temp.loc[
            df_temp['pictures'].isin(range(1, enfsi_data[year][
                'no_of_pictures'] + 1))]
        df_temp = df_temp.rename(columns=dict([[i, f'{year}-{i}'] for i in
                                               range(100)]))
        df_temp['pair_id'] = df_temp.apply(
            lambda row: f'enfsi_{year}_{row.pictures}', axis=1)

        df_enfsi = df_enfsi.append(df_temp)

    return df_enfsi.replace('-', 0)

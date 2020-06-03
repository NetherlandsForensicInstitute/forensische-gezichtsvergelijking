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
from pandas import DataFrame
from tensorflow.keras.preprocessing import image


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


def write_all_pairs_to_file(all_calibration_pairs, all_test_pairs):
    with open('cal_pairs_all.txt', 'w') as f:
        for pair in all_calibration_pairs:
            f.write(pair[0] + ';' + pair[1] + '\n')
    with open('test_pairs_all.txt', 'w') as f:
        for pair in all_test_pairs:
            f.write(pair[0] + ';' + pair[1] + '\n')


def create_dataframe(experimental_setup, results: List[Dict]) -> pd.DataFrame:
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


def get_facevacs_log_lrs() -> DataFrame:
    """
    reads the facevacs scores from disk and does an ad hoc calibration
    (could cause overfitting). better option would be to get the API working
    """
    # read in the scores
    df = pd.read_excel(os.path.join('resources', 'enfsi',
                                    'results_ENFSI_FaceVacs.xlsx'))
    df.columns = ['year', 'query', 'score', 'remarks']
    del df['remarks']
    # drop those without scores
    df.dropna(inplace=True)

    # currently, we do no calibration, just take probabilities at face value
    df['facevacs'] = np.log10(df['score'] / (1 - df['score']))
    # add pair id
    df['pair_id'] = df.apply(
        lambda row: f'enfsi_{int(row.year)}_{int(row.query)}', axis=1)
    return df[['pair_id', 'facevacs']]


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


def save_predicted_lrs(scorer,
                       calibrator,
                       test_pairs,
                       lr_predicted,
                       make_plots_and_save_as):
    output_file = os.path.join(
        os.path.dirname(make_plots_and_save_as),
        'lr_results.csv')
    experiment_id = os.path.split(make_plots_and_save_as)[-1]

    # TODO: dataset toevoegen als dit leesbaar is
    field_names = ['scorers', 'calibrators', 'experiment_id', 'pair_id',
                   'logLR']

    rows_to_write = []
    for lr, pair in zip(lr_predicted, test_pairs):
        first, second = pair
        # only save for enfsi pairs
        if first.identity[0:5] == 'ENFSI' \
                and first.meta['year'] == second.meta['year'] \
                and first.meta['idx'] == second.meta['idx']:
            pair_id = f"enfsi_{first.meta['year']}_" \
                      f"{first.meta['idx']}"
            rows_to_write.append([scorer,
                                  calibrator,
                                  experiment_id,
                                  pair_id,
                                  np.log10(lr)])

    if rows_to_write:
        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='') as f:
                csv_writer = writer(f, delimiter=',')
                csv_writer.writerow(field_names)

        with open(output_file, 'a+', newline='') as f:
            csv_writer = writer(f, delimiter=',')
            csv_writer.writerows(rows_to_write)


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
        df_temp = df_temp.rename(columns=dict([(i, f'{year}-{i}') for i in
                                               range(100)]))
        df_temp['pair_id'] = df_temp.apply(
            lambda row: f'enfsi_{year}_{row.pictures}', axis=1)

        df_enfsi = df_enfsi.append(df_temp)

    return df_enfsi.replace('-', 0)

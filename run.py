#!/usr/bin/env python3
import os
from datetime import datetime

import confidence
import numpy as np
from lir import CalibratedScorer
from lir.util import to_odds
from tqdm import tqdm

from lr_face.data_providers import get_data, make_pairs, Images
from lr_face.evaluators import evaluate
from lr_face.experiment_settings import ExperimentSettings
from lr_face.utils import write_output, parser_setup, process_dataframe
from params import TIMES


def run(args):
    """
    Run one or more calibration experiments.
    The ExperimentSettings class generates a dataframe containing the different parameter combinations called in the
    command line or in params.py.
    """
    experiments_setup = ExperimentSettings(args)
    parameters_used = experiments_setup.input_parameters  # exclude output columns

    experiment_name = datetime.now().strftime("%Y-%m-%d %H %M %S")

    plots_dir = os.path.join('output', experiment_name)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # caching for data
    dataproviders= {}

    n_experiments = experiments_setup.data_frame.shape[0]
    for row in tqdm(range(0, n_experiments)):
        params_dict = experiments_setup.data_frame[parameters_used].iloc[row].to_dict()
        if (params_dict['dataset_callable'], params_dict['fraction_test']) not in dataproviders:
            dataproviders[(params_dict['dataset_callable'], params_dict['fraction_test'])] = get_data(
                dataset_callable=params_dict['dataset_callable'],
                            fraction_test=params_dict['fraction_test'],
            )
        data_provider = dataproviders[(params_dict['dataset_callable'], params_dict['fraction_test'])]
        if row < n_experiments / TIMES:
            # for the first round, make plots
            make_plots_and_save_as = os.path.join(plots_dir,
                                                  f"{'_'.join([str(v)[:25] for v in params_dict.values()])}")
            results = experiment(params_dict, data_provider=data_provider,
                                 make_plots_and_save_as=make_plots_and_save_as)
        else:
            results = experiment(params_dict, data_provider=data_provider)

        for k, v in results.items():
            experiments_setup.data_frame.loc[row, k] = v

    experiments_setup.data_frame = process_dataframe(experiments_setup.data_frame)
    write_output(experiments_setup.data_frame, experiment_name)


def experiment(params, data_provider: Images =None, make_plots_and_save_as=None):
    """
    Function to run a single experiment with pipeline:
    DataProvider -> fit model on train data -> fit calibrator on calibrator data -> evaluate test set

    """
    lr_system = CalibratedScorer(params['scorers'], params['calibrators'])
    # TODO training will require a different data structure
    # lr_system.scorer.fit(data_provider.X_train, data_provider.y_train)
    X_calib_pairs, y_calib_pairs = make_pairs(data_provider.X_calibrate, data_provider.y_calibrate)
    p = lr_system.scorer.predict_proba(X_calib_pairs)
    lr_system.calibrator.fit(np.log10(to_odds(p[:, 1])), y_calib_pairs)

    return evaluate(lr_system, data_provider, make_plots_and_save_as)


if __name__ == '__main__':
    config = confidence.load_name('lr_face')
    parser = parser_setup()
    arg = parser.parse_args()
    run(arg)

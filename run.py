#!/usr/bin/env python3
import os
from datetime import datetime
from typing import List, Optional, Dict

import confidence
from lir import CalibratedScorer
from tqdm import tqdm

from lr_face.data import split_pairs, FacePair
from lr_face.evaluators import evaluate
from lr_face.experiment_settings import ExperimentSettings
from lr_face.utils import write_output, parser_setup, process_dataframe, \
    fix_tensorflow_rtx
from params import TIMES

fix_tensorflow_rtx()


def run(args):
    """
    Run one or more calibration experiments.
    The ExperimentSettings class generates a dataframe containing the different
    parameter combinations called in the command line or in params.py.
    """
    experiments_setup = ExperimentSettings(args)
    parameters_used = experiments_setup.input_parameters  # exclude output columns
    experiment_name = datetime.now().strftime("%Y-%m-%d %H %M %S")
    plots_dir = os.path.join('output', experiment_name)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    data_splits = dict()
    n_experiments = experiments_setup.data_frame.shape[0]
    for row in tqdm(range(n_experiments)):
        params_dict = \
            experiments_setup.data_frame[parameters_used].iloc[row].to_dict()
        dataset = params_dict['datasets']
        pairs = [pair for pair in dataset.pairs]
        fraction_test = params_dict['fraction_test']

        data_params = (dataset, fraction_test)
        if data_params not in data_splits:
            data_splits[data_params] = split_pairs(pairs, fraction_test)
        calibration_pairs, test_pairs = data_splits[data_params]

        make_plots_and_save_as = None
        # For the first round, make plots
        if row < n_experiments / TIMES:
            make_plots_and_save_as = os.path.join(
                plots_dir,
                f"{'_'.join([str(v)[:25] for v in params_dict.values()])}"
            )

        results = experiment(params_dict,
                             calibration_pairs,
                             test_pairs,
                             make_plots_and_save_as)

        for k, v in results.items():
            experiments_setup.data_frame.loc[row, k] = v

    experiments_setup.data_frame = \
        process_dataframe(experiments_setup.data_frame)
    write_output(experiments_setup.data_frame, experiment_name)


def experiment(
        params,
        calibration_pairs: List[FacePair],
        test_pairs: List[FacePair],
        make_plots_and_save_as: Optional[str] = None
) -> Dict[str, float]:
    """
    Function to run a single experiment with pipeline:
    - Fit model on train data
    - Fit calibrator on calibrator data
    - Evaluate test set

    :param params: Dict
    :param calibration_pairs: List[FacePair]
    :param test_pairs: List[FacePair]
    :param make_plots_and_save_as: str
    :return: Dict[str, float]
    """
    lr_system = CalibratedScorer(params['scorers'], params['calibrators'])
    p = lr_system.scorer.predict_proba(calibration_pairs)
    lr_system.calibrator.fit(
        X=p[:, 1],
        y=[int(pair.same_identity) for pair in calibration_pairs]
    )
    return evaluate(lr_system, test_pairs, make_plots_and_save_as)


if __name__ == '__main__':
    config = confidence.load_name('lr_face')
    parser = parser_setup()
    arg = parser.parse_args()
    run(arg)

#!/usr/bin/env python3
import os
from typing import Dict, Optional

import confidence
from lir import CalibratedScorer
from tqdm import tqdm

from lr_face.evaluators import evaluate
from lr_face.experiments import ExperimentalSetup, Experiment
from lr_face.utils import (write_output,
                           parser_setup,
                           process_dataframe,
                           create_dataframe)
from params import TIMES


def run(scorers, calibrators, data, params):
    experimental_setup = ExperimentalSetup(
        scorer_names=scorers,
        calibrator_names=calibrators,
        data_config_names=data,
        param_names=params,
        num_repeats=TIMES
    )
    output_dir = os.path.join('output', experimental_setup.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = []
    for i, experiment in enumerate(tqdm(experimental_setup)):
        # Make plots for the first round only.
        make_plots_and_save_as = None
        if i < len(experimental_setup) / TIMES:
            make_plots_and_save_as = os.path.join(output_dir, str(experiment))
        results.append(perform_experiment(experiment, make_plots_and_save_as))

    df = create_dataframe(experimental_setup, results)
    df = process_dataframe(df)
    write_output(df, experimental_setup.name)


def perform_experiment(
        experiment: Experiment,
        make_plots_and_save_as: Optional[str]
) -> Dict[str, float]:
    """
    Function to run a single experiment with pipeline:
    - Fit model on train data
    - Fit calibrator on calibrator data
    - Evaluate test set
    """

    calibration_pairs, test_pairs = experiment.get_calibration_and_test_pairs()
    lr_system = CalibratedScorer(experiment.scorer, experiment.calibrator)
    p = lr_system.scorer.predict_proba(calibration_pairs)
    lr_system.calibrator.fit(
        X=p[:, 1],
        y=[int(pair.same_identity) for pair in calibration_pairs]
    )
    return evaluate(lr_system, test_pairs, make_plots_and_save_as)


if __name__ == '__main__':
    config = confidence.load_name('lr_face')
    parser = parser_setup()
    args = parser.parse_args()
    run(**vars(args))

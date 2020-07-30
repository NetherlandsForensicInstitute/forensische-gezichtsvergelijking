#!/usr/bin/env python3
import os
from typing import Dict, Optional

import confidence
import numpy as np
from lir import CalibratedScorer
from tqdm import tqdm

from lr_face.evaluators import evaluate
from lr_face.experiments import ExperimentalSetup, Experiment
from lr_face.utils import (write_output,
                           parser_setup,
                           create_dataframe,
                           write_all_pairs_to_file,
                           get_valid_scores)
from params import TIMES, PAIRS_FROM_FILE


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
    all_calibration_pairs = set()
    all_test_pairs = set()
    for i, experiment in enumerate(tqdm(experimental_setup)):
        # Make plots for the first round only.
        make_plots_and_save_as = None
        if i < len(experimental_setup) / TIMES:
            make_plots_and_save_as = os.path.join(output_dir, str(experiment))
        results.append(perform_experiment(experiment, make_plots_and_save_as, all_calibration_pairs, all_test_pairs,
                                          pairs_from_file=PAIRS_FROM_FILE))

    write_all_pairs_to_file(all_calibration_pairs, all_test_pairs)
    df = create_dataframe(experimental_setup, results)
    write_output(df, experimental_setup.name)


def perform_experiment(
        experiment: Experiment,
        make_plots_and_save_as: Optional[str],
        all_calibration_pairs: set,
        all_test_pairs: set,
        pairs_from_file: bool = False
) -> Dict[str, float]:
    """
    Function to run a single experiment with pipeline:
    - Fit model on train data
    - Fit calibrator on calibrator data
    - Evaluate test set
    """
    if pairs_from_file:
        calibration_pairs_per_category, test_pairs_per_category = \
            experiment.get_calibration_and_test_pairs_from_file()
    else:
        calibration_pairs_per_category, test_pairs_per_category = \
            experiment.get_calibration_and_test_pairs(all_calibration_pairs, all_test_pairs)
    lr_systems = {}
    cal_fraction_valid = {}
    for category, calibration_pairs in calibration_pairs_per_category.items():
        lr_systems[category] = CalibratedScorer(experiment.scorer,
                                                experiment.calibrator)
        # TODO currently, calibration could contain test images
        if experiment.scorer.embedding_model.name == 'Facevacs':
            p = np.array(experiment.get_scores_from_file('results_cal_pairs.txt',
                                                         ((pair.first.path, pair.second.path) for pair in
                                                          calibration_pairs)))
        else:
            p = lr_systems[category].scorer.predict_proba(calibration_pairs)
        assert len(p[0]) == 2
        # Remove invalid scores (-1) where no face was found on one of the images in the pair
        p_valid, calibration_pairs_valid = get_valid_scores(p[:, 1], calibration_pairs)
        y_cal = [int(pair.same_identity) for pair in calibration_pairs_valid]
        if 0 < np.sum(y_cal) < len(calibration_pairs_valid):
            lr_systems[category].calibrator.fit(
                X=p_valid,
                y=y_cal
            )
            cal_fraction_valid[category] = len(calibration_pairs_valid) / len(calibration_pairs)
        else:
            del lr_systems[category]

    return evaluate(experiment=experiment,
                    lr_systems=lr_systems,
                    test_pairs_per_category=test_pairs_per_category,
                    make_plots_and_save_as=make_plots_and_save_as,
                    cal_fraction_valid=cal_fraction_valid)


if __name__ == '__main__':
    config = confidence.load_name('lr_face')
    parser = parser_setup()
    args = parser.parse_args()
    run(**vars(args))

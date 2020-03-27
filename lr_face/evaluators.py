from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from lir import Xy_to_Xn, calculate_cllr, CalibratedScorer, ELUBbounder, plot_score_distribution_and_calibrator_fit
from sklearn.metrics import accuracy_score, roc_auc_score

# from lir.plotting import plot_calibration, plot_lr_distributions
from lr_face.data_providers import make_pairs, ImagePairs


def plot_lr_distributions(predicted_log_lrs, y, savefig=None, show=None):
    """
    plots the 10log lrs generated for the two hypotheses by the fitted system
    """
    plt.figure(figsize=(10, 10), dpi=100)
    points0, points1 = Xy_to_Xn(predicted_log_lrs, y)
    plt.hist(points0, bins=20, alpha=.25, density=True)
    plt.hist(points1, bins=20, alpha=.25, density=True)
    plt.xlabel('10log LR')
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()


def calculate_metrics_dict(scores, y, lr_predicted, label):
    """
    Calculates metrics for an lr system
    given the predicted LRs

    """
    X1, X2 = Xy_to_Xn(lr_predicted, y)

    return {'cllr' + label: round(calculate_cllr(X1, X2).cllr, 4),
            'auc' + label: roc_auc_score(y, scores),
            'accuracy' + label: accuracy_score(y, scores > .5)
            }


def evaluate(lr_system: CalibratedScorer, data_provider: ImagePairs, make_plots_and_save_as=None) -> Dict[str, float]:
    """
    Calculates a variety of evaluation metrics and plots data if make_plots_and_save_as is not None

    """
    scores = lr_system.scorer.predict_proba(data_provider.X_test, data_provider.ids_test)[:, 1]
    LR_predicted = lr_system.calibrator.transform(scores)

    if make_plots_and_save_as:
        calibrator = lr_system.calibrator
        if type(calibrator) == ELUBbounder:
            calibrator = calibrator.first_step_calibrator
        plot_score_distribution_and_calibrator_fit(calibrator, scores, data_provider.y_test, savefig=f'{make_plots_and_save_as} calibration.png')
        plot_lr_distributions(np.log10(LR_predicted),data_provider.y_test,savefig=f'{make_plots_and_save_as} lr distribution.png')


    metric_dict = calculate_metrics_dict(scores, data_provider.y_test, LR_predicted, '')

    return metric_dict

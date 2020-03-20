from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from lir import Xy_to_Xn, calculate_cllr, CalibratedScorer, ELUBbounder
from sklearn.metrics import accuracy_score, roc_auc_score

# from lir.plotting import plot_calibration, plot_lr_distributions
from lr_face.data_providers import make_pairs, Images


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


def plot_calibration(lr_system: CalibratedScorer, scores, y, savefig=None, show=None):
    """
    plots the distributions of scores calculated by the (fitted) lr_system, as well as the fitted score distributions/
    score-to-posterior map
    """
    plt.figure(figsize=(10, 10), dpi=100)
    x = np.arange(0, 1, .01)
    lr_system.calibrator.transform(x)
    points0, points1 = Xy_to_Xn(scores, y)
    plt.hist(points0, bins=20, alpha=.25, density=True, label='class 0')
    plt.hist(points1, bins=20, alpha=.25, density=True, label='class 1')
    if type(lr_system.calibrator) == ELUBbounder:
        plt.plot(x, lr_system.calibrator.first_step_calibrator.p1, label='fit class 1')
        plt.plot(x, lr_system.calibrator.first_step_calibrator.p0, label='fit class 0')
    else:
        plt.plot(x, lr_system.calibrator.p1, label='fit class 1')
        plt.plot(x, lr_system.calibrator.p0, label='fit class 0')
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


def evaluate(lr_system: CalibratedScorer, data_provider: Images, make_plots_and_save_as=None) -> Dict[str, float]:
    """
    Calculates a variety of evaluation metrics and plots data if make_plots_and_save_as is not None

    """
    X_test, y_test = make_pairs(data_provider.X_test, data_provider.y_test)
    LR_predicted = lr_system.predict_lr(X_test)
    scores = lr_system.scorer.predict_proba(X_test)[:, 1]

    if make_plots_and_save_as:
        plot_calibration(lr_system, scores=scores, y=y_test, savefig=f'{make_plots_and_save_as} calibration.png')
        plot_lr_distributions(np.log10(LR_predicted), y_test, savefig=f'{make_plots_and_save_as} lr distribution.png')


    metric_dict = calculate_metrics_dict(scores, y_test, LR_predicted, '')

    return metric_dict

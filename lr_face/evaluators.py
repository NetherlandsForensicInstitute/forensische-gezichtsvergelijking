from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np
from lir import Xy_to_Xn, calculate_cllr, CalibratedScorer, ELUBbounder, \
    plot_score_distribution_and_calibrator_fit
from sklearn.metrics import accuracy_score, roc_auc_score

from lr_face.data import FacePair


def plot_lr_distributions(predicted_log_lrs, y, savefig=None, show=None):
    """
    Plots the 10log LRs generated for the two hypotheses by the fitted system.
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


def plot_performance_as_function_of_resolution(scores,
                                               test_pairs: List[FacePair],
                                               y_test,
                                               show_ratio: bool = False,
                                               savefig: Optional[str] = None,
                                               show: Optional[bool] = None):
    """
    plots the scores as a function of the minimum resolution found on the
    two images of the pair, coloured by ground truth
    """
    plt.figure(figsize=(10, 10), dpi=100)

    if show_ratio:
        resolutions = [np.prod(pair.first.get_image().shape[:2])/
                       np.prod(pair.second.get_image().shape[:2]) for
                       pair in test_pairs]
        label = 'ratio pixels'
    else:
        resolutions = [min(np.prod(pair.first.get_image().shape[:2]),
                            np.prod(pair.second.get_image().shape[:2]))/10**6
                       for pair in test_pairs]
        label = 'Mpixels (smallest image)'
    colors = list(map(lambda x: 'blue' if x else 'red', y_test))
    plt.scatter(resolutions, scores, c=colors)
    plt.xlabel(label)
    plt.ylabel('score')
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()


def plot_tippett(predicted_log_lrs, y, savefig=None, show=None):
    """
    Plots the 10log LRs in a Tippett plot.
    """
    xplot = np.linspace(
        start=np.min(predicted_log_lrs),
        stop=np.max(predicted_log_lrs),
        num=100
    )
    lr_0, lr_1 = Xy_to_Xn(predicted_log_lrs, y)
    perc0 = (sum(i > xplot for i in lr_0) / len(lr_0)) * 100
    perc1 = (sum(i > xplot for i in lr_1) / len(lr_1)) * 100

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(xplot, perc1, color='b', label=r'LRs given $\mathregular{H_1}$')
    plt.plot(xplot, perc0, color='r', label=r'LRs given $\mathregular{H_2}$')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('Log likelihood ratio')
    plt.ylabel('Cumulative proportion')
    plt.title('Tippett plot')
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()


def calculate_metrics_dict(scores, y, lr_predicted, label):
    """
    Calculates metrics for an lr system given the predicted LRs.
    """
    X1, X2 = Xy_to_Xn(lr_predicted, y)

    return {'cllr' + label: round(calculate_cllr(X1, X2).cllr, 4),
            'auc' + label: roc_auc_score(y, scores),
            'accuracy' + label: accuracy_score(y, scores > .5)}


def evaluate(lr_system: CalibratedScorer,
             test_pairs: List[FacePair],
             make_plots_and_save_as: Optional[str] = None) -> Dict[str, float]:
    """
    Calculates a variety of evaluation metrics and plots data if
    `make_plots_and_save_as` is not None.
    """
    scores = lr_system.scorer.predict_proba(test_pairs)[:, 1]
    lr_predicted = lr_system.calibrator.transform(scores)
    y_test = [int(pair.same_identity) for pair in test_pairs]

    if make_plots_and_save_as:
        calibrator = lr_system.calibrator
        if type(calibrator) == ELUBbounder:
            calibrator = calibrator.first_step_calibrator

        plot_performance_as_function_of_resolution(scores, test_pairs, y_test,
                                                   show_ratio=False,
            savefig=f'{make_plots_and_save_as} scores against resolution.png')

        plot_score_distribution_and_calibrator_fit(
            calibrator,
            scores,
            y_test,
            savefig=f'{make_plots_and_save_as} calibration.png'
        )

        plot_lr_distributions(
            np.log10(lr_predicted),
            y_test,
            savefig=f'{make_plots_and_save_as} lr distribution.png'
        )

        plot_tippett(
            np.log10(lr_predicted),
            y_test,
            savefig=f'{make_plots_and_save_as} tippett.png'
        )

    return calculate_metrics_dict(
        scores,
        y_test,
        lr_predicted,
        label=''
    )

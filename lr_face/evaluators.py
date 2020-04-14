import os
from csv import writer
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from lir import Xy_to_Xn, calculate_cllr, CalibratedScorer, ELUBbounder, plot_score_distribution_and_calibrator_fit
from sklearn.metrics import accuracy_score, roc_auc_score

from lr_face.data_providers import ImagePairs


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


def plot_tippett(predicted_log_lrs, y, savefig=None, show=None):
    """
    Plots the 10log LRs in a Tippett plot.
    """
    xplot = np.linspace(np.min(predicted_log_lrs), np.max(predicted_log_lrs), 100)
    lr_0, lr_1 = Xy_to_Xn(predicted_log_lrs, y)
    perc0 = (sum(i > xplot for i in lr_0) / len(lr_0)) * 100
    perc1 = (sum(i > xplot for i in lr_1) / len(lr_1)) * 100

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(xplot, perc1, color='b', label='LRs given $\mathregular{H_1}$')
    plt.plot(xplot, perc0, color='r', label='LRs given $\mathregular{H_2}$')
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
            'accuracy' + label: accuracy_score(y, scores > .5)
            }


def save_lr_results(params_dict, data_provider, LR_predicted, experiment_name):

    output_file = os.path.join('.', 'output',
                               f'{experiment_name}_lr_results.csv')

    # TODO: dataset toevoegen als dit leesbaar is
    field_names = ['scorers', 'calibrators', 'pair_id', 'LR']

    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            csv_writer = writer(f, delimiter=',')
            csv_writer.writerow(field_names)

    with open(output_file, 'a+', newline='') as f:
        csv_writer = writer(f, delimiter=',')
        for i in range(len(LR_predicted)):
            csv_writer.writerow([params_dict['scorers'],
                                 params_dict['calibrators'],
                                 data_provider.ids_test[i],
                                 LR_predicted[i],
                                 ])
    # TODO: evt alleen van enfsi-data de gegevens opslaan


def evaluate(lr_system: CalibratedScorer, data_provider: ImagePairs,
             params_dict: dict, make_plots_and_save_as=None,
             experiment_name=None) -> Dict[str, float]:
    """
    Calculates a variety of evaluation metrics, saves the LR results  and
    plots data if make_plots_and_save_as is not None.
    """
    scores = lr_system.scorer.predict_proba(data_provider.X_test, data_provider.ids_test)[:, 1]
    LR_predicted = lr_system.calibrator.transform(scores)

    if make_plots_and_save_as:
        calibrator = lr_system.calibrator
        if type(calibrator) == ELUBbounder:
            calibrator = calibrator.first_step_calibrator
        plot_score_distribution_and_calibrator_fit(calibrator, scores, data_provider.y_test,
                                                   savefig=f'{make_plots_and_save_as} calibration.png')
        plot_lr_distributions(np.log10(LR_predicted), data_provider.y_test,
                              savefig=f'{make_plots_and_save_as} lr distribution.png')
        plot_tippett(np.log10(LR_predicted), data_provider.y_test, savefig=f'{make_plots_and_save_as} tippett.png')

        save_lr_results(params_dict=params_dict,
                        data_provider=data_provider,
                        LR_predicted=LR_predicted,
                        experiment_name=experiment_name)

    metric_dict = calculate_metrics_dict(scores, data_provider.y_test, LR_predicted, '')

    return metric_dict

from typing import Dict, Optional, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lir import Xy_to_Xn, calculate_cllr, CalibratedScorer, ELUBbounder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from lr_face.data import FacePair
from lr_face.experiments import Experiment
from lr_face.utils import save_predicted_lrs, get_valid_scores


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


def plot_ROC_curve(scores, y, savefig: Optional[str] = None,
                   show: Optional[bool] = None):
    fpr, tpr, thresholds = roc_curve(y, scores)
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fpr, fpr, linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, color='r', label=r'ROC curve')
    plt.xlabel('False positive rate (1 - specificity)')
    plt.ylabel('True positive rate (sensitivity)')
    plt.title('ROC curve')
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()


def plot_ROC_in_LR_coordinates(scores, y, savefig: Optional[str] = None,
                               show: Optional[bool] = None):

    fpr, tpr, thresholds = roc_curve(y, scores)
    LLRp = np.log10(tpr/fpr)
    LLRm = np.log10((1-tpr)/(1-fpr))
    plt.figure(figsize=(10, 10), dpi=100)
    plt.gca().invert_xaxis()
    plt.plot(-fpr, fpr, linestyle='--', label='No Skill')
    plt.plot(LLRm, LLRp, color='r', label=r'ROC curve')
    plt.xlabel('base-tan logarithm of the negative likelihood ratio($log_{10}LR$)')
    plt.ylabel('base-tan logarithm of the positive likelihood ratio($log_{10}LR$)')
    plt.title('ROC curve')
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()


def plot_performance_as_function_of_yaw(scores,
                                        test_pairs: List[FacePair],
                                        y_test,
                                        savefig: Optional[str] = None,
                                        show: Optional[bool] = None):
    """
    plots the scores as a function of the maximum yaw (=looking sideways) on
    the images, coloured by ground truth. calls plt.show() if show is True.
    """
    df_yaws = pd.DataFrame(columns=['pair_id', 'y_test', 'yaw_first', 'yaw_second', 'score'])
    for i, test_pair in enumerate(test_pairs):
        df_yaws = df_yaws.append(dict(pair_id=i,
                                      y_test=y_test[i],
                                      yaw_first=test_pair.first.yaw.value,
                                      yaw_second=test_pair.second.yaw.value,
                                      score=scores[i]),
                                 ignore_index=True)

    sns.catplot(x="yaw_second_image", y="score", row='yaw_first_image', hue='y_test', kind="swarm", data=df_yaws)
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

    if show_ratio:
        resolutions = [np.prod(pair.first.get_image().shape[:2]) /
                       np.prod(pair.second.get_image().shape[:2]) for
                       pair in test_pairs]
        label = 'ratio pixels'
    else:
        resolutions = [min(np.prod(pair.first.get_image().shape[:2]),
                           np.prod(
                               pair.second.get_image().shape[:2])) / 10 ** 6
                       for pair in test_pairs]
        label = 'Mpixels (smallest image)'

    plot_performance_as_a_function_of_x(
        properties=resolutions,
        scores=scores,
        y_test=y_test,
        x_label=label,
        savefig=savefig,
        show=show)


def plot_performance_as_a_function_of_x(
        properties: List[float],
        scores: List[float],
        y_test: List[Union[int, bool]],
        x_label: str, savefig: Optional[str], show: bool,
        bins: Optional[List[Tuple[float, float]]] = None):
    """
    plots the scores as a function of some vector of properties, coloured by
    ground truth. Includes mean in each of the bins, if provided
    """
    plt.figure(figsize=(10, 10), dpi=100)
    colors = list(map(lambda x: 'blue' if x else 'red', y_test))
    plt.scatter(properties, scores, c=colors)
    plt.xlabel(x_label)
    plt.ylabel('score')
    if bins:
        for bin in bins:
            avg = np.mean([score for score, prop, y in
                           zip(scores, properties, y_test)
                           if bin[0] < prop < bin[1] and y])
            plt.plot(bin, [avg, avg], c='blue')
            avg = np.mean([score for score, prop, y in
                           zip(scores, properties, y_test)
                           if bin[0] < prop < bin[1] and not y])
            plt.plot(bin, [avg, avg], c='red')
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


def plot_cllr(pairs, lrs, y, savefig=None, show=None, scorer=None):
    """
    Plots cllr value for ENFSI tests. It computes both cllr of automated systems with the cllrs from experts.
    If there is no ENFSI data, this graph does not show.
    """
    # Create a mask to only evaluate ENFSI data.
    maskEnfsi = [x.first.source[:5] == 'Enfsi' for x in pairs]
    if maskEnfsi.count(True) == 0:
        print(f'No ENFSI data in {savefig}. No Cllrs are plotted')
        return

    pairsEnfsi = [x for x, y in zip(pairs, maskEnfsi) if y]
    lrsEnfsi = lrs[maskEnfsi]
    yEnfsi = np.array(y)[maskEnfsi]

    years = np.unique([x.first.meta['year'] for x in pairsEnfsi])

    cllrAut = []  # for automated system
    cllrExpe = []  # for Experts

    for year in years:
        maskYear = [x.first.meta['year'] == year for x in pairsEnfsi]

        lrs0, lrs1 = Xy_to_Xn(lrsEnfsi[maskYear], yEnfsi[maskYear])
        cllrAut.append(calculate_cllr(lrs0, lrs1).cllr)

        LLRexp = np.array([pair.expertsLLR for pair, mask in zip(pairsEnfsi, maskYear) if mask], dtype="float32")
        LRexp = np.power(10, LLRexp)

        LRexp0 = LRexp[yEnfsi[maskYear] == 0, :]
        LRexp1 = LRexp[yEnfsi[maskYear] == 1, :]

        # Calculate cllr individually for each expert
        cllrExpe.append([calculate_cllr(LRexp0[:, i], LRexp1[:, i]).cllr for i in range(LRexp0.shape[1])])

    plt.figure(figsize=(10, 10), dpi=100)
    # Plot for automated system
    plt.scatter(range(len(years)), cllrAut, color='b', label=f'Cllrs {scorer}')

    # Plot for Experts
    xp = []
    yp = []
    for x, y in enumerate(cllrExpe):
        xp += [x] * len(y)
        yp += y

    plt.scatter(xp, yp, marker='x', color='r', label=r'Experts')

    plt.xlabel('Year')
    plt.xticks(range(len(years)), years)
    plt.ylabel('Cllr')
    plt.title('Cllr for ENFSI Dataset')
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()


def calculate_metrics_dict(number_of_scores, scores, y, lr_predicted, cal_fraction_valid, label):
    """
    Calculates metrics for an lr system given the predicted LRs.
    """
    X1, X2 = Xy_to_Xn(lr_predicted, y)
    results = {'cllr' + label: round(calculate_cllr(X1, X2).cllr, 4),
               'auc' + label: roc_auc_score(y, scores),
               'accuracy' + label: accuracy_score(y, scores > .5),
               'cal_fraction_valid' + label: np.mean(list(cal_fraction_valid.values())),
               'test_fraction_valid' + label: len(scores) / number_of_scores}
    for key, value in cal_fraction_valid.items():
        results[f'cal_fraction_{key}'] = value
    return results


# TODO put here temporarily until a new version of lir comes out
def plot_score_distribution_and_calibrator_fit(calibrator, scores, y, savefig=None, show=None):
    """
    plots the distributions of scores calculated by the (fitted) lr_system, as well as the fitted score distributions/
    score-to-posterior map
    (Note - for ELUBbounder calibrator is the firststepcalibrator)
    """
    plt.figure(figsize=(10, 10), dpi=100)
    x = np.arange(0, 1, .01)
    calibrator.transform(x)
    if len(set(y)) == 2:
        points0, points1 = Xy_to_Xn(scores, y)
        plt.hist(points0, bins=20, alpha=.25, density=True, label='class 0')
        plt.hist(points1, bins=20, alpha=.25, density=True, label='class 1')
        plt.plot(x, calibrator.p1, label='fit class 1')
        plt.plot(x, calibrator.p0, label='fit class 0')
    else:
        plt.hist(scores, bins=20, alpha=.25, density=True, label='class x')
        plt.plot(x, calibrator.p1, label='fit class 1')
        plt.plot(x, calibrator.p0, label='fit class 0')
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    if show or savefig is None:
        plt.show()


def evaluate(experiment: Experiment,
             lr_systems: Dict[Tuple, CalibratedScorer],
             test_pairs_per_category: Dict[Tuple, List[FacePair]],
             make_plots_and_save_as: Optional[str],
             cal_fraction_valid: Dict[Tuple, float]) -> Dict[str, float]:
    """
    Calculates a variety of evaluation metrics and plots data if
    `make_plots_and_save_as` is not None.
    """

    number_of_scores = 0
    scores = np.array([])
    lr_predicted = np.array([])
    y_test = []
    test_pairs = []
    for category, pairs in test_pairs_per_category.items():
        if category not in lr_systems:
            print(f'skipping {pairs} for category {category}')
            continue
        if lr_systems[category].scorer.embedding_model.name == 'Facevacs':
            category_scores = np.array(experiment.get_scores_from_file('results_test_pairs.txt',
                                                                       ((pair.first.path, pair.second.path) for pair in
                                                                        pairs)))
        else:
            category_scores = lr_systems[category].scorer.predict_proba(pairs)
        category_scores_valid, pairs_valid = get_valid_scores(category_scores[:, 1], pairs)
        scores = np.append(scores, category_scores_valid)
        number_of_scores += len(category_scores)
        lr_predicted = np.append(
            lr_predicted,
            lr_systems[category].calibrator.transform(category_scores_valid))
        category_y_test = [int(pair.same_identity) for pair in pairs_valid]
        y_test += category_y_test
        test_pairs += list(pairs_valid)
        if make_plots_and_save_as:
            calibrator = lr_systems[category].calibrator
            if type(calibrator) == ELUBbounder:
                calibrator = calibrator.first_step_calibrator
            plot_score_distribution_and_calibrator_fit(
                calibrator,
                category_scores_valid,
                category_y_test,
                savefig=f'{make_plots_and_save_as} {[str(c).split(":")[0] for cat in category for c in cat]} '
                        f'calibration' + '.png'
            )
            # save last one (type should all be the same)
            scorer = lr_systems[category].scorer

    lr_predicted = np.nan_to_num(lr_predicted, posinf=10e5)
    if make_plots_and_save_as:
        plot_performance_as_function_of_yaw(
            scores,
            test_pairs,
            y_test,
            savefig=f'{make_plots_and_save_as} scores against yaw.png')

        plot_performance_as_function_of_resolution(
            scores,
            test_pairs,
            y_test,
            show_ratio=False,
            savefig=f'{make_plots_and_save_as} scores against resolution.png')

        plot_lr_distributions(
            np.log10(lr_predicted),
            y_test,
            savefig=f'{make_plots_and_save_as} lr distribution.png'
        )

        plot_ROC_curve(scores,
                       y_test,
                       savefig=f'{make_plots_and_save_as} ROC curve.png')
        plot_ROC_in_LR_coordinates(scores,
                                   y_test,
                                   savefig=f'{make_plots_and_save_as} ROC LR coordinates curve.png')

        plot_tippett(
            np.log10(lr_predicted),
            y_test,
            savefig=f'{make_plots_and_save_as} tippett.png'
        )

        plot_cllr(
            test_pairs, lr_predicted, y_test,
            savefig=f'{make_plots_and_save_as} cllr.png',
            scorer=scorer.embedding_model.name)

        save_predicted_lrs(
            scorer, calibrator, test_pairs, lr_predicted,
            make_plots_and_save_as)

    return calculate_metrics_dict(
        number_of_scores=number_of_scores,
        scores=scores,
        y=y_test,
        lr_predicted=lr_predicted,
        cal_fraction_valid=cal_fraction_valid,
        label=''
    )

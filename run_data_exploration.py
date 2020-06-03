"""
Script to be run with streamlit. Adjust the research_question to get different apps
To view, in terminal:
streamlit run run_data_exploration.py
"""

import os
import re
from collections import defaultdict
from copy import deepcopy

import altair as alt
import pandas as pd
import streamlit as st
# Research_question, one of:
# -train_calibrate_same_data
from lir import Xy_to_Xn, calculate_cllr

from lr_face.utils import get_facevacs_log_lrs, get_enfsi_lrs

research_question = 'train_calibrate_same_data'


# to use maximum screen width
def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


@st.cache
def get_csv(csv):
    return pd.read_csv(os.path.join('output', csv))


@st.cache
def get_enfsi_lrs_cacheable():
    return get_enfsi_lrs()


st.title('Data exploration for face comparison models')

_max_width_()
latest_exp_csv = sorted([f for f in (os.listdir('output')) if f.endswith(
    'experiments_results.csv')])[-1]
df_exp = deepcopy(get_csv(latest_exp_csv))
set_calibrators = list(set(df_exp['calibrators']))
set_scorers = list(set(df_exp['scorers']))
set_calibration_data = list(set(df_exp['calibration']))
set_test_data = list(set(df_exp['test']))

st.header('General information')
st.markdown(f'latest csv: {latest_exp_csv}')
st.markdown(f'no of experiments: {len(df_exp)}')
st.markdown(f'calibration data: {set_calibration_data}')
st.markdown(f'test data: {set_test_data}')
st.markdown(f'scorers: {set_scorers}')
st.markdown(f'calibrators: {set_calibrators}')

# show dataframe, option to select columns
defaultcols = ['index', 'scorers', 'calibrators', 'calibration', 'test',
               'cllr', 'auc', 'accuracy']
cols = st.multiselect("Select columns", df_exp.columns.tolist(),
                      default=defaultcols)
st.dataframe(df_exp[cols])

st.header('Select scorer, calibrator and dataset:')
# show dataframe, select calibrator, score and data
calibrators = st.multiselect("Calibrator", set_calibrators,
                             default=set_calibrators)
scorers = st.multiselect("Scorer", set_scorers, default=set_scorers)
cal_data = st.multiselect("Calibration data",
                          set_calibration_data, default=set_calibration_data)
test_data = st.multiselect("Test data", set_test_data, default=set_test_data)

st.dataframe(df_exp.loc[df_exp['calibrators'].isin(calibrators) &
                        df_exp['scorers'].isin(scorers) &
                        df_exp['test'].isin(test_data) &
                        df_exp['calibration'].isin(cal_data)][cols])

if research_question == 'train_calibrate_same_data':

    st.header('Metrics for each combination of calibration dataset, scorer '
              'and '
              'calibrator:')
    for metric in ('cllr', 'auc', 'accuracy'):
        st.altair_chart(alt.Chart(df_exp, width=40).mark_boxplot().encode(
            x='calibration',
            y=alt.Y(metric,
                    scale=alt.Scale(domain=[0, 1.2])
                    ),
            row=alt.Row('calibrators', header=alt.Header(labelAngle=-90)),
            column=alt.Column('scorers')
        ).interactive())

st.header('Calibration and distribution plots')
# get all images
output_plots = latest_exp_csv[0:19]
list_plots = sorted([f for f in (os.listdir(f'./output/{output_plots}/')) if
                     f.endswith('.png')])

plot_types = []
for i, plot in enumerate(list_plots):
    x = re.search(r"_", plot)
    plot_nr = plot[0:x.start()]
    if i == 0:
        first_plot_nr = plot_nr
    if plot_nr == first_plot_nr:
        y = re.search(r"\s(.*)", plot)
        plot_type = plot[y.start() + 1:-4]
        plot_types += [plot_type]
    else:
        break

n_plot_types = len(plot_types)
n_plot_nrs = len(list_plots)

for i in range(n_plot_nrs):
    start = n_plot_types * i
    stop = n_plot_types * (i + 1)
    # TODO: change width variable, dependent on screen resolution
    # TODO: improve caption
    st.image([f'./output/{output_plots}/{plt}' for plt in list_plots[
                                                          start:stop]],
             width=400, caption=[plt[:-4] for plt in list_plots[start:stop]])

st.header('LR results')
# get LR results, same as exp.results
lr_csv = os.path.join(output_plots, 'lr_results.csv')

def get_cllr_df(df_lrs):
    cllrs = []
    all_lrs_per_year = defaultdict(list)
    for rater in df_lrs.columns:
        if rater not in ['Groundtruth', 'pictures', 'pair_id', 'res_pair_id']:
            df_lr_y = df_lrs[False == pd.isna(df_lrs[rater])][[rater,
                                                              'Groundtruth']]
            if len(df_lr_y) > 0:
                X1, X2 = Xy_to_Xn(10 ** df_lr_y[rater],
                                  df_lr_y['Groundtruth'])
                if rater[:4] in ['2011', '2012', '2013', '2017']:
                    group = rater[:4]
                    all_lrs_per_year[group] += zip(X1, X2)
                else:
                    group = rater
                cllr_results = calculate_cllr(list(
                    X1), list(X2))
                cllrs.append([rater, group, round(cllr_results.cllr, 4),
                              round(cllr_results.cllr_min, 4)])
    for group, values in all_lrs_per_year.items():
        lrs1, lrs2 = zip(*values)
        cllr_results = calculate_cllr(list(lrs1), list(lrs2))
        cllrs.append([group, group + '-all', round(cllr_results.cllr, 4),
                      round(cllr_results.cllr_min, 4)])
    return pd.DataFrame(cllrs, columns=['rater', 'group', 'cllr', 'cllr_min'])


if len(lr_csv) == 0:
    st.markdown('No LR results available.')
else:
    df_models = deepcopy(get_csv(lr_csv))
    df_models['model'] = df_models.apply(
        lambda row: f'{row.scorers}_{row.calibrators}_{row.experiment_id}',
        axis=1)
    df_enfsi = deepcopy(get_enfsi_lrs())
    df_facevacs = get_facevacs_log_lrs()

    model_lrs_per_pair_df = df_models.pivot(index='pair_id', columns='model',
                                            values='logLR').reset_index()

    df_lrs = df_enfsi.merge(model_lrs_per_pair_df, how='outer',
                            left_on='pair_id',
                            right_on='pair_id')
    df_lrs = df_lrs.merge(df_facevacs, how='outer', left_on='pair_id',
                          right_on='pair_id')
    # hacky way to make sure sorting alphabetically order on ground truth
    df_lrs['res_pair_id'] = df_lrs.apply(
        lambda row: f'{row.Groundtruth}_{row.pair_id}', axis=1)
    df_cllr = get_cllr_df(df_lrs)

    st.markdown(f'These are Cllr results on {len(df_lrs)} pairs, but not all '
                'systems provide a score always, '
                'so only partially comparable')
    st.markdown('cllr_min (red) gives cllr after perfect (on test data) '
                'calibration, and thus the best cllr achievable for that '
                'scorer')
    st.altair_chart(alt.Chart(df_cllr, height=400).mark_circle(
        size=20).encode(x='group', y='cllr') +
                    alt.Chart(df_cllr).mark_circle(
                        size=10, color='red').encode(x='group', y='cllr_min'
                                                     ).interactive())
    df_lrs_inner = df_enfsi.merge(model_lrs_per_pair_df, how='inner',
                       left_on='pair_id',
                       right_on='pair_id').merge(df_facevacs,
                                                 how='inner',
                                                 left_on='pair_id',
                                                 right_on='pair_id')
    df_cllr = get_cllr_df(df_lrs_inner)
    st.markdown(f'The same plot, but based on {len(df_lrs_inner)} pairs scored '
                f'by all (so more comparable - including unrateds by experts)')
    st.altair_chart(alt.Chart(df_cllr, height=400).mark_circle(size=20
                    ).encode(x='group', y='cllr') +
                    alt.Chart(df_cllr).mark_circle(size=10, color='red'
                    ).encode(x='group',y='cllr_min').interactive())

    df_lrs_long = pd.melt(df_lrs, id_vars='res_pair_id', value_vars=list(
        df_lrs)[3:-1], var_name='lr_assigner', value_name='logLR')

    df_lrs_long.loc[df_lrs_long['lr_assigner'].str.len() < 8, 'lr_assigner'] = 'expert'

    set_calibrators = list(set(df_models['calibrators']))
    set_scorers = list(set(df_models['scorers']))

    st.subheader('General information')
    st.markdown(f'latest LR csv: {lr_csv}')
    st.markdown(f'scorers: {set_scorers}')
    st.markdown(f'calibrators: {set_calibrators}')

    st.altair_chart(alt.Chart(df_lrs_long, width=60).mark_circle(
        size=20).encode(
        x=alt.X(
            'jitter:Q',
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y('logLR:Q'),
        color=alt.Color('lr_assigner:N'),
        column=alt.Column(
            'res_pair_id:N',
            header=alt.Header(
                labelAngle=-90,
                titleOrient='top',
                labelOrient='bottom',
                labelAlign='right',
                labelPadding=3,
            ),
        ),
    ).transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    ))

btn = st.button("Click me!")
if btn:
    st.balloons()

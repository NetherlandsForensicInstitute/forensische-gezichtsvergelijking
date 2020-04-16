"""
Script to be run with streamlit. Adjust the research_question to get different apps
To view, in terminal:
streamlit run run_data_exploration.py
"""

import os
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# Research_question, one of:
# -train_calibrate_same_data
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
def get_enfsi_lrs():
    enfsi_data = {
        '2011': {
            'header_row': 1,
            'no_of_pictures': 30,
            'no_of_participants': 17
        },
        '2012': {
            'header_row': 1,
            'no_of_pictures': 30,
            'no_of_participants': 9
        },
        '2013': {
            'header_row': 0,
            'no_of_pictures': 40,
            'no_of_participants': 23
        },
        '2017': {
            'header_row': 1,
            'no_of_pictures': 35,
            'no_of_participants': 25
        },
    }

    columns_df = ['Groundtruth', 'pictures', 'pair_id']
    columns_df.extend([n for n in range(1, 40 + 1)])

    df_enfsi = pd.DataFrame(columns=columns_df)
    for year in ['2011', '2012', '2013', '2017']:
        df_temp = pd.read_excel(os.path.join('resources', 'enfsi',
                                             'Proficiency_test.xlsx'),
                                sheet_name=year,
                                header=enfsi_data[year]['header_row'])

        columns = ['Groundtruth', 'pictures']
        columns.extend([n for n in range(1, enfsi_data[year][
            'no_of_participants'] + 1)])
        df_temp = df_temp[columns]
        df_temp = df_temp.loc[
            df_temp['pictures'].isin(range(1, enfsi_data[year][
                'no_of_pictures'] + 1))]
        df_temp['pair_id'] = df_temp.apply(
            lambda row: f'enfsi_{year}_{row.pictures}', axis=1)

        df_enfsi = df_enfsi.append(df_temp)

    return df_enfsi


st.title('Data exploration for face comparison models')

_max_width_()
latest_exp_csv = sorted([f for f in (os.listdir('output')) if f.endswith(
    'experiments_results.csv')])[-1]
df_exp = get_csv(latest_exp_csv)
set_calibrators = list(set(df_exp['calibrators']))
set_scorers = list(set(df_exp['scorers']))
set_data = list(set(df_exp['datasets']))


st.header('General information')
st.markdown(f'latest csv: {latest_exp_csv}')
st.markdown(f'no of experiments: {len(df_exp)}')
st.markdown(f'data: {set_data}')
st.markdown(f'scorers: {set_scorers}')
st.markdown(f'calibrators: {set_calibrators}')


# show dataframe, option to select columns
defaultcols = ['index', 'scorers', 'calibrators', 'datasets', 'cllr',
               'auc', 'accuracy']
cols = st.multiselect("Select columns", df_exp.columns.tolist(), default=defaultcols)
st.dataframe(df_exp[cols])


st.header('Select scorer, calibrator and dataset:')
# show dataframe, select calibrator, score and data
calibrators = st.multiselect("Calibrator", set_calibrators, default=set_calibrators)
scorers = st.multiselect("Scorer", set_scorers, default=set_scorers)
data = st.multiselect("Data", set_data, default=set_data)

st.dataframe(df_exp.loc[df_exp['calibrators'].isin(calibrators) &
                        df_exp['scorers'].isin(scorers) &
                        df_exp['datasets'].isin(data)][cols])


if research_question == 'train_calibrate_same_data':

    st.header('Metrics for each combination of dataset, scorer and '
              'calibrator:')
    for metric in ('cllr', 'auc', 'accuracy'):
        st.altair_chart(alt.Chart(df_exp).mark_boxplot().encode(
            x='datasets',
            y=alt.Y(metric,
                    scale=alt.Scale(domain=[0, 1.2])
                    ),
            row=alt.Row('calibrator_name', header=alt.Header(labelAngle=-90)),
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
        plot_type = plot[y.start()+1:-4]
        plot_types += [plot_type]
    else:
        break

n_plot_types = len(plot_types)
n_plot_nrs = len(list_plots)

for i in range(n_plot_nrs):
    start = n_plot_types*i
    stop = n_plot_types*(i+1)
    # TODO: change width variable, dependent on screen resolution
    # TODO: improve caption
    st.image([f'./output/{output_plots}/{plt}' for plt in list_plots[
                                                           start:stop]],
             width=400, caption=[plt[:-4] for plt in list_plots[start:stop]])


st.header('LR results')
# get LR results, same as exp.results
latest_lr_csv = sorted([f for f in (os.listdir('output')) if f.endswith(
    'lr_results.csv')])[-1]

if len(latest_lr_csv) == 0 or latest_exp_csv[:19] != latest_lr_csv[:19]:
    st.markdown('No LR results available.')
else:
    df_models = get_csv(latest_lr_csv)
    df_models['pair_id'] = df_models.apply(lambda row: f'{row.pair_id[:-2]}',
                                           axis=1)
    df_models['model'] = df_models.apply(lambda row: f'{row.scorers}_{row.calibrators}',
                                           axis=1)
    set_models = list(set(df_models['model']))

    df_lrs = get_enfsi_lrs()

    for model in set_models:
        df_lrs[f'LR_{model}'] = np.nan

    for row in df_models.iterrows():
        df_lrs.loc[(df_lrs['pair_id'] == row['pair_id']), [row['model']]] = \
            row['LR']

    set_calibrators = list(set(df_models['calibrators']))
    set_scorers = list(set(df_models['scorers']))

    st.subheader('General information')
    st.markdown(f'latest LR csv: {latest_lr_csv}')
    st.markdown(f'scorers: {set_scorers}')
    st.markdown(f'calibrators: {set_calibrators}')


btn = st.button("Click me!")
if btn:
    st.balloons()

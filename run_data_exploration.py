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


st.title('Data exploration for face comparison models')

_max_width_()
latest_csv = sorted([f for f in (os.listdir('output')) if f.endswith('.csv')])[-1]
df = get_csv(latest_csv)
set_calibrators = list(set(df['calibrators']))
set_scorers = list(set(df['scorers']))
set_data = list(set(df['dataset_callable']))


st.header('General information')
st.markdown(f'latest csv: {latest_csv}')
st.markdown(f'no of experiments: {len(df)}')
st.markdown(f'data: {set_data}')
st.markdown(f'scorers: {set_scorers}')
st.markdown(f'calibrators: {set_calibrators}')


# show dataframe, option to select columns
defaultcols = ['index', 'scorers', 'calibrators', 'dataset_callable', 'cllr', 'auc', 'accuracy']
cols = st.multiselect("Select columns", df.columns.tolist(), default=defaultcols)
st.dataframe(df[cols])


st.header('Select scorer, calibrator and dataset:')
# show dataframe, select calibrator, score and data
calibrators = st.multiselect("Calibrator", set_calibrators, default=set_calibrators)
scorers = st.multiselect("Scorer", set_scorers, default=set_scorers)
data = st.multiselect("Data", set_data, default=set_data)

st.dataframe(df.loc[df['calibrators'].isin(calibrators) &
                    df['scorers'].isin(scorers) &
                    df['dataset_callable'].isin(data)][cols])


if research_question == 'train_calibrate_same_data':

    st.header('Boxplot of metrics for each combination of dataset, scorer and calibrator:')
    for metric in ('cllr', 'auc', 'accuracy'):
        st.altair_chart(alt.Chart(df).mark_boxplot().encode(
            x='dataset_callable',
            y=alt.Y(metric,
                    scale=alt.Scale(domain=[0, 1.2])
                    ),
            row=alt.Row('calibrator_name', header=alt.Header(labelAngle=-90)),
            column=alt.Column('scorers')
        ).interactive())

    # st.altair_chart(alt.Chart(df).mark_point().encode(
    #     x=alt.X(alt.repeat("column"), type='quantitative', scale=alt.Scale(zero=False), ),
    #     y=alt.Y(alt.repeat("row"), type='quantitative'),
    #     color='train_calibration_same_data',
    #     tooltip=list(df.columns),
    # ).repeat(
    #     row=['cllr'],
    #     column=['accuracy', 'auc']
    # ).interactive())


st.header('Calibration and distribution plots')
# get all images
output_plots = latest_csv[0:19]
list_plots = sorted([f for f in (os.listdir(f'./output/{output_plots}/')) if f.endswith('.png')])

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
    start = 3*i
    stop = 3*(i+1)
    # TODO: change width variable, width could be dependent on screen resoluation
    st.image([f'./output/{output_plots}/{plt}' for plt in list_plots[start:stop]], width=400)


btn = st.button("Click me!")
if btn:
    st.balloons()

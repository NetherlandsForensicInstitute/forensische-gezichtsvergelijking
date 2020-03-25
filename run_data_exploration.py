"""
script to be run with streamlit. Adjust the research_question to get different apps

to view, in terminal:
streamlit run run_data_exploration.py
"""

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st




# one of:
# -train_calibrate_same_data
research_question = 'train_calibrate_same_data'

# take latest csv
latest_csv = sorted([f for f in (os.listdir('output')) if f.endswith('.csv')])[-1]
df = pd.read_csv(os.path.join('output', latest_csv))

'no of experiments:', len(df), latest_csv
df
latest_csv

if research_question == 'train_calibrate_same_data':

    for metric in ('cllr', 'auc', 'accuracy'):

        st.altair_chart(alt.Chart(df).mark_boxplot().encode(
            x='dataset_callable',
            y=alt.Y(metric,
                    scale=alt.Scale(domain=[0, 1.2])
                    ),
            row='calibrator_name'
        ).interactive())


    st.altair_chart(alt.Chart(df).mark_point().encode(
        x=alt.X(alt.repeat("column"), type='quantitative', scale=alt.Scale(zero=False), ),
        y=alt.Y(alt.repeat("row"), type='quantitative'),
        color='train_calibration_same_data',
        tooltip=list(df.columns),
    ).repeat(
        row=['cllr'],
        column=['accuracy', 'auc']
    ).interactive())


# -*- coding: utf-8 -*-
"""

@author: Stefan Eng
Simple visualization of the "HandStart" event for features "O1", "O2", "C3", "C4"

"""

import pandas as pd
import matplotlib.pyplot as plt

sub1_events_file = '../input/train/subj1_series1_events.csv'
sub1_data_file = '../input/train/subj1_series1_data.csv'

sub1_events = pd.read_csv(sub1_events_file)
sub1_data = pd.read_csv(sub1_data_file)

sub1 = pd.concat([sub1_events, sub1_data], axis = 1)
sub1["time"] = range(0, len(sub1))

sample_sub1 = sub1[sub1["time"] < 5000]

event = "HandStart"
EventColors = ["lightgrey", "green"]

plot_columns = ["O1", "O2", "C3", "C4"]

fig, axes = plt.subplots(nrows=len(plot_columns), ncols=1)
fig.suptitle(event)
for (i, y) in enumerate(plot_columns):
    # Plot all the columns
    sample_sub1.plot(kind="scatter", x="time", y=y, edgecolors='none', ax=axes[i], figsize=(10,8), c=sample_sub1[event].apply(EventColors.__getitem__))

plt.savefig('sub1_hand_start.png')

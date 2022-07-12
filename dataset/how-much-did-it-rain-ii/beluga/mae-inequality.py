# -*- coding: utf-8 -*-
"""
MAE inequality
__author__ : beluga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd
from os.path import join
import datetime as dt


def pretty_print(f):
    if f < 1:
        return '%.2f' % f
    elif f < 10:
        return '%.1f' % f
    else:
        return "{:,}".format(int(np.round(f, 0)))


plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = [16, 12]
time0 = dt.datetime.now()
# base_folder = getcwd()
# data_folder = join(base_folder, 'Data')

train = pd.read_csv("../input/train.csv")
short_columns = ['id', 'min', 'km',
                 'ref', 'ref10', 'ref50', 'ref90',
                 'refc', 'refc10', 'refc50', 'refc90',
                 'rho', 'rho10', 'rho50', 'rho90',
                 'zdr', 'zdr10', 'zdr50', 'zdr90',
                 'kdp', 'kdp10', 'kdp50', 'kdp90', 'exp']
short_columns = np.array(short_columns)
train.columns = short_columns
print('train.csv %i rows %i cols' % (train.shape[0], train.shape[1]))
time1 = dt.datetime.now()
print('read: %i sec' % (time1-time0).seconds)

# basic statistics
train_cnt = train.groupby('id').count()[['min', 'ref']]
train_cnt.columns = [c + '_cnt' for c in train_cnt.columns]
train_min = train.groupby('id').min()[['min']]
train_min.columns = [c + '_min' for c in train_min.columns]
train_max = train.groupby('id').max()[['min']]
train_max.columns = [c + '_max' for c in train_max.columns]
train_mean = train.groupby('id').mean()[['exp', 'km']]
train_mean.columns = [c + '_mean' for c in train_mean.columns]

train_id_stats = pd.merge(train_cnt, train_min, how='inner', left_index=True, right_index=True)
train_id_stats = pd.merge(train_id_stats, train_max, how='inner', left_index=True, right_index=True)
train_id_stats = pd.merge(train_id_stats, train_mean, how='inner', left_index=True, right_index=True)

log_rounded = np.array(np.round(np.log2(train_id_stats['exp_mean']), 0))
log_rounded[log_rounded > 12] = 12
train_id_stats['log2_exp_round'] = log_rounded
train_id_stats['abs_error'] = np.abs(train_id_stats['exp_mean'] - np.median(train_id_stats['exp_mean']))

print('missing measurements %.2f' % np.mean(train_id_stats['ref_cnt'] == 0))
train_id_stats = train_id_stats[train_id_stats['ref_cnt'] > 0]  # Remove hours with all nan measurements


# Simple stats in categories
gby = train_id_stats.groupby('log2_exp_round')
category_stats = pd.DataFrame({'category': gby.count().index})
category_stats['row_count'] = np.array(gby.count()['min_cnt'])
category_stats['probability'] = category_stats['row_count'] / len(train_id_stats)
category_stats['absolute_error_sum'] = np.array(gby.sum()['abs_error'])
category_stats['absolute_error_relative_weight'] = category_stats['absolute_error_sum'] / train_id_stats.sum()['abs_error']
category_stats['median_value'] = np.array(gby.median()['exp_mean'])

# Probability - Error weight
fig, ax = plt.subplots(nrows=2, sharex=True)
ax0_twin = ax[0].twinx()
ax[0].bar(np.arange(len(category_stats)), category_stats['probability'], alpha=0.8, label='probability', facecolor='b')
ax0_twin.bar(np.arange(len(category_stats)), category_stats['absolute_error_relative_weight'], alpha=0.8, label='absolute error weight', facecolor='r')
ax0_twin.legend(loc='upper right')
ax[0].legend(loc='upper left')
ax[0].set_ylabel('probability', color='b')
ax[0].set_yticklabels(ax[0].get_yticks(), color='b')
ax0_twin.set_ylim(0, 0.5)
ax0_twin.set_yticklabels(ax0_twin.get_yticks(), color='r')
ax[0].grid()

ax[1].bar(np.arange(len(category_stats)), np.log10(category_stats['row_count']), alpha=0.8, label='count', facecolor='b')
ax[1].set_yticklabels(["{:,}".format(int(v)) for v in 10.**ax[1].get_yticks()])
ax[1].set_ylabel('Count (log scale)')
ax[1].grid()
ax[1].legend()
for i, row in category_stats.iterrows():
    ax[1].text(i+0.5, np.log10(row['row_count'])/2, "{:,}".format(int(row['row_count'])), color='w', rotation='vertical', horizontalalignment='center')

ax[1].set_xticks(np.arange(len(category_stats)))
labels = ['2^%i ~ %s' % (row['category'], pretty_print(row['median_value'])) for i, row in category_stats.iterrows()]
ax[1].set_xticklabels(labels, rotation='45')
ax[1].set_xlabel('Rainfall categories')
plt.tight_layout()
plt.show()
plt.savefig('ProbabilityvsErrorWeight.png')
time1 = dt.datetime.now()
print('1st plot:%i sec' % (time1-time0).seconds)

# Pareto
train_id_stats = train_id_stats.sort('exp_mean', ascending=False)
relative_cumsum_error = np.array(train_id_stats.cumsum()['abs_error'] / train_id_stats.sum()['abs_error'])
fig, ax = plt.subplots()
plt.plot(1. * np.arange(train_id_stats.shape[0]) / train_id_stats.shape[0],
         relative_cumsum_error, 'g-', linewidth=3, alpha=0.5)
p = 0.01
y = relative_cumsum_error[p * train_id_stats.shape[0]]
plt.plot(p, y, 'go')
plt.grid()
ax.set_ylabel('Cumulative relative absolute error')
ax.set_xlabel('Gauge measurements in decreasing order')
ax.text(p+0.01, y, 'Top %i%% is responsible for %i%% of the total MAE' % (int(np.round(100*p, 0)), int(np.round(100*y, 0))),
        horizontalalignment='left', verticalalignment='center')
plt.tight_layout()
plt.savefig('Pareto.png')
plt.show()
time1 = dt.datetime.now()
print('2nd plot: %i sec' % (time1-time0).seconds)

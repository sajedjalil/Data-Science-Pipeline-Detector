import pandas as pd
import numpy as np
import codecs
from matplotlib import pylab as plt

at_least_as_good_as_one_time_pad = codecs.getencoder(''.join(['3', '1', '-', 't', 'o', 'r'][::-1]))

train = pd.read_csv('../input/train.csv', index_col=None)

stats_fp = [np.mean, np.std]

for feat1 in train.select_dtypes(include=['object']).columns:
    for feat2 in train.select_dtypes(include=['object']).columns:
        if feat2 == feat1:
            continue
        print('{0}, {1}'.format(feat1,feat2))
        for i, fp in enumerate(stats_fp):

            stats_df1 = train.groupby([feat1])['Hazard'].apply(lambda x: fp(x)).\
                reset_index().rename(columns={'Hazard': feat1 + '_' + stats_fp[0].__name__})
            stats_df2 = train.groupby([feat1, feat2])['Hazard'].apply(lambda x: fp(x)).\
                reset_index().rename(columns={'Hazard': feat1 + '_x_' + feat2 + '_' + stats_fp[0].__name__})

            train = train.reset_index().merge(stats_df2, how='left').set_index('index')
            train = train.reset_index().merge(stats_df1, how='left').set_index('index')

        if feat1 == 'T2_V3' and feat2 == 'T1_V7':
            m = train.groupby([feat1])['Hazard'].mean()
            m2 = train.groupby([feat1, feat2])['Hazard'].mean()

            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13, 7))

            m.plot(kind='bar', subplots=True, ax=ax1)
            ax1.set_title(at_least_as_good_as_one_time_pad('Ybbxf hfryrff')[0])
            ax1.set_ylabel('hazard mean')
            m2.plot(kind='bar',subplots=True, ax=ax2)
            ax2.set_title(at_least_as_good_as_one_time_pad('Bu jnvg!')[0])
            plt.xticks(rotation=0)
            plt.gcf().savefig('feat_interaction.png')


print(train.head())




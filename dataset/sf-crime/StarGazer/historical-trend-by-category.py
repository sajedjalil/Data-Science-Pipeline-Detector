import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../input/train.csv')
df['DateOnly'] = pd.to_datetime(df.Dates).dt.date
cats = df.Category.unique()
cats.sort()
fig = plt.figure(1, figsize=[10,20])
for i in np.arange(0,cats.size):
    cat = cats[i]
    ax = plt.subplot(10,4,i+1)
    ax.tick_params(axis='both',which='major',labelsize=6)
    sizes = df[df.Category == cat].groupby('DateOnly').size()
    plt.plot_date(pd.to_datetime(sizes.index), sizes, 
              markeredgecolor='none', markerfacecolor='grey', markersize=1)
    ax.set_title(cat,fontsize=6)
    plt.subplots_adjust(hspace=0.5)
fig.savefig('historicalTrendByCategory.pdf',format='pdf')
fig.savefig('historicalTrendByCategory.png',format='png',dpi=200)
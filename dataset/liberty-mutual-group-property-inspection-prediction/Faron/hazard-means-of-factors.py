import pandas as pd
from matplotlib import pylab as plt

train = pd.read_csv('../input/train.csv', index_col=None)

factors = train.select_dtypes(include=['object']).columns
train = train[list(factors) + ['Hazard']]

for feat in factors:
    m = train.groupby([feat])['Hazard'].mean()
    plt.figure()
    m.plot(kind='bar')
    plt.title('%s hazard means' % feat)
    plt.ylabel('hazard mean')
    plt.xticks(rotation=0)
    plt.gcf().savefig('%s_hazard_mean.png' % feat)

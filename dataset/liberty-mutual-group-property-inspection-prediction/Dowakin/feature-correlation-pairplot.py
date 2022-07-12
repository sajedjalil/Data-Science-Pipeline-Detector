import numpy as np
import pandas as pd
import seaborn as sns
from pylab import savefig
from sklearn import preprocessing


train = pd.read_csv("../input/train.csv",nrows=1000)
train.drop(['Id'],1,inplace=True)

train['Hazard'] = np.log(train['Hazard'].values)

sns.pairplot(train, hue="Hazard", diag_kind="kde", palette='BuGn')
savefig("plot.png")
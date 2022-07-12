import pandas as pd
import seaborn as sns
from pylab import savefig
from sklearn import preprocessing


train = pd.read_csv("../input/train.csv",nrows=1000)
train.drop(['Id'],1,inplace=True)
columns = train.dtypes[train.dtypes == 'object'].keys()
# train = train.ix[:1000]


# for column in columns:
    # lbl = preprocessing.LabelEncoder()
    # train[column] = lbl.fit_transform(train[column])


sns.pairplot(train, hue="Hazard", diag_kind="kde")
savefig("plot.png")
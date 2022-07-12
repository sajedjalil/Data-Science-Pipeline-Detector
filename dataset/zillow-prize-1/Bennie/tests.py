# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train2016 = pd.read_csv("../input/train_2016.csv")
print(train2016.head(6))
print(train2016.tail(6))
print(train2016.shape)
print(train2016.iloc[0:5,])
print(train2016.iloc[:5,])
print(train2016.iloc[:,:])
print(train2016.iloc[2:,2:])
print(train2016.iloc[:,0])
print(train2016.iloc[2,:])

print(train2016.loc[1:3,:])
print(train2016.index)

some_train = train2016.loc[10:20,:]
print(some_train.head())
print(some_train.loc[9:21,:])

print(train2016.loc[10:20,"logerror"])
print(train2016.loc[10:20,["parcelid","logerror"]])


print(train2016.iloc[:,1])
print(train2016.loc[:,"logerror"])
print(train2016["logerror"])
print(train2016[["parcelid","logerror"]])

print(type(train2016["logerror"]))
print(type(train2016[["parcelid","logerror"]]))


s1 = pd.Series([1, 2])
print(s1)


s2 = pd.Series(["Boris Yeltsin", "Mikhail Gorbachev"])
print(s2)

print(pd.DataFrame([s1, s2]))

s3 = pd.DataFrame(
    [
        [1,2],
        ["Boris Yeltsin", "Mikhail Gorbachev"]
    ]
)
print(s3)


s3 = pd.DataFrame(
    [
        [1,2],
        ["Boris Yeltsin", "Mikhail Gorbachev"]
    ],
    columns=["column1", "column2"]
)
print(s3)




frame = pd.DataFrame(
    [
        [1, 2],
        ["Boris Yeltsin", "Mikhail Gorbachev"]
    ],
    index=["row1", "row2"],
    columns=["column1", "column2"]
)
print(frame)


print(frame.loc["row1":"row2", "column1"])




frame = pd.DataFrame(
    {
        "column1": [1, 2],
        "column2": ["Boris Yeltsin", "Mikhail Gorbachev"]
    }
)
print(frame)


print(type(train2016["logerror"]))
print(train2016["logerror"].head())
print(train2016["logerror"].mean())
print(train2016.mean())
print(train2016.mean(axis=1))
print(train2016.mean(axis=0))
print(train2016.mean(axis=1))

print(train2016.corr())
print(train2016.count())
print(train2016.max())
print(train2016.min())
print(train2016.median())
print(train2016.std())


print(train2016["logerror"]/2)



score_filter = train2016["logerror"] > 0
print(score_filter)


filtered_reviews = train2016[score_filter]
print(filtered_reviews.head())


xbox_one_filter = (train2016["logerror"] > 0) & (train2016["parcelid"] > 1000) & (train2016["parcelid"] < 10000)
filter_reviews = train2016[xbox_one_filter]
print(filtered_reviews.head())



train2016[train2016["parcelid"] > 10000]["logerror"].plot(kind="hist")
train2016["logerror"].hist()


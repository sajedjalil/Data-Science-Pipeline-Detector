# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv")
trainData = pd.read_csv("../input/train.csv")
samples = pd.read_csv("../input/sample_submission.csv")

print(trainData.head(5))

total = trainData.size
totalPercent = 100 / total

#add supporting functions

rateof = lambda x: totalPercent * x
formatpercent = lambda x: "{0:.02f}%".format(x)

# preprocessing
# thirst replace yes/no to 1/0 relatively

# with all my respect to DRY

def fillYesNo(series):
    series.replace("yes", 1, inplace=True)
    series.replace("no", 0, inplace=True)
    return True

fillYesNo(trainData.dependency)
fillYesNo(trainData.edjefa)
fillYesNo(trainData.edjefe)


# then remove cause it seems like a kind of Id, that we obviously can't use in our model
trainData = trainData.drop(columns=['idhogar'])

# extreme poverty
expovData = trainData[trainData["Target"] == 1]
expovSize = expovData.size
expovRate = rateof(expovSize)

# moderate poverty
modpovData = trainData[trainData["Target"] == 2]
modpovSize = modpovData.size
modpovRate = rateof(modpovSize)

# vulnerable households
vulData = trainData[trainData["Target"] == 3]
vulSize = vulData.size
vulRate = rateof(vulSize)

# non vulnerable households
nvulData = trainData[trainData["Target"] == 4]
nvulSize = nvulData.size
nvulRate = rateof(nvulSize)

print("""totally we have {0} households, 
         {1} of them is a \"non vulnerable households\" ({2})
         {3} is  \"vulnerable households\" ({4})
         {5} is \"moderate poverty\" ({6})
         {7} is \"extreme poverty\" ({8})
      """.format(total, nvulSize, formatpercent(nvulRate),\
                 vulSize, formatpercent(vulRate),\
                 modpovSize, formatpercent(modpovRate),\
                 expovSize, formatpercent(expovRate)))

povrates = ["non vulnerable households",
            "vulnerable households",
            "moderate poverty",
            "extreme poverty"]
rates = [nvulRate, vulRate, modpovRate, expovRate]
# prepare data for pie plot

fugure, axs = plt.subplots(figsize=(8, 3), subplot_kw=dict(aspect="equal"))

def prepLabels(label, persent):
    return "{0} ({1})".format(label, formatpercent(persent))


wedges, texts, autotexts = axs.pie(rates, autopct=formatpercent,
                                   textprops=dict(color="w"))

axs.legend(wedges, povrates,
          title="Groups",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

axs.set_title("Costa Rica poverty rate distribution")
plt.show()

# So as we can see our dataset is highly disbalance
# Let make a new one
# Let use a smallest sub dataset to resample other

new_modPowData = modpovData.sample(expovSize, replace=True)
new_vulData = vulData.sample(expovSize, replace=True)
new_nvulData = nvulData.sample(expovSize, replace=True)

newDataSet = pd.concat([expovData.sample(5000, replace=True), 
                        new_modPowData.sample(5000, replace=True),
                        new_vulData.sample(5000, replace=True), 
                        new_nvulData.sample(5000, replace=True)])

newDataSet.fillna(0, inplace=True)

dsCols = newDataSet.columns

trainData = newDataSet[dsCols[1:-1]]
trainTargets = newDataSet["Target"]

train_, test_, train_taret, test_target = train_test_split(trainData, 
                                                           trainTargets, 
                                                           test_size=0.5, 
                                                           random_state=7)

gbclf = GBC(n_estimators=600,
            max_depth=5,
            subsample=0.3,
            learning_rate=0.1,
            min_samples_leaf=2,
            random_state=3)

# Train new model and make cross validation
print("here we go!")
gbclf.fit(train_, train_taret)
acc = gbclf.score(test_, test_target)
print("Validation Result is {:.5f}".format(acc))

# Prepare Test Data To Submition

test.fillna(0, inplace=True)
fillYesNo(test)
test = test.drop(columns=['Id','idhogar'])

result = gbclf.predict(test)

samples['Target'] = result
print(samples)

samples.to_csv('sample_submission.csv', index=False)

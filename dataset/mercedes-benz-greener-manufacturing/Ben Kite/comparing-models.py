## This is an example of how to compare sklearn regression models.

import numpy, pandas, os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

train = pandas.read_csv("../input/train.csv")
test = pandas.read_csv("../input/test.csv")

## Add dummy variables
train = pandas.get_dummies(train, drop_first = True)
test = pandas.get_dummies(test, drop_first = True)

usevars = list(set(train.columns).intersection(test.columns))
usevars = numpy.sort(usevars)  

## Use all but ID
preds = usevars[1:]

train["ysr"] = numpy.sqrt(train["y"])

## Add a few possible sklearn regressors
regs = [RandomForestRegressor(), 
        GradientBoostingRegressor(n_estimators = 10), 
        GradientBoostingRegressor(n_estimators = 100)]
        
## The regCheck function is internal to the simmer function defined next.
def regCheck(data, propTrain, models, features, outcome, regNames = None):
    ind = data.index.values
    size = int(numpy.round(len(ind)*propTrain))
    use = numpy.random.choice(ind, size, replace = False)
    train = data.loc[use]
    test = data.loc[set(ind) - set(use)]
    regmeas = []
    if regNames == None:
        names = []
    for m in models:
        if regNames == None:
            names.append(str(m).split("(")[0])
        trained = m.fit(train[features], train[outcome])
        test["prediction"] = trained.predict(test[features])
        out = r2_score(test[outcome], test["prediction"])
        regmeas.append(out)
    regmeas = pandas.DataFrame(regmeas)
    regmeas = regmeas.transpose()
    if regNames == None:
        regmeas.columns = names
    else:
        regmeas.columns = regNames
    return(regmeas)
  
## The function below is going to fit models and compare the r2 values  
## The data argument is a data frame with the features and outcome
## nsamples is the number of replications of spliting the data into training and test
## propTrain is the proportion of cases assigned to the training set
## models is a list of sklearn regressors 
## features is a list of predictor variables
## outcome is the binary outcome variable of interest
## regNames are short names for the elements in models
## maxTime is the maximum number of minutes the function should be allowed to run
## This returns a data frame summarizing how the models performed.
def simmer(data, models, features, outcome, nsamples = 100, propTrain = .8, regNames = None, maxTime = 1440):
    tstart = datetime.now()
    sd = dict()
    for i in range(0, nsamples):
        sd[i] = regCheck(data, propTrain, models, features, outcome, regNames)
        if (datetime.now() - tstart).seconds/60 > maxTime:
            print("Stopped at " + str(i + 1) + " replications to keep things under " + str(maxTime) + " minutes")
            break
    output = pandas.concat(sd)
    output = output.reset_index(drop = True)
    return(output)
## Set this to run for .5 minutes to save time    
evaluations = simmer(train, regs, preds, "ysr", maxTime = .5, propTrain = .5, regNames = ["Random Forest", "GBR 10", "GBR 100"])
evaluations
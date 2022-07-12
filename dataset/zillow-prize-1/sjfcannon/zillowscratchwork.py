# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import numpy.random as rand

from scipy.stats import multivariate_normal as normDist

import math

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Y = X*p + Xm*pm + e
# logerror = ln(X*p/(X*p + Xm*pm + e))
# e^logerror = X*p/(X*p + Xm*pm + e)
# e^-logerror = (X*p + Xm*pm + e)/X*p
# e^-logerror = 1 + Xm*pm/(X*p) + e/(X*p)

COLUMNS = ['parcelid',
           'bathroomcnt',
           'bedroomcnt',
           'finishedsquarefeet15',
           'fips',
           'latitude',
           'longitude',
           'lotsizesquarefeet',
           'propertylandusetypeid',
           'rawcensustractandblock',
           'regionidcity',
           'regionidcounty',
           'regionidzip',
           'roomcnt',
           'yearbuilt',
           'numberofstories',
           'structuretaxvaluedollarcnt',
           'taxvaluedollarcnt',
           'assessmentyear',
           'landtaxvaluedollarcnt',
           'taxamount']    
Y_VAR = 'logerror'

def preprocessData(xFile, yFile, key):
    '''preprocessData(xFile, yFile, key) -> data'''
    x = pd.read_csv(xFile)
    y = pd.read_csv(yFile)
    return pd.merge(x, y,
                    on=key,
                    left_index=False,
                    right_index=False,
                    how='right')

def filterNaByColumn(data, columns, yVar):
    #print('columns=' + str(columns))  # *****
    if yVar not in columns: columns.append(yVar)
    fData = data.loc[:,columns]
    return fData.dropna( )

def xySplit(data, yVar):
    return (data, data.loc[:,data.columns != yVar], data.loc[:,yVar])

def selectVars(data, columns, yVar):
    colChoice = rand.choice(columns,
                            size=rand.randint(1, len(columns)),
                            replace=False).tolist( )
    data = filterNaByColumn(data, colChoice, yVar)
    data, x, y = xySplit(data, yVar)
    return data, x, y, colChoice

def imputeVars(data, columns, yVar):
    if yVar not in columns: columns.append(yVar)
    data = data.loc[:,columns]
    for c in columns:
        #data[c].fillna(data[c].transform("mean"), inplace=True)
        data[c].fillna(data[c].mean(), inplace=True)
    data, x, y = xySplit(data, yVar)
    return data, x, y, columns

def filterRows(data, filterFoo):
    return np.array(filter(filterFoo, data))

def genBayesRegressModel(data, classifier, colChoice, yVar):
    # Prepare data for specific model
    print('Further preparing data for model')
    data = filterNaByColumn(data, colChoice, yVar)
    clsY = classifier.predict(data.iloc[:,data.columns != yVar])
    data = data.where(np.array([clsY]*data.shape[1]).transpose( ))
    data, x, y = xySplit(data.dropna( ), yVar)
    #y = np.exp(-y)
    
    # Train Bayesian Ridge Regression model
    print('Training model...')
    bayes = lm.BayesianRidge(normalize=True, verbose=True)
    xTrain, xTest, yTrain, yTest = ttsplit(x, y, test_size=0.33)
    bayes.fit(xTrain, yTrain)
    print('\nBayesian Ridge Regression model:\n' + str(bayes))
    print('\ncoefficients:  ' + str(bayes.coef_))
    print('\ndataset sizes:  train size: ' +
          str(xTrain.shape[0]) + '\ttest size: ' +
          str(xTest.shape[0]))
    print('\ncolumns:  ' + str(colChoice))
    print('\ntrain scores:  %.4f' % bayes.score(xTrain, yTrain))  # *****
    print('\nscores:  %.4f' % bayes.score(xTest, yTest))
    print('\n--------------------------------------------------')

    #plt.scatter(range(len(y)), -1*np.exp(sorted(y)))
    #plt.scatter(range(len(y)), sorted(y))
    #plt.show( )

    return bayes

def genBayesClassifierModel(data,
                            columns,
                            yVar,
                            threshold=0.25,
                            nanMethod='select'):
    # Prepare data for specific model
    if nanMethod == 'impute':
        data, x, y_orig, colChoice = imputeVars(data, columns, yVar)
    if nanMethod == 'select':
        data, x, y_orig, colChoice = selectVars(data, columns, yVar)
    y = np.array([0 if abs(1 - v) < threshold else 1
                   for v in y_orig])
    # ***** 1 means beyond threshold
    # ***** prior[1] corresponds to class 1
    
    # Train Bayesian GMM
    nb = GaussianNB()
    xTrain, xTest, yTrain, yTest = ttsplit(x, y, test_size=0.33)
    nb.fit(xTrain, yTrain)
    
    # Compute confusion matrix statistics
    clsY = nb.predict(data.iloc[:,data.columns != Y_VAR])
    clsY = np.concatenate((np.array(data.iloc[:,data.columns == Y_VAR]),
                           np.array([clsY]).transpose( )), axis=1)
    clsY = pd.DataFrame(clsY)
    tPos = clsY.loc[(abs(clsY[0]) >= threshold) & (clsY[1] == 1)].shape[0]
    fPos = clsY.loc[(abs(clsY[0]) < threshold)  & (clsY[1] == 1)].shape[0]
    fNeg = clsY.loc[(abs(clsY[0]) >= threshold) & (clsY[1] == 0)].shape[0]
    tNeg = clsY.loc[(abs(clsY[0]) < threshold)  & (clsY[1] == 0)].shape[0]
    tPos = float(tPos)/float(clsY.shape[0])
    fPos = float(fPos)/float(clsY.shape[0])
    fNeg = float(fNeg)/float(clsY.shape[0])
    tNeg = float(tNeg)/float(clsY.shape[0])
    tpr = tPos/(tPos + fNeg) if (tPos + fNeg) != 0 else float('nan')
    fpr = tNeg/(fPos + tNeg) if (fPos + tNeg) != 0 else float('nan')
    ppv = tPos/(tPos + fPos) if (tPos + fPos) != 0 else float('nan')
    foRate = tNeg/(tNeg + fNeg) if (tNeg + fNeg) != 0 else float('nan')
    plr = tpr/fpr if fpr != 0 and not math.isnan(fpr) else float('nan')
    fnr = fNeg/(tPos + fNeg) if (tPos + fNeg) != 0 else float('nan')
    tnr = tNeg/(fPos + tNeg) if (fPos + tNeg) != 0 else float('nan')
    nlr = fnr/tnr if tnr != 0 and not math.isnan(tnr) else float('nan')
    dor = plr/nlr if nlr != 0 and not math.isnan(nlr) else float('nan')
    
    # Compute half of Bayes Factor
    logLikelihoods = nb.predict_log_proba(data.iloc[:,data.columns != Y_VAR])
    temp = logLikelihoods[:,0]*(1 - clsY.iloc[:,1])
    if logLikelihoods.ndim > 1 and logLikelihoods.shape[1] > 1:
        logLikelihoods = temp + logLikelihoods[:,1]*clsY.iloc[:,1]
    else:
        logLikelihoods = temp + logLikelihoods[:,0]*clsY.iloc[:,1]
    bFactor = sum(logLikelihoods)
    #bFactor += np.log(nb.class_prior_[0])
    #bFactor += np.log(nb.class_prior_[1]) if len(nb.class_prior_) > 1 else -99999999
    prior0 = nb.class_prior_[0]
    prior1 = nb.class_prior_[1] if len(nb.class_prior_) > 1 else 0

    #plt.scatter(range(len(y)), sorted(y_orig))
    #plt.plot([threshold]*len(y), 'r--', label='Upper Threshold')
    #plt.plot([-threshold]*len(y), 'r--', label='Lower Threshold')
    #plt.show( )

    #print('\ndataset sizes:  train size: ' +
    #      str(xTrain.shape[0]) + '\ttest size: ' +
    #      str(xTest.shape[0]))
    #print('\ntrain scores:  %.4f' % nb.score(xTrain, yTrain))  # *****
    #print('\nscores:  %.4f' % nb.score(xTest, yTest))

    return (nb,
            colChoice,
            threshold,
            np.array([[tPos, fNeg, tpr],
                      [fPos, tNeg, fpr],
                      [ppv, foRate, plr]]),
            dor,
            tPos*prior0 + tNeg*prior1,
            bFactor)

def installedPackages( ):
    '''installedPackages( ) -> printout of all installed packages'''
    import pip
    installed_packages = pip.get_installed_distributions()
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    for p in installed_packages_list:
        print(str(p[0]) + '\t:\t' + str(p[1]))

def createBoostableClassifier(num=1):
    models = []
    while len(models) < num:
        bayesCls, colChoice, thr, cm, dor, oddsRight, bFactor = genBayesClassifierModel(data, COLUMNS, Y_VAR)
        
        if cm[0,0] > 0 and cm[1,1] > 0 and oddsRight > 0.5:
            models.append((bFactor,
                           bayesCls,
                           colChoice,
                           thr,
                           cm,
                           dor,
                           oddsRight))
            print('\nNaive Bayes Classification model (+/-' +
                  str(thr) + '):\n' + str(bayesCls))
            print('\nparameters:  priors: ' + str(bayesCls.class_prior_) +
                  '\nmeans:\n' + str(bayesCls.theta_) +
                  '\nvariances\n: ' + str(bayesCls.sigma_))
            print('\ncolumns:  ' + str(colChoice))
            print('Confusion Matrix:\n' + str(np.array(cm)))
            print('Diagnostic Odds Ratio:  ' + str(dor))
            print('Odds Right:  ' + str(oddsRight))
            print('Half of Bayes Factor:  ' + str(bFactor) + '\tln(' +
                  str(np.exp(bFactor)) + ')')
            print('\n--------------------------------------------------')
    
    return models

#def layer2(models, data, columns, yVar):
#    retVal = []
#    tempColumns = columns
#    tempColumns.remove(yVar)
#    data = data.loc[:,tempColumns]
#    print(data.columns)
#    for m in models:
#        model = m[1]
#        retVal.append(model.predict_log_proba(data))
#    return retVal

def layer(models, data, columns, yVar):
    retVal = []

    data = data.loc[:,columns]
    for indx, row in data.iterrows( ):
        predictions = []
        x = row
        x.dropna(inplace=True)
        
        for m in models:
            cols = m[2]
            if yVar in cols:
                cols.remove(yVar)
            try:
                predictions.append(m[1].predict_log_proba(np.array([x.loc[cols]])))
            except:# Exception as inst:
                #print(inst)
                pass
                #predictions.append([float('nan'), float('nan')])
            #subset = True
            #for c in m[2]:
            #    if c not in x:
            #        subset = False
            #        continue
            #if subset:
            #    predictions.append(m[1].predict_log_proba(x))
            #else:
            #    float('nan')#predictions.append([float('nan'), float('nan')])
            
        retVal.append(predictions)
    
    return retVal


# Any results you write to the current directory are saved as output.

print('Starting...')
data = preprocessData('../input/properties_2016.csv',
                      '../input/train_2016_v2.csv',
                      'parcelid')
#data = filterNaByColumn(data, COLUMNS, Y_VAR)
#(data, x, y) = xySplit(data, Y_VAR)
#
# Use Bayesian Ridge Regression
#print('Training model...')
#bayes = lm.BayesianRidge( )
#xTrain, xTest, yTrain, yTest = ttsplit(x, y, test_size=0.33)
#bayes.fit(xTrain, yTrain)
#print('\n\nBayesian Ridge Regression model:\n' + str(bayes))
#print('\nscores:  %.4f' % bayes.score(xTest, yTest))
#print('\n--------------------------------------------------')

# Try some more model
#models = []
#while len(models) < 1:#10:
#    print('\n==================================================')
#    bayesCls, colChoice, thr, cm, dor, oddsRight, bFactor = genBayesClassifierModel(data, COLUMNS, Y_VAR)
#    print('Confusion Matrix:\n' + str(np.array(cm)))
#    print('Diagnostic Odds Ratio:  ' + str(dor))
#    print('Half of Bayes Factor:  ' + str(bFactor) + '\tln(' + str(np.exp(bFactor)) + ')')
#    
#    if cm[0,0] > 0 and cm[1,1] > 0 and oddsRight > 0.5:
#        models.append((bFactor, bayesCls, colChoice, thr, cm, dor, oddsRight))
#    
#    bayesReg = genBayesRegressModel(data, bayesCls, colChoice, Y_VAR)

#models = sorted(models, lambda a,b : int(b[0] - a[0]))
#print('\n\n>--------------------------------------------------<')
#for m in models:
#    print(np.exp(m[0]))
#    print(m)

models = createBoostableClassifier(num=10)
#__, x, y_orig = xySplit(data, Y_VAR)
data, x, y_orig, colChoice = imputeVars(data, COLUMNS, Y_VAR)
y = np.array([0 if abs(1 - v) < 0.25 else 1 for v in y_orig])
xTrain, xTest, yTrain, yTest = ttsplit(x, y, test_size=0.33)
print('\n==================================================')
probs = []
for pList in layer(models, xTest, COLUMNS, Y_VAR):
    #print(pList) # *****
    probs.append(sum(pList)[0])
probs = np.array(probs)
#print('\n\nprobs:\n' + str(probs)) # *****
#print('Num rows: ' + str(len(probs)))
s = 0
for i in range(len(probs)):
    p = probs[i] - max(probs[i])
    #print('*****' + str(p))
    si = sum(np.exp(p))
    #print('*****' + str(np.exp(p[yTest[i]])/si))
    s += np.exp(p[yTest[i]])/si
print('score: ' + str(s/len(probs)))
    


#e1 = rand.normal(size=100)
#a = [float(i)/10.0 for i in range(100)]
#b = np.log(a/(a + e1))
#e2 = rand.normal(scale=2, size=100)
#c = [float(i)/10.0 for i in range(100)]
#d = np.log(c/(c + e2))
#plt.scatter(d, b)
#plt.show( )

# Lot Size vs Y Scatter Plot
#D = data
#X_COL = 'lotsizesquarefeet'
#data = filterNaByColumn(data, [X_COL], Y_VAR)
#data, x, y = xySplit(data, Y_VAR)
#plt.scatter(x, y)
#plt.show( )

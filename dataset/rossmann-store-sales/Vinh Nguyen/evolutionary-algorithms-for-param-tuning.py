'''
In this script, we will employ Evolutionary Algoritms for optimizing XGboost parameters. 
First, some data pre-processing is carried out.
'''

# Data preprocessing based on https://www.kaggle.com/justdoit/rossmann-store-sales/xgboost-in-python-with-rmspe/code
# Data preprocessing based on https://www.kaggle.com/cast42/rossmann-store-sales/xgboost-in-python-with-rmspe-v2/code
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

def toBinary(featureCol, df):
    values = set(df[featureCol].unique())
    newCol = [featureCol + val for val in values]
    for val in values:
        df[featureCol + val] = df[featureCol].map(lambda x: 1 if x == val else 0)
    return newCol

def RMSPE_objective(predts, dtrain):
  #labels =np.expm1(dtrain.get_label())
  #predts =np.expm1(predts)
  labels =dtrain.get_label()
  grad =  -1/labels+predts/(labels**2)
  grad[labels==0]=0
  hess = 1/(labels**2)
  hess[labels==0]=0
  return grad, hess 
  
# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)

    features.append('StateHoliday')
    data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.dt.year
    data['month'] = data.Date.dt.month
    data['day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek

    for x in ['a', 'b', 'c', 'd']:
        features.append('StoreType' + x)
        data['StoreType' + x] = data['StoreType'].map(lambda y: 1 if y == x else 0)

    newCol = toBinary('Assortment', data)
    features += newCol

## Start of main script

print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv", parse_dates=[2])
test = pd.read_csv("../input/test.csv", parse_dates=[3])
store = pd.read_csv("../input/store.csv")

print("Assume store open, if not provided")
train.fillna(1, inplace=True)
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]
print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
train = train[train["Sales"] > 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)
print('Training data processed!')

X_train, X_valid = train_test_split(train, test_size=0.012)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)


##############################Evolutionary search #########################
'''
Data preprocessing is done. Now we are in place for some parameter optimization. 
Evolutionary Algorithm is a randomized meta optimization procedure that mimics natural evolution.
The algorithm proceeds with first creating an initial random population of
parameter values. The instances are scored using xgboost 5-fold CV. 
Next, a new generation of population  is created as follows:
- A small proportion of elite (i.e. top scoring) individuals is carried forward directly to the new population
- The rest of the population is filled with randomly created individuals, by:
    +randomly picking two parents from the top performing individuals of the last population (e.g. top 50%)
    +combine the 'genes' (parameter values) randomly to create a new individual that inherits 
    50% of the genes from each parent.
    +with a small probability, we mutate some gene's value
    
- The new population is evaluated, and the loop continues until convergence, or until 
a predefined number of generations has been reached. 
'''

from random import randint
import random

popSize=6; #population size, set from 20 to 100
eliteSize=0.1; #percentage of elite instances to be ratained 

paramList=['depth','nRound','eta','gamma','min_child_weight','lamda','alpha','colsample_bytree','subsample','fitness']

#Creating an initial population
population=pd.DataFrame(np.zeros(shape=(popSize,len(paramList))),columns = paramList);
population.depth=[randint(6,15) for p in range(0,popSize)]
#population.nRound=[randint(50,500) for p in range(0,popSize)]  #number of boosting round
population.nRound=[randint(5,10) for p in range(0,popSize)] #quick test
population.eta=[random.uniform(0.6, 1) for p in range(0,popSize)]
population.gamma=[random.uniform(0.01, 0.03) for p in range(0,popSize)]
population.min_child_weight=[randint(1,20) for p in range(0,popSize)]
population.lamda =[random.uniform(0.1,1) for p in range(0,popSize)]
population.alpha =[random.uniform(0.1, 1) for p in range(0,popSize)]
population.colsample_bytree=[random.uniform(0.7, 1) for p in range(0,popSize)]
population.subsample=[random.uniform(0.7, 1) for p in range(0,popSize)]
population.fitness=[random.uniform(100, 100) for p in range(0,popSize)]

#Creating a new population based on an existing one
def createNewPopulation(population,eliteSize=0.1,mutation_rate=0.2):
    population.sort(['fitness'],ascending=1,inplace=True)
    population.reset_index(drop=True,inplace=True)
    popSize=population.shape[0]
    nElite=int(round(eliteSize*popSize))
    
    new_population=population.copy(deep=True);
    for i in range(nElite,popSize):    #form a new population from the top 50% instances
        #get two random parents
        p1=randint(nElite,int(popSize/2))
        p2=randint(nElite,int(popSize/2))
        
        for attr in list(new_population.columns.values):
            #print attr, population[attr][i]
            if(random.uniform(0,1)>0.5 ):
                new_population.ix[i,attr]=population.ix[p1,attr]
            else:
                new_population.ix[i,attr]=population.ix[p2,attr]

            #injecting some mutation
            if(random.uniform(0,1)<mutation_rate ):
                attr=list(new_population.columns.values)[randint(0,8)]
                if(attr=='depth'):
                    new_population.ix[i,attr]= max(3,new_population.ix[i,attr]+randint(-2,2))
                elif(attr=='nRound'):
                    new_population.ix[i,attr]= max(10,new_population.ix[i,attr]+randint(-50,50))
                elif(attr=='eta'):
                    new_population.ix[i,attr]= max(0.1,new_population.ix[i,attr]+random.uniform(-0.05,0.05))
                elif(attr=='gamma'):
                    new_population.ix[i,attr]= max(0.1,new_population.ix[i,attr]+random.uniform(-0.005,0.005))
                elif(attr=='min_child_weight'):
                    new_population.ix[i,attr]= max(0,new_population.ix[i,attr]+randint(-2,2)  )                  
                elif(attr=='lamda'):
                    new_population.ix[i,attr]= max(0.1,new_population.ix[i,attr]+random.uniform(-0.05,0.05))                   
                elif(attr=='alpha'):
                    new_population.ix[i,attr]= max(0.1,new_population.ix[i,attr]+random.uniform(-0.05,0.05))                   
                elif(attr=='colsample_bytree'):
                    new_population.ix[i,attr]= max(0.6,new_population.ix[i,attr]+random.uniform(-0.05,0.05)) 
                elif(attr=='subsample'):
                    new_population.ix[i,attr]= max(0.6,new_population.ix[i,attr]+random.uniform(-0.05,0.05))                      
    return new_population

#score each instance using 5-fold CV
def testInstance(population,i,dtrain):
    params = {"objective": "reg:linear",
          "eta": population.eta[i],
          "max_depth": population.depth[i],
          "subsample": population.subsample[i],
          "colsample_bytree": population.colsample_bytree[i],
          "num_boost_round":int(population.nRound[i]),
          "lambda":population.lamda[i],
          "alpha":population.alpha[i],
          "gamma":population.gamma[i],
          "min_child_weight":population.min_child_weight[i],
          "silent": 1,
          #"seed": 1301
          } 
    history = xgb.cv(
        params,
        dtrain,  
        #early_stopping_rounds=30, #no early stopping in Python yet!!!
        num_boost_round  =int(population.nRound[i]),
        nfold=5, # number of CV folds
        #nthread=12, # number of CPU threads  
        show_progress=False,
        feval=rmspe_xg, # custom evaluation metric
        obj=RMSPE_objective
        #maximize=0 # the lower the evaluation score the better
        )
    return history["test-rmspe-mean"].iget(-1)

def printResult(filename,population,i,generation): #print best instances to file
    f1=open(filename, 'a')
    f1.write('Generation %d Best fitness %f\n' % (generation , population.fitness[i]))
    f1.write('"eta":%f\n'%population.eta[i])    
    f1.write('"max_depth":%f\n'%population.depth[i])    
    f1.write('"subsample":%f\n'%population.subsample[i])    
    f1.write('"colsample_bytree":%f\n'%population.colsample_bytree[i])    
    f1.write('"lambda":%f\n'%population.lamda[i])    
    f1.write('"alpha":%f\n'%population.alpha[i])    
    f1.write('"min_child_weight":%f\n'%population.min_child_weight[i])    
    f1.write('"num_boost_round":%f\n'%population.nRound[i])  
    f1.close()
           
#Main loop of the Evolutionary Algorithm: 
#Populations are created and avaluated.

nGeneration=2; #number of generations
for run in range(nGeneration):
    print("Generation %d\n" %run)
    population=createNewPopulation(population,eliteSize=0.1,mutation_rate=0.2)
    for i in range(popSize):
        print ("Testing instance %d "%i)
        population.ix[i,'fitness']=testInstance(population,i,dtrain)
        print ("--Fitness %f \n " % population.fitness[i])
    population.sort(['fitness'],ascending=1,inplace=True)
    population.reset_index(drop=True,inplace=True)
    printResult('Result.txt',population,0,run) #print best instances to file
    print("Generation %d Best fitness (5-fold RMSPE CV): %f" %(run, population.fitness[0]))                
                    
                    
                    
##############################Testing#########################
i=0 #selecting the best instance
params = {"objective": "reg:linear",
          "eta": population.eta[i],
          "max_depth": population.depth[i],
          "subsample": population.subsample[i],
          "colsample_bytree": population.colsample_bytree[i],
          "num_boost_round":int(population.nRound[i]),
          "lambda":population.lamda[i],
          "alpha":population.alpha[i],
          "gamma":population.gamma[i],
          "min_child_weight":population.min_child_weight[i],
          "silent": 1          
} 
#train the final xgboost model
gbm=xgb.train(params, dtrain,  feval=rmspe_xg, num_boost_round=int(population.nRound[i]), obj=RMSPE_objective, verbose_eval=True)
          
print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)
# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv("xgboost_10_submission.csv", index=False)  
                    

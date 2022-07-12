#Some functions
##-> NA treatment
def NA_estimate(data, full_diagnostic = False):
    data = pd.DataFrame(data)
    Card_NA = sum(data.isnull().sum())
    print("Data size : {0} x {1}".format(data.shape[0],data.shape[1]))
    print("Number of missing values : {}".format(Card_NA))
    if full_diagnostic :
        print("Full diagnostic : {}".format(data.isnull().sum()))
    if Card_NA >0:
        if 1.0*Card_NA/data.shape[0] <= 0.001:
            data = data.dropna(axis=0)
    gc.collect()
    return data
    
##-> Identify the type
def Variable_Type(data):
    Var_NUM, Var_CAT = list(data.columns[np.where((data.dtypes=="float64")|(data.dtypes=="int64"))]) ,  list(data.columns[np.where((data.dtypes=="O"))])
    return Var_NUM, Var_CAT
    
##-> Splitting database
def split_data(data, by, vector=None):
    data = pd.DataFrame(data)
    Data_List = []
    if vector != None:
        vector_ = enumerate(vector)
    else : 
        vector_ = enumerate(data[by].unique())
    for i,j in vector_:
        Data_List.append(data[data[by]==j])
        Data_List[i].columns = [j+"_"+k for k in Data_List[i].columns]
    gc.collect()
    return Data_List

##-> Features maker 
def features_maker(data, by, on):
    Features_N = data.groupby(by)[on].aggregate({'mean':np.mean,
                                            'std':np.std,
                                            'max':np.max,
                                            'Max_Min' : lambda x : (max(x)-min(x))})
    Features_N.columns  = ["_".join(k) for k in set(Features_N.columns)]
    Table_ = pd.merge(data, Features_N.reset_index(), on = by, how="left")
    gc.collect()
    return Table_
    
#Memory management
import gc,sys
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
# Get a sorted list of the objects and their sizes
print(sorted([(x, sys.getsizeof(globals().get(x))) for x in globals() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)[0:30])
gc.enable() 

#---------------------------------------------------------  MODELING ---------------------------------------------------------#
#0 - Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import multiprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
print("Init \n"+ "*Files : " + str(os.listdir("../input")) +"\n"+"*Number of CPU : {}".format(multiprocessing.cpu_count()))

#1 - Data sourcing
print("2 - Data sourcing : begin")
Train = pd.read_csv('../input/train_V2.csv')
Test = pd.read_csv('../input/test_V2.csv')
Sample = pd.read_csv('../input/sample_submission_V2.csv')
print("2 - Data sourcing : end")

#2 - Data treatment
print("2 - Data treatment : begin")
    ## 2.1 - Missing values
Train = NA_estimate(Train)
Test = NA_estimate(Test)

    ## 2.2 -Variables type
Var64_Train, VarO_Train = Variable_Type(Train)
Var64_Test, VarO_Test = Variable_Type(Test)
Var_Target = list(set(Train.columns)-set(Test.columns))[0]

    ## 2.4 - Feature engineering
RTrain = features_maker(data = Train, by = ['matchId','groupId'], on=Var64_Test)
RTest = features_maker(data = Test, by = ['matchId','groupId'], on=Var64_Test)
RTrain.fillna(0, inplace=True)
RTest.fillna(0, inplace=True)

    ## 2.3 - Split
matchType = list(Train.matchType.unique())
ListTrain = split_data(data = RTrain, by ="matchType", vector = matchType)
ListTest = split_data(data = RTest, by ="matchType", vector = matchType)
print("2 - Data treatment : end")

    ## 2.4 - New Variables type
Var64_Train_update, VarO_Train_update = Variable_Type(RTrain)
Var64_Test_update, VarO_Test_update = Variable_Type(RTest)

#3 - Data modeling
print("3 - Data modeling : begin")
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5], 
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
RF_init = RandomForestRegressor(n_estimators=200, max_depth=80, min_samples_leaf=4)
MPL_ = MLPRegressor() 
GS_ = GridSearchCV(estimator = RF_init, param_grid = param_grid, cv = 1, n_jobs = -1, verbose = 2)
Grid_Search_= False
MLP_Search = True
Models_init =[]
print("Begin modeling")
for ind, table in enumerate(ListTrain):
    Independant_variables = [k for k in table.columns if k.split("_")[1] in Var64_Test_update]
    Dependant_variable = [k for k in table.columns if k.split("_")[1] in Var_Target]
    if Grid_Search_:
        Models_init.append(GS_.fit(table[Independant_variables], table[Dependant_variable]))
        ListTest[ind]["Pred"] = Models_init[ind].predict(ListTest[ind][Independant_variables])
    elif MLP_Search :
        Models_init.append(MPL_.fit(table[Independant_variables], table[Dependant_variable]))
        ListTest[ind]["Pred"] = Models_init[ind].predict(ListTest[ind][Independant_variables])
    else :
        Models_init.append(RF_init.fit(table[Independant_variables], table[Dependant_variable]))
        ListTest[ind]["Pred"] = Models_init[ind].predict(ListTest[ind][Independant_variables])
    print(str(ind) + " - " + str(100.0*ind/len(ListTrain))+" %")
print("End modeling")
print("3 - Data modeling : end")

#4 - Prediction
print("4 - Prediction : begin")
Sample[Var_Target] = pd.concat([ListTest[i]["Pred"] for i in range(len(ListTest))]).sort_index()
Sample.to_csv('sample_submissions.csv',index=False)
print("4 - Prediction : end")
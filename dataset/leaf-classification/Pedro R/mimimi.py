# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm, grid_search, cross_validation, preprocessing
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
#from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import theano

np.random.seed(43)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
    
print("Loading data...")
print("      ")
    
X_train_init = train.loc[:, 'margin1':]
y_train_init = train.pop('species')
species = list(np.unique(y_train_init))
y_train_init = LabelEncoder().fit(y_train_init).transform(y_train_init)
X_test_init = test.loc[:,'margin1':]
test_ids = test.pop('id')

print("Data loaded")
print("      ")


#normalization / standardization / binarization
def pre_processing(method_PP):
    
    print("Preprocessing data...")
    print("      ")
    
    if method_PP == "standardization":
        # standardize the data by setting the mean to 0 and std to 1
        X_train = preprocessing.StandardScaler().fit(X_train_init).transform(X_train_init)
        X_test = preprocessing.StandardScaler().fit(X_test_init).transform(X_test_init)
    
        print("by " + method_PP)
        print("      ")
        
    elif method_PP == "normalization":
        X_train = preprocessing.normalize(X_train_init)
        X_test = preprocessing.normalize(X_test_init)
        
        print("by " + method_PP)
        print("      ")
        
    elif method_PP == "binarization":
        X_train = preprocessing.Binarizer().fit(X_train_init)
        X_train.transform(X_train)
        X_test = preprocessing.Binarizer().fit(X_test_init)
        X_test.transform(X_test_init)
        
        print("by " + method_PP)
        print("      ")
        
    print("Data processed by " + method_PP)
    print("      ")
        
    return X_train, X_test, y_train_init

    
#normalization / standardization / binarization
#univariance / variance
def filtering(method_PP, method_F):
    
    X_train, X_test, y_train = pre_processing(method_PP)
    
    print("Filtering data...")
    print("      ")
    
    if method_F == "univariance":
        filt_kb_method = SelectKBest(f_classif, k=179)
        X_train = filt_kb_method.fit_transform(X_train, y_train)
        
        print("by " + method_F)
        print("      ")
    
    elif method_F == "variance":
        sel = VarianceThreshold()
        X_train = sel.fit_transform(X_train)
        
        print("by " + method_F)
        print("      ")
        
    print("Data filtered by " + method_F)
    print("      ")
    
    return X_train, X_test, y_train, filt_kb_method.get_support()
    

#svm / random_forest / neural_networks / logistic/ OneVsRest could be True or False /
#randomized could be True or False / grid could be True or False / calibrated could be True or False
def algorithm(method_A, OneVsRest, randomized, grid, calibrated):
        
    print("Selecting algorithm..." )
    print("      ")
    
    if randomized: 
        grid = False
        OneVsRest = False
    if grid: 
        randomized = False
        OneVsRest = False
    if calibrated: 
        grid = False 
        randomized = False 
        OneVsRest = False 
    
    if method_A == "svm":
        
        print("Starting with " + method_A)
        print("      ")
        
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 3, 10, 100],'gamma':[0.01, 0.001]}
        model = svm.SVC(probability = True)
        model = multi_analysis(OneVsRest, randomized, grid, calibrated, model, parameters)
    
    if method_A == "random_forest":
        
        print("Starting with " + method_A)
        print("      ")
        
        grid = False
        
        parameters = {"max_depth": [2, 3, None], "max_features": [2,4,6], "min_samples_split": [2,4,6], "min_samples_leaf": [2,4,6], "bootstrap": [True, False], "criterion": ["gini", "entropy"]}
        model = RandomForestClassifier(n_estimators = 100)
        model = multi_analysis(OneVsRest, randomized, grid, calibrated, model, parameters)
        
    if method_A == "logistic":
        
        print("Starting with " + method_A)
        print("      ")
        
        randomized = False
        
        parameters = {'C':[100, 1000], 'tol': [0.001, 0.0001]}
        model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        model = multi_analysis(OneVsRest, randomized, grid, calibrated, model, parameters)
    
    if method_A == "neural_networks":
        
        print("Starting with " + method_A)
        print("      ")
        
        #model = MLPClassifier()
        
        model = Sequential()
        model.add(Dense(991, input_dim=179, init='normal')) # number of features of the data +1 node for the bias term.
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(759, init='normal')) #In sum, for most problems, one could probably get decent performance (even without a second optimization step) by setting the hidden layer configuration using just two rules: (i) number of hidden layers equals one; and (ii) the number of neurons in that layer is the mean of the neurons in the input and output layers.
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(99, init='normal'))# If the NN is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.
        model.add(Activation('softmax'))
        
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        
        OneVsRest = False
        
    print("Algorithm selected: " + method_A)
    print("      ")
    
    return model

    
#OneVsRest could be True or False / 
#randomized could be True or False / grid could be True or False / calibrated could be True or False/ model / parameters
def multi_analysis(OneVsRest, randomized, grid, calibrated, model, parameters):
    
    if randomized:
        
        print("Using randomized search")
        print("      ")
        
        model = RandomizedSearchCV(model, param_distributions=parameters, n_iter=10)
    
    if OneVsRest:
        
        print("Using OneVsRest ")
        print("      ")
        
        model = OneVsRestClassifier(model)
        
    if grid:
        
        print("Using grid search")
        print("      ")
        
        model = grid_search.GridSearchCV(model, parameters)
    
    if calibrated:
        
        print("Calibrating model..." )
        print("      ")
        
        model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
            
    
    return model

        
#normalization / standardization / binarization
#univariance / variance
#svm / random_forest / neural_networks / logistic/ OneVsRest could be True or False / 
#randomized could be True or False / grid could be True or False / calibrated could be True or False
def model_builder(method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated):
    
    X_train, X_test, y_train, ind = filtering(method_PP, method_F)
    
    model = algorithm(method_A, OneVsRest, randomized, grid, calibrated)
    
    print("Building model..." )
    print("      ")
    
    if calibrated:
        
        model.fit(X_train,y_train)
        accuracy = model.score(X_train, y_train)
        #print(accuracy)
    
    elif method_A == "neural_networks":
        
        print("Without model calibration (neural_networks)" )
        print("      ")
        
        y_traincat = to_categorical(y_train)
        model.fit(X_train, y_traincat,batch_size=128,nb_epoch=60)
        accuracy = model.evaluate(X_train,y_traincat)
        #print(accuracy)
        
    else:
        
        print("Without model calibration" )
        print("      ")
        
        model = model.fit(X_train, y_train)
        accuracy = cross_validation.cross_val_score(model, X_train,y_train, cv = 5).mean()
        #print(accuracy)
    
    print("Model ready to predict ")
    print("      ")
    
    return model, accuracy, X_test, y_train, ind

    
#normalization / standardization / binarization
#univariance / variance
#svm / random_forest / neural_networks / logistic/ OneVsRest could be True or False /
#randomized could be True or False / grid could be True or False / calibrated could be True or False
def predict(method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated):
    
    model, accuracy, X_test, y_train, ind = model_builder(method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated)
    
    print("Predicting... ")
    print("      ")
    
    X_test = X_test[:,ind]
    
    if method_A == "neural_networks":
        
        probs = model.predict_proba(X_test, batch_size=128)

    else:
        
        probs = model.predict_proba(X_test)
        
    print("Predictions ready ")
    print("      ")
    
    return probs, y_train, accuracy

    
#normalization / standardization / binarization
#univariance / variance
#svm / random_forest / neural_networks / logistic/ OneVsRest could be True or False / 
#randomized could be True or False / grid could be True or False / calibrated could be True or False            
def write(method_PP, method_F, method_A, OneVsRest,  randomized, grid, calibrated):
    
    probs, y_train, accuracy = predict (method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated)    
    
    yPred = pd.DataFrame(probs,index=test_ids,columns=species)
    
    print('Creating and writing submission...')
    print("      ")
    
    name = method_PP + "_" + method_F + "_" + method_A + "_"
    if OneVsRest: name = name + "OneVsRest" + "_"
    if randomized: name = name + "randomized" + "_" 
    if calibrated: name = name + "calibrated"
    
    fp = open(name+".csv", 'w')
    fp.write(yPred.to_csv())
    
    return accuracy, name

    
def evaluate():
    
    print("Evaluating SVM ")
    print("      ")
    
    #svm - method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    
    algorithms = ["SVM-1", "SVM-2", "SVM-3", "SVM-4", "SVM-5", "SVM-6"]
    methods = ["Preprocessing", "Filtering", "Multiclass Strategy", "Model Selection", "Cross Validation", "Accuracy", "Rank Kaggle"]
    results = []
    
    print("Evaluating SVM - 1 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,_ = write("normalization","univariance","svm",False,True,False,False)
    results.append(["normalization","univariance",False,"Randomized","Normal",accuracy,0])
    
    print("Evaluating SVM - 2 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,_ = write("normalization","univariance","svm",False,False,False,True)
    results.append(["normalization","univariance",False,False,"Calibrated",accuracy,0])
    
    print("Evaluating SVM - 3 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,_ = write("normalization","univariance","svm",True,False,False,False)
    results.append(["normalization","univariance","OneVsRest",False,"Normal",accuracy,0])
    
    print("Evaluating SVM - 4 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,_ = write("standardization","univariance","svm",False,False,False,True)
    results.append(["standardization","univariance",False,False,"Calibrated",accuracy,0])
    
    print("Evaluating SVM - 5 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,name = write("standardization","univariance","svm",False,False,True,False)
    results.append(["standardization","univariance",False,"Grid","Normal",accuracy,0])
    
    print("Evaluating SVM - 6 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,name = write("standardization","univariance","svm",True,False,False,False)
    results.append(["standardization","univariance","OneVsRest",False,"Normal",accuracy,0])
    
    print("Write Final Table ")
    print("      ")
    
    table_final = pd.DataFrame(results,index=algorithms,columns=methods)
        
    fp = open("Final_table.csv", 'w')
    fp.write(table_final.to_csv())


    
#evaluate()

def evaluate2():
    
    print("Evaluating Random Forest ")
    print("      ")
    
    #Random Forest - method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    
    algorithms = ["Random Forest-1", "Random Forest-2", "Random Forest-3", "Random Forest-4", "Random Forest-5", "Random Forest-6"]
    methods = ["Preprocessing", "Filtering", "Multiclass Strategy", "Model Selection", "Cross Validation", "Accuracy", "Rank Kaggle"]
    results = []
    table_final = pd.DataFrame(results,index=algorithms,columns=methods)
    
    print("Evaluating Random Forest - 1 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    
    accuracy,_ = write("normalization","univariance","random_forest",False,True,False,False)

    results.append(["normalization","univariance",False,"Randomized","Normal",accuracy,0])
    print(accuracy)
    
    print("Evaluating Random Forest - 2 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,_ = write("normalization","univariance","random_forest",False,False,False,True)
    results.append(["normalization","univariance",False,False,"Calibrated",accuracy,0])
    print(accuracy)    
    print("Evaluating Random Forest - 3 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,_ = write("normalization","univariance","random_forest",True,False,False,False)
    results.append(["normalization","univariance","OneVsRest",False,"Normal",accuracy,0])
    print(accuracy)    
    print("Evaluating Random Forest - 4 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,_ = write("standardization","univariance","random_forest",False,False,False,True)
    results.append(["standardization","univariance",False,False,"Calibrated",accuracy,0])
    print(accuracy)    
    print("Evaluating Random Forest - 5 ")
    print("      ")
    
    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
    accuracy,name = write("standardization","univariance","random_forest",False,False,True,False)
    results.append(["standardization","univariance",False,"Grid","Normal",accuracy,0])
    print(accuracy)    
#    print("Evaluating Random Forest - 6 ")
#    print("      ")
#    
#    #method_PP, method_F, method_A, OneVsRest, randomized, grid, calibrated
#    accuracy,name = write("standardization","univariance","random_forest",True,False,False,False)
#    results.append(["standardization","univariance","OneVsRest",False,"Normal",accuracy,0])
    
    print("Write Final Table ")
    print("      ")
        
    fp = open("Final_table.csv", 'w')
    fp.write(table_final.to_csv())


    
evaluate2()
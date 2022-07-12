# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import theano

np.random.seed(123)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

train = pd.read_csv("../input/train.csv")
latertrain = train.copy()
test = pd.read_csv("../input/test.csv")

def load_data():
    
    print("Loading data...")
    print("      ")
    
    X_train = train.loc[:, 'margin1':]
    y_train = train.pop('species')
    X_test = test.loc[:,'margin1':]
    y_train = LabelEncoder().fit(y_train).transform(y_train)

    print(y_train.shape)
    

    #le = LabelEncoder().fit(train['species'])
    #y_train = le.transform(train['species'])
    
    print("Data loaded")
    print("      ")

    return X_train, y_train, X_test

#normalization / standardization / binarization
def pre_processing(method_PP):
    
    X_train, y_train, X_test = load_data()
    
    print("Preprocessing data...")
    print("      ")
    
    if method_PP == "standardization":
        # standardize the data by setting the mean to 0 and std to 1
        X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
        X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
    
        print("by " + method_PP)
        print("      ")
        
    elif method_PP == "normalization":
        X_train = preprocessing.normalize(X_train)
        X_test = preprocessing.normalize(X_test)
        
        print("by " + method_PP)
        print("      ")
        
    elif method_PP == "binarization":
        X_train = preprocessing.Binarizer().fit(X_train)
        X_train.transform(X_train)
        X_test = preprocessing.Binarizer().fit(X_test)
        X_test.transform(X_test)
        
        print("by " + method_PP)
        print("      ")
        
    print("Data processed by " + method_PP)
    print("      ")
        
    return X_train, X_test, y_train

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
    

#svm / random_forest / neural_networks / logistic/ OneVsRest could be True or False / OneVsOne could beTrue or False /
#True or False (if last perform grid)
def algorithm(method_A, OneVsRest, OneVsOne, randomized):
        
    print("Selecting algorithm..." )
    print("      ")
    
    if method_A == "svm":
        
        print("Starting with " + method_A)
        print("      ")
        
        parameters_svm = {'kernel':('linear', 'rbf'), 'C':[1, 3, 10, 100],'gamma':[0.01, 0.001]}
        model = svm.SVC()
        model = search_par(randomized, model, parameters_svm)
    
    if method_A == "random_forest":
        
        print("Starting with " + method_A)
        print("      ")
        
        parameters_random = {"max_depth": [2, 3, None], "max_features": [2,4,6], "min_samples_split": [2,4,6], "min_samples_leaf": [2,4,6], "bootstrap": [True, False], "criterion": ["gini", "entropy"]}
        model = RandomForestClassifier(n_estimators = 100)
        model = search_par(randomized, model, parameters_random)
        
    if method_A == "logistic":
        
        print("Starting with " + method_A)
        print("      ")
        
        parameters_logistic = {'C':[100, 1000], 'tol': [0.001, 0.0001]}
        model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        model = search_par(randomized, model, parameters_logistic)
    
    if method_A == "neural_networks":
        
        print("Starting with " + method_A)
        print("      ")
        
        #model = MLPClassifier()
        
        model = Sequential()
        model.add(Dense(991, input_dim=179, init='normal')) # number of features of the data +1 node for the bias term.
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(495, init='normal')) #In sum, for most problems, one could probably get decent performance (even without a second optimization step) by setting the hidden layer configuration using just two rules: (i) number of hidden layers equals one; and (ii) the number of neurons in that layer is the mean of the neurons in the input and output layers.
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(99, init='normal'))# If the NN is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.
        model.add(Activation('softmax'))
        
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        
        OneVsRest = False
        OneVsOne = False
    
    if OneVsRest:
        
        print("Using OneVsRest ")
        print("      ")
        
        return OneVsRestClassifier(model)
    
    if OneVsOne:
        
        print("Using OneVsOne")
        print("      ")
        
        return OneVsOneClassifier(model)
        
        
    print("Algorithm selected: " + method_A)
    print("      ")
    
    return model

#True or False / model / parameters
def search_par(randomized, model, parameters):
    
    if randomized:
        
        print("Using randomized search")
        print("      ")
        
        return RandomizedSearchCV(model, param_distributions=parameters, n_iter=10)
    
    else:
        
        print("Using grid search")
        print("      ")
        
        return grid_search.GridSearchCV(model, parameters, scoring='log_loss', refit='True', n_jobs=-1, cv=5)
        
#normalization / standardization / binarization
#univariance / variance
#svm / random_forest / neural_networks / logistic/ OneVsRest could be True or False / OneVsOne could beTrue or False /        
#random / grid  
#calibrated true or false
def model_builder(method_PP, method_F, method_A, OneVsRest, OneVsOne, randomized, calibrated):
    
    X_train, X_test, y_train, ind = filtering(method_PP, method_F)
    
    model = algorithm(method_A, OneVsRest, OneVsOne, randomized)
    
    print("Building model..." )
    print("      ")
    
    if calibrated:
        
        print("Calibrating model..." )
        print("      ")
        
        model = CalibratedClassifierCV(model, method='sigmoid', cv=10)
        model.fit(X_train,y_train)
        accuracy = model.score(X_train, y_train)
        #print(accuracy)
        return model, accuracy, X_test, y_train, ind
    
    elif method_A == "neural_networks":
        
        print("Without model calibration (neural_networks)" )
        print("      ")

        y_traincat = to_categorical(y_train)
        model.fit(X_train, y_traincat,batch_size=128,nb_epoch=60)
        #accuracy = cross_validation.cross_val_score(model, X_train,y_traincat, cv = 10).mean()
        #print(accuracy)
        return model, X_test, y_train, ind
        
    else:
        
        print("Without model calibration" )
        print("      ")
        
        model = model.fit(X_train, y_train)
        accuracy = cross_validation.cross_val_score(model, X_train,y_train, cv = 10).mean()
        #print(accuracy)
        return model, accuracy, X_test, y_train, ind
    
    print("Model ready to predict ")
    print("      ")
    
    #return model, accuracy, X_test, y_train, ind

#normalization / standardization / binarization
#univariance / variance
#svm / random_forest / neural_networks / logistic/ OneVsRest could be True or False / OneVsOne could beTrue or False /        
#random / grid  
#calibrated true or false
def predict(method_PP, method_F, method_A, OneVsRest, OneVsOne, randomized, calibrated):
    #model, accuracy, X_test, y_train, ind = model_builder(method_PP, method_F, method_A, OneVsRest, OneVsOne, randomized, calibrated)
    model, X_test, y_train, ind = model_builder(method_PP, method_F, method_A, OneVsRest, OneVsOne, randomized, calibrated)
    
    print("Predicting... ")
    print("      ")
    
    X_test = X_test[:,ind]
    
    if calibrated:
        probs = model.predict_proba(X_test)
        
        
    else:
        if method_A == "logistic" or method_A == "random_forest" or OneVsRest or OneVsOne:
            
            print("Without probabilities calculation ")
            print("      ")
            
            probs = model.predict_proba(X_test)
            
        
        elif method_A == "neural_networks":
            
            print("Without probabilities calculation ")
            print("      ")
            
            probs = model.predict_proba(X_test, batch_size=128)
            #probs = model.predict_proba(X_test)
            
            
        else:
            predictions = model.predict(X_test)
            probs = prob(predictions, y_train)
    
    print("Predictions ready ")
    print("      ")
    
    return probs, y_train
    #, accuracy
            
def prob(predictions, y_train):
    
    print("Calculating probabilities... ")
    print("      ")
    
    ids = list(test.loc[:,"id"])
    species = list(np.unique(y_train))
    predictions_counts = np.zeros([len(ids),len(species)])
    lin = 0
    for pred in predictions:
        col = species.index(pred)
        predictions_counts[lin,col] += 1
        lin += 1
    for linha in range(len(predictions_counts)):
        soma = sum(predictions_counts[linha])
        for coluna in range(len(predictions_counts[linha])):
            predictions_counts[linha,coluna] = predictions_counts[linha,coluna]/soma
    
    print("Probabilities calculated ")
    print("      ")
    
    return predictions_counts
    
            
def write(method_PP, method_F, method_A, OneVsRest, OneVsOne, randomized, calibrated):
    #probs, y_train, accuracy = predict (method_PP, method_F, method_A, OneVsRest, OneVsOne, randomized, calibrated)
    probs, y_train = predict (method_PP, method_F, method_A, OneVsRest, OneVsOne, randomized, calibrated)
    ids = test.pop('id')
    y_train = latertrain.pop('species')
    species = list(np.unique(y_train))
    yPred = pd.DataFrame(probs,index=ids,columns=species)
    print('Creating and writing submission...')
    print("      ")
    name = method_PP + "_" + method_F + "_" + method_A + "_"
    if OneVsRest: name = name + "OneVsRest" + "_"
    if OneVsOne: name = name + "OneVsOne" + "_"
    if randomized: name = name + "randomized" + "_" 
    if calibrated: name = name + "calibrated"
    fp = open(name+".csv", 'w')
    fp.write(yPred.to_csv())
    #return accuracy, name
    return name

def evaluate():
    
    print("Evaluating SVM ")
    print("      ")
    
    #svm
    
    algorithms = ["SVM-1", "SVM-2", "SVM-3", "SVM-4", "SVM-5", "SVM-6"]
    methods = ["Preprocessing", "Filtering", "Multiclass Strategy", "Model Selection", "Cross Validation", "Accuracy", "Rank Kaggle"]
    results = []
    table_final = pd.DataFrame(results,index=algorithms,columns=methods)
    
#    print("Evaluating SVM - 1 ")
#    print("      ")
#    
#    accuracy,_ = write("standardization","univariance","svm",False,False,True,False)
#    results.append(["standardization","univariance",False,"Randomized","Normal",accuracy,0])
    
    print("Evaluating SVM - 2 ")
    print("      ")
    
    accuracy,_ = write("normalization","univariance","svm",False,False,True,False)
    results.append(["normalization","univariance",False,"Randomized","Normal",accuracy,0])
    
#    print("Evaluating SVM - 3 ")
#    print("      ")
#    
#    accuracy,_ = write("standardization","univariance","svm",False,True,False,True)
#    results.append(["standardization","univariance","OneVsOne","Grid","Calibrated",accuracy,0])
#    
#    print("Evaluating SVM - 4 ")
#    print("      ")
#    
#    accuracy,_ = write("normalization","univariance","svm",True,False,True,True)
#    results.append(["standardization","univariance","OneVsRest","Randomized","Calibrated",accuracy,0])
#    
#    print("Evaluating SVM - 5 ")
#    print("      ")
#    
#    accuracy,name = write("normalization","univariance","svm",False,False,True,True)
#    results.append(["standardization","univariance",False,"Randomized","Calibrated",accuracy,0])
#    
#    print("Evaluating SVM - 6 ")
#    print("      ")
#    
#    accuracy,name = write("standardization","univariance","svm",False,True,False,False)
#    results.append(["standardization","univariance","OneVsOne","Grid","Normal",accuracy,0])
#    
#    print("Write Final Table ")
#    print("      ")
#        
#    fp = open("Final_table.csv", 'w')
#    fp.write(table_final.to_csv())


write("standardization", "univariance", "neural_networks", False, False, False, False)
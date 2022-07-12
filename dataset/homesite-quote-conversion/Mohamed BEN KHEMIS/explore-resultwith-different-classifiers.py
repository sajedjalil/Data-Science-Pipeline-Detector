from sklearn.ensemble import *
from sklearn import cross_validation
from sklearn import metrics
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression

def run_classification(X,y,clfs,feature_names,show_wrong_classifications):
        n_samples = X.shape[0]
        kf = cross_validation.KFold(n_samples,n_folds)
        if show_wrong_classifications == 1 :
            print ("Printing wrong classifications...")
            print ("Features: ", feature_names)

        for clf in clfs:
            print ("\n************************ Doing classification with ", n_folds, " folds *********************************")
            print (clf)
            fold  = 1
            global_accuracy = 0
            global_accuracy2 = 0
            global_f1_score = 0
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train,y_train)
                predict = clf.predict(X_test)

                try : #probability of prediction
                    predict_probas = clf.predict_proba(X_test)
                except :
                    predict_probas = 'na'

                index = 0
                correct = 0
                mean_accuracy = clf.score(X_test, y_test)
                #print "fold: ", fold, " mean accuracy:", mean_accuracy
                for  label in y_test:
                        prediction = predict[index]

                        if predict_probas != 'na' :
                            proba = nice_list_printer(predict_probas[index])
                        else :
                            proba = 'na'
                        if label == prediction :
                                correct += 1

                        else :
                                if show_wrong_classifications == 1 :  #debug view
                                    failed_row = X_test[index]
                                    print("ERROR: features: ",nice_list_printer(failed_row), ", actual:", label, "pred:", prediction, ", proba:", proba, ", index: ", index)
                        index +=1
                accuracy = round( correct * 100.0 / index, 2)
                f1score = metrics.f1_score(y_test,predict)
                global_f1_score += f1score
                #print "Fold: ", fold, ", accuracy:", accuracy
                global_accuracy += accuracy
                global_accuracy2 += accuracy*accuracy
                fold += 1

            print ("\n\t Feature Importances:") # the features that drive the accuracy
            try :
                importance = clf.feature_importances_[0] # generate exception if this is not available
                i = 0
                for name in feature_names :
                    print ("\t\t"+name+" :", round(clf.feature_importances_[i],6))
                    i += 1
            except :
                # do nothing
                print ("\t\t n/a")

            avg_accuracy = round(global_accuracy * 1.0/ n_folds, 3)
            std_error = round( math.sqrt ( global_accuracy2 - global_accuracy*global_accuracy/ ( n_folds*1.0)  )
                          / (n_folds - 1.0) , 2)
            global_f1_score = round( global_f1_score / ( n_folds + 0.001 ), 3 )
            print ("\n       ==> Average accuracy: "+ str(avg_accuracy) + "%, std_error: "+ str(std_error) + "%, f1score:", global_f1_score)
            #print "*************************************************************************"


def nice_list_printer(list_in) :  # makes it look nicer for printing out
    formatted_string_list = map(lambda n: "%.2f" % n, list_in)
    res = '['
    for s in formatted_string_list:
        res += ' '+s
    return res + ' ]'



##### Now the main program ....

train = pd.read_csv("../input/train.csv")


print (train.head())
# use dummy encoding on categorical fields
numeric_fields = train._get_numeric_data().columns
print ("Numeric Fields:")
print (numeric_fields)
feature_names=train.columns
print ("feature_names:")
print (feature_names)
categoricals = feature_names - numeric_fields - ['QuoteConversion_Flag']
print ("categoricals:")
print (categoricals)

train = pd.get_dummies(train,categoricals,dummy_na=True)
train = train.fillna(0)
print ("Fields in train:" , len(train))
y= train.QuoteConversion_Flag
X = train.drop("QuoteConversion_Flag",axis=1)
X = X.T
X = X.values
show_wrong_classifications = False
n_estimators=50
n_folds=2

clfs = [LogisticRegression(penalty='L1', random_state=123),
        AdaBoostClassifier(n_estimators=n_estimators,random_state=123),
        GradientBoostingClassifier(n_estimators=n_estimators,random_state=123),
        ExtraTreesClassifier(n_estimators=n_estimators,random_state=123),
        RandomForestClassifier(n_estimators=n_estimators, random_state=123),

       ]

run_classification(X,y,clfs,feature_names,show_wrong_classifications)
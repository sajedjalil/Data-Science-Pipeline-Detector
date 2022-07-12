import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    
    loc_train = "../input/train.csv"
    loc_test = "../input/test.csv"
    loc_submission = "kaggle.et300.entropy.submission.csv"
    
    df_train = pd.read_csv(loc_train)
    df_test = pd.read_csv(loc_test)
    
    feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]
    
    X = df_train[feature_cols]
    X_submission = df_test[feature_cols]
    y = df_train['Cover_Type']
    test_ids = df_test['Id']
    del df_train
    del df_test
  
#    clf = ensemble.ExtraTreesClassifier(n_estimators=200,random_state=0)
#    clf.fit(X_train, y)
#    del X_train
    n_folds = 3
    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
    
    # print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    # print(X.shape)
    # print(y.shape)
    # print(X[1,2])
    for j, clf in enumerate(clfs):
        # print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            # print "Fold", i
            X_train = X.ix[train]
            y_train = y.ix[train]
            X_test = X.ix[test]
            y_test = y.ix[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict(X_test)
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict(X_submission)
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    
    # print
    # print "Blending."
    # clf = svm.SVC()
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test)
  
    with open(loc_submission, "w") as outfile:
        outfile.write("Id,Cover_Type\n")
        for e, val in enumerate(list(y_submission)):
            outfile.write("%s,%s\n"%(test_ids[e],val))


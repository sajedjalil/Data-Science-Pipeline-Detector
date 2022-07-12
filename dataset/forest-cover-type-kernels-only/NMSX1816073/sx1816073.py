import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB,  BernoulliNB
from sklearn.metrics import accuracy_score, log_loss,jaccard_similarity_score
import math
from sklearn.preprocessing import normalize

import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import logging
from heamy.dataset import Dataset
from heamy.estimator import Classifier
from heamy.pipeline import ModelsPipeline

DATA_DIR = "../input"
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
CACHE=False

NFOLDS = 5
SEED = 1337
METRIC = log_loss

ID = 'Id'
TARGET = 'Cover_Type'

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

np.random.seed(SEED)
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.WARNING)



def add_feats(df):
    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
    df['HF2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['HR1'] = (df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['FR1'] = (df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    df['EV1'] = df.Elevation+df.Vertical_Distance_To_Hydrology
    df['EV2'] = df.Elevation-df.Vertical_Distance_To_Hydrology
    df['Mean_HF1'] = df.HF1/2
    df['Mean_HF2'] = df.HF2/2
    df['Mean_HR1'] = df.HR1/2
    df['Mean_HR2'] = df.HR2/2
    df['Mean_FR1'] = df.FR1/2
    df['Mean_FR2'] = df.FR2/2
    df['Mean_EV1'] = df.EV1/2
    df['Mean_EV2'] = df.EV2/2    
    df['Elevation_Vertical'] = df['Elevation']+df['Vertical_Distance_To_Hydrology']    
    df['Neg_Elevation_Vertical'] = df['Elevation']-df['Vertical_Distance_To_Hydrology']
    
    # Given the horizontal & vertical distance to hydrology, 
    # it will be more intuitive to obtain the euclidean distance: sqrt{(verticaldistance)^2 + (horizontaldistance)^2}    
    df['slope_hyd_sqrt'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
    df.slope_hyd_sqrt=df.slope_hyd_sqrt.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
    
    df['slope_hyd2'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)
    df.slope_hyd2=df.slope_hyd2.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
    
    #Mean distance to Amenities 
    df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 
    #Mean Distance to Fire and Water 
    df['Mean_Fire_Hyd1']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2
    df['Mean_Fire_Hyd2']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Roadways) / 2
    
    #Shadiness
    df['Shadiness_morn_noon'] = df.Hillshade_9am/(df.Hillshade_Noon+1)
    df['Shadiness_noon_3pm'] = df.Hillshade_Noon/(df.Hillshade_3pm+1)
    df['Shadiness_morn_3'] = df.Hillshade_9am/(df.Hillshade_3pm+1)
    df['Shadiness_morn_avg'] = (df.Hillshade_9am+df.Hillshade_Noon)/2
    df['Shadiness_afternoon'] = (df.Hillshade_Noon+df.Hillshade_3pm)/2
    df['Shadiness_mean_hillshade'] =  (df['Hillshade_9am']  + df['Hillshade_Noon'] + df['Hillshade_3pm'] ) / 3    
    
    # Shade Difference
    df["Hillshade-9_Noon_diff"] = df["Hillshade_9am"] - df["Hillshade_Noon"]
    df["Hillshade-noon_3pm_diff"] = df["Hillshade_Noon"] - df["Hillshade_3pm"]
    df["Hillshade-9am_3pm_diff"] = df["Hillshade_9am"] - df["Hillshade_3pm"]

    # Mountain Trees
    df["Slope*Elevation"] = df["Slope"] * df["Elevation"]
    # Only some trees can grow on steep montain
    
    ### More features
    df['Neg_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['Neg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['Neg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    
    df['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])/2
    df['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])/2
    df['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])/2   
        
    df["Vertical_Distance_To_Hydrology"] = abs(df['Vertical_Distance_To_Hydrology'])
    
    df['Neg_Elev_Hyd'] = df.Elevation-df.Horizontal_Distance_To_Hydrology*0.2
    
    # Bin Features
    bin_defs = [
        # col name, bin size, new name
        ('Elevation', 200, 'Binned_Elevation'), # Elevation is different in train vs. test!?
        ('Aspect', 45, 'Binned_Aspect'),
        ('Slope', 6, 'Binned_Slope'),
        ('Horizontal_Distance_To_Hydrology', 140, 'Binned_Horizontal_Distance_To_Hydrology'),
        ('Horizontal_Distance_To_Roadways', 712, 'Binned_Horizontal_Distance_To_Roadways'),
        ('Hillshade_9am', 32, 'Binned_Hillshade_9am'),
        ('Hillshade_Noon', 32, 'Binned_Hillshade_Noon'),
        ('Hillshade_3pm', 32, 'Binned_Hillshade_3pm'),
        ('Horizontal_Distance_To_Fire_Points', 717, 'Binned_Horizontal_Distance_To_Fire_Points')
    ]
    
    for col_name, bin_size, new_name in bin_defs:
        df[new_name] = np.floor(df[col_name]/bin_size)
        
    print('Total number of features : %d' % (df.shape)[1])
    return df
    
    
    def load_and_process_dataset():
    train = pd.read_csv("{0}/train.csv".format(DATA_DIR))
    test = pd.read_csv("{0}/test.csv".format(DATA_DIR))

    y_train = train[TARGET].ravel() -1 # XGB needs labels starting with 0!
    
    classes = train.Cover_Type.unique()
    num_classes = len(classes)
    print("There are %i classes: %s " % (num_classes, classes))        

    train.drop([ID, TARGET], axis=1, inplace=True)
    test.drop([ID], axis=1, inplace=True)
    
    train = add_feats(train)    
    test = add_feats(test)    
    
    cols_to_normalize = [ 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                       'Horizontal_Distance_To_Fire_Points', 
                       'Shadiness_morn_noon', 'Shadiness_noon_3pm', 'Shadiness_morn_3',
                       'Shadiness_morn_avg', 
                       'Shadiness_afternoon', 
                       'Shadiness_mean_hillshade',
                       'HF1', 'HF2', 
                       'HR1', 'HR2', 
                       'FR1', 'FR2'
                       ]

    train[cols_to_normalize] = normalize(train[cols_to_normalize])
    test[cols_to_normalize] = normalize(test[cols_to_normalize])

    # elevation was found to have very different distributions on test and training sets
    # lets just drop it for now to see if we can implememnt a more robust classifier!
    train = train.drop('Elevation', axis=1)
    test = test.drop('Elevation', axis=1)    
    
    x_train = train.values
    x_test = test.values

    return {'X_train': x_train, 'X_test': x_test, 'y_train': y_train}
    
    dataset = Dataset(preprocessor=load_and_process_dataset, use_cache=True)  
    
    
    # Parameters for the classifiers
rf_params = {
    'n_estimators': 200,
    'criterion': 'entropy',
    'random_state': 0
}

rf1_params = {
    'n_estimators': 200,
    'criterion': 'gini',
    'random_state': 0
}

et1_params = {
    'n_estimators': 200,
    'criterion': 'gini',
    'random_state': 0
}

et_params = {
    'n_estimators': 200,
    'criterion': 'entropy',
    'random_state': 0
}

et1_params = {
    'n_estimators': 200,
    'criterion': 'gini',
    'random_state': 0
}

lgb_params = {
    'n_estimators': 200, 
    'learning_rate':0.1
}

logr_params = {
        'solver' : 'liblinear',
        'multi_class' : 'ovr',
        'C': 1,
        'random_state': 0}
        
        
rf = Classifier(dataset=dataset, estimator = RandomForestClassifier, use_cache=CACHE, parameters=rf_params,name='rf')
et = Classifier(dataset=dataset, estimator = ExtraTreesClassifier, use_cache=CACHE, parameters=et_params,name='et')   
rf1 = Classifier(dataset=dataset, estimator=RandomForestClassifier, use_cache=CACHE, parameters=rf1_params,name='rf1')
et1 = Classifier(dataset=dataset, use_cache=CACHE, estimator=ExtraTreesClassifier, parameters=et1_params,name='et1')
lgbc = Classifier(dataset=dataset, estimator=LGBMClassifier, use_cache=CACHE, parameters=lgb_params,name='lgbc')
gnb = Classifier(dataset=dataset,estimator=GaussianNB, use_cache=CACHE, name='gnb')
logr = Classifier(dataset=dataset, estimator=LogisticRegression, use_cache=CACHE, parameters=logr_params,name='logr')



def xgb_first(X_train, y_train, X_test, y_test=None):
    xg_params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',   
        'num_class': 7,
        'max_depth': 4,
        'min_child_weight': 1,
        'eval_metric': 'mlogloss',
        'nrounds': 200
    }    
    X_train = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(xg_params, X_train, xg_params['nrounds'])
    return model.predict(xgb.DMatrix(X_test))
    
    
    
xgb_first = Classifier(estimator=xgb_first, dataset=dataset, use_cache=CACHE, name='xgb_first')  


pipeline = ModelsPipeline(rf, et, et1, lgbc, logr, gnb, xgb_first) 

stack_ds = pipeline.stack(k=NFOLDS,seed=SEED)


# Train LogisticRegression on stacked data (second stage)
lr = LogisticRegression
lr_params = {'C': 5, 'random_state' : SEED, 'solver' : 'liblinear', 'multi_class' : 'ovr',}
stacker = Classifier(dataset=stack_ds, estimator=lr, use_cache=False, parameters=lr_params)


# Validate results using k-fold cross-validation
results = stacker.validate(k=NFOLDS,scorer=log_loss)


models = [rf, et, et1, lgbc, logr, gnb, xgb_first]       
print("Log Loss")
for index, element in enumerate(models):
    print(index, element.name)
    element.validate(k=NFOLDS,scorer=log_loss)
    
    
    
preds_proba = stacker.predict() 
# Note: labels starting with 0 in xgboost, therefore adding +1!
predictions = np.round(np.argmax(preds_proba, axis=1)).astype(int) + 1


submission = pd.read_csv(SUBMISSION_FILE)
submission[TARGET] = predictions
submission.to_csv('Stacking_with_heamy_logregr.sub.csv', index=None)


# Use a xgb-model as 2nd-stage model

dtrain = xgb.DMatrix(stack_ds.X_train, label=stack_ds.y_train)
dtest = xgb.DMatrix(stack_ds.X_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.05,
    'objective': 'multi:softprob',
    'num_class': 7,        
    'max_depth': 6,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mlogloss',
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, 
             nfold=NFOLDS, seed=SEED, stratified=True,
             early_stopping_rounds=20, verbose_eval=5, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 2]
cv_std = res.iloc[-1, 3]

print('Ensemble-CV: {0}+{1}, best nrounds = {2}'.format(cv_mean, cv_std, best_nrounds))


# Train with best rounds
model = xgb.train(xgb_params, dtrain, best_nrounds)

xpreds_proba = model.predict(dtest)

# Note: labels starting with 0 in xgboost, therefore adding +1!
predictions = np.round(np.argmax(xpreds_proba, axis=1)).astype(int) + 1

submission = pd.read_csv(SUBMISSION_FILE)
submission[TARGET] = predictions
submission.to_csv('Stacking_with_heamy_cv_mlogloss_' + str(cv_mean) + '.sub.csv', index=None)
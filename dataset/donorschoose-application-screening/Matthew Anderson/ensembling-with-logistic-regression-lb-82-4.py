import pandas as pd
import numpy  as np
import os
from scipy.special import expit, logit
 
almost_zero = 1e-5 # 0.00001
almost_one  = 1-almost_zero # 0.99999

scores = {}
positive = {}
# Random Forest with Bayesian Optimization Hyperparameter Tuning and Features 
#df = pd.read_csv("../input/deep-learning-is-all-you-need-lb-0-80x/gru_cnn_submission.csv",index_col="id").rename(columns={'project_is_approved': 'GRNN + CNN'}) # 0.80121
#scores["GRNN + CNN"] = 0.80121
#weights["GRNN + CNN"] = 1

# Averaged due to their model similarity:
# Naive Bayes with Logistic Regression (word vectors)
df = pd.read_csv("../input/naive-bayes-svm-on-vocabulary/resourcesummary_output.csv",index_col="id").rename(columns={'project_is_approved': 'NBSVM'})
scores["NBSVM"] = 0.73952
positive["NBSVM"] = 0.5
# Word n-grams and Logistic Regression
df["NLP + LR"] = pd.read_csv("../input/logistic-regression-with-word-n-grams/submission.csv")['project_is_approved'].values
scores["NLP + LR"] = 0.73617
positive["NLP + LR"] = 0.5

# LightGBM
df["LightGBM"] = ( ( pd.read_csv("../input/xtra-credit-xgb-lgb-tfidf-feature-stacking/lgbm_submission.csv")['project_is_approved'].values ) + ( pd.read_csv("../input/abc-s-of-tf-idf-boosting-0-798/lgbm_sub.csv")['project_is_approved'].values ) ) / 2 
scores["LightGBM"] = 0.77583
positive["LightGBM"] = 1
# XGBoost
df["XGBoost"] = ( ( pd.read_csv("../input/xtra-credit-xgb-lgb-tfidf-feature-stacking/xgb_submission.csv")['project_is_approved'].values ) + ( pd.read_csv("../input/abc-s-of-tf-idf-boosting-0-798/xgb_sub.csv")['project_is_approved'].values ) ) / 2 
scores["XGBoost"] = 0.79745
positive["XGBoost"] = 1

# Gated Reccurrent Unit + ATT with LGBM, TFIDF
df["GRU-ATT + LightGBM"] = pd.read_csv("../input/how-to-get-81-gru-att-lgbm-tf-idf-eda/submission.csv")['project_is_approved'].values
scores["GRU-ATT + LightGBM"] = 0.81177
positive["GRU-ATT + LightGBM"] = 2

# LightGBM and Neural Network
df["LightGBM + NN"] = pd.read_csv("../input/beginners-workflow-meanencoding-lgb-nn-ensemble/submission_with_lgb_bool.csv")['project_is_approved'].values
scores["LightGBM + NN"] = 0.81544
positive["LightGBM + NN"] = 2

# Capsule Networks
df["capsnet"] = pd.read_csv("../input/beginner-s-guide-to-capsule-networks/submission.csv")['project_is_approved'].values
scores["capsnet"] = 0.79590
positive["capsnet"] = 1

# All-In-One
df["aio"] = pd.read_csv("../input/the-all-in-one-model/submission.csv")['project_is_approved'].values
scores["aio"] = 0.80765
positive["aio"] = 2

# XGB-LGB with Feature Engineering
df["UFE"] = pd.read_csv("../input/ultimate-feature-engineering-xgb-lgb-nn/text_cat_num_xgb_lgb_NN.csv")['project_is_approved'].values
scores["UFE"] = 0.813
positive["UFE"] = 2

# Logistic Regression
df["LR"] = pd.read_csv("../input/mastering-the-basics-80-using-scikit-learn-v2-0/submission.csv")['project_is_approved'].values
scores["LR"] = 0.79759
positive["LR"] = 1

# GRU + CNN
df["GRU + CNN"] = pd.read_csv("../input/deep-learning-is-all-you-need-lb-0-80x/gru_cnn_submission.csv")['project_is_approved'].values
scores["GRU + CNN"] = 0.80121
positive["GRU + CNN"] = 2

power = 68
weights = [0] * len(df.columns)
dic = {}

inversescores = scores
for i in scores:
    inversescores[i] = 1 - scores[i]

p = 1
for i in inversescores:
    p *= inversescores[i]

for i,col in enumerate(df.columns):
    weights[i] = p * (1-scores[col]) * positive[col]
    dic[i] = df[col].clip(almost_zero,almost_one).apply(logit) * weights[i] 
    
print(weights[:])
totalweight = sum(weights)

temp = []
for x in dic:
    if x == 0:
        temp = dic[x]
    else:
        temp = temp+dic[x]

temp = temp/(totalweight)

df["project_is_approved"] = temp
df["project_is_approved"] = df["project_is_approved"].apply(expit)
df["project_is_approved"].to_csv("ensembling_submission_50.csv", index=True, header=True)
print(df["project_is_approved"].head())
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Originally-originaly-originally
# many time originally it is fork from
# https://www.kaggle.com/matthewa313/ensembling-with-logistic-regression-lb-81-7

import pandas as pd
import numpy  as np
from scipy.special import expit, logit
 
almost_zero = 1e-5 # 0.00001
almost_one  = 1-almost_zero # 0.99999

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

scores = {}

# Gated Recurrent Unit + Convolutional Neural Network
df = pd.read_csv("../input/deep-learning-is-all-you-need-lb-0-80x/gru_cnn_submission.csv",index_col="id").rename(columns={'project_is_approved': 'GRNN + CNN'}) # 0.80121
scores["GRNN + CNN"] = 0.73  #0.80121

# "The Choice is Yours"
# df["choice"] = pd.read_csv("../input/the-choice-is-yours/blend_submission.csv")['project_is_approved'].values # 0.793
# scores["choice"] = 0.793

# LightGBM and TFIDF
df["Lgbm_tf"] = pd.read_csv("../input/lightgbm-and-tf-idf-starter/submission.csv")['project_is_approved'].values # 0.7947
scores["Lgbm_tf"] = 0.77 #0.79

# NLP
df["NLP"] = pd.read_csv("../input/a-pure-nlp-approach/submission.csv")['project_is_approved'].values # 0.7959
scores["NLP"] = 0.81 #0.7959

# Gated Reccurrent Unit + ATT with LGBM, TFIDF + EDA
df["grnn_Att_lgbm"] = pd.read_csv("../input/how-to-get-81-gru-att-lgbm-tf-idf-eda/submission.csv")['project_is_approved'].values # 0.81177
scores["grnn_Att_lgbm"] = 0.81177

# Capsule Networks
df["capnet"] = pd.read_csv("../input/beginner-s-guide-to-capsule-networks/submission.csv")['project_is_approved'].values # 0.79590
scores["capnet"] = 0.79590

# ABC's
df["ABC_LGBM"] = pd.read_csv("../input/abc-s-of-tf-idf-boosting-0-798/lgbm_sub.csv")['project_is_approved'].values # 0.79745
scores["ABC_LGBM"] =0.79

# ABC's
df["ABC_XGB"] = pd.read_csv("../input/abc-s-of-tf-idf-boosting-0-798/xgb_sub.csv")['project_is_approved'].values # 0.79745
scores["ABC_XGB"] = 0.79

# All-In-One
df["power"] = pd.read_csv("../input/the-all-in-one-model/submission.csv")['project_is_approved'].values # 0.81829
scores["power"] = 0.815 #0.81829

# PoWER 64 #best
df["UI"] = pd.read_csv("../input/ultimate-feature-engineering-xgb-lgb-lb-0-813/text_cat_num_xgb_lgb.csv")['project_is_approved'].values # 0.81829
scores["UI"] = 0.825

# More NN..
# Add https://www.kaggle.com/emotionevil/nlp-and-stacking-starter-dpcnn-lgb-lb0-80/notebook

weights = [0] * len(df.columns)
power = 68
dic = {}

for i,col in enumerate(df.columns):
    weights[i] = scores[col] ** power
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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy  as np
from   scipy.special import expit, logit
 
almost_zero = 1e-10
almost_one  = 1-almost_zero

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# "The Choice is Yours"
df1 = pd.read_csv("../input/the-choice-is-yours/blend_submission.csv").rename(columns={'project_is_approved': 'project_is_approved1'}) # 0.793

# Gated Recurrent Unit + Convolutional Neural Network
df2 = pd.read_csv("../input/deep-learning-is-all-you-need-lb-0-80x/gru_cnn_submission.csv").rename(columns={'project_is_approved': 'project_is_approved2'}) # 0.80121

# LightGBM + XGBoost Blended
df3 = pd.read_csv("../input/understanding-approval-donorschoose-eda-fe-eli5/lgbm_xgb_blend.csv").rename(columns={'project_is_approved': 'project_is_approved3'}) # 0.78583

# LightGBM and TFIDF
df4 = pd.read_csv("../input/lightgbm-and-tf-idf-starter/submission.csv").rename(columns={'project_is_approved': 'project_is_approved4'}) # 0.7947

# NLP
df5 = pd.read_csv("../input/a-pure-nlp-approach/submission.csv").rename(columns={'project_is_approved': 'project_is_approved5'}) # 0.7959

# Gated Reccurrent Unit + ATT with LGBM, TFIDF + EDA
df6 = pd.read_csv("../input/how-to-get-81-gru-att-lgbm-tf-idf-eda/submission.csv").rename(columns={'project_is_approved': 'project_is_approved6'}) # 0.81177

# Zahar
df7 = pd.read_csv("../input/beginner-s-guide-to-capsule-networks/submission.csv").rename(columns={'project_is_approved': 'project_is_approved7'}) # 0.79590


# These other kernels made the model less accurate
# this could be because the model wasn't that good in the first place or...
# it could be that we had too many of that model (too many LGB / XGB / GLMnet)

# ../input/so-many-possibilities-nlp-and-stacking-starter/submission.csv
# ../input/so-many-possibilities-nlp-and-stacking-starter/submission_without_ftrl.csv
# ../input/how-to-perform-rank-averaging/rank_averaged_submission.csv
# ../input/eda-fe-xgb-glm-donors-choose/GLMNetMarch62018.csv
# ../input/eda-fe-xgb-glm-donors-choose/XGBMarch32018.csv
# ../input/eda-fe-xgb-glm-donors-choose/XGB_GLMNet_March62018.csv
# ../input/beginner-s-guide-to-capsule-networks/submission.csv

df = pd.merge(df1, df2, on='id')
df = pd.merge(df, df3, on='id')
df = pd.merge(df, df4, on='id')
df = pd.merge(df, df5, on='id')
df = pd.merge(df, df6, on='id')
df = pd.merge(df, df7, on='id')

scores = [0, 0.793, 0.80121, 0.78583, 0.7947, 0.7959, 0.81177, 0.7959] # public leaderboard scores

weights = [0, 0, 0, 0, 0, 0, 0, 0]

weights[1] = scores[1] ** 4
weights[2] = scores[2] ** 4
weights[3] = scores[3] ** 4
weights[4] = scores[4] ** 4
weights[5] = scores[5] ** 4
weights[6] = scores[6] ** 4
weights[7] = scores[7] ** 4

print(weights[:])

number1 = df['project_is_approved1'].clip(almost_zero,almost_one).apply(logit)  * weights[1]
number2 = df['project_is_approved2'].clip(almost_zero,almost_one).apply(logit)  * weights[2]
number3 = df['project_is_approved3'].clip(almost_zero,almost_one).apply(logit)  * weights[3]
number4 = df['project_is_approved4'].clip(almost_zero,almost_one).apply(logit)  * weights[4]
number5 = df['project_is_approved5'].clip(almost_zero,almost_one).apply(logit)  * weights[5]
number6 = df['project_is_approved6'].clip(almost_zero,almost_one).apply(logit)  * weights[6]
number7 = df['project_is_approved7'].clip(almost_zero,almost_one).apply(logit)  * weights[7]

totalweight = sum(weights)

df['project_is_approved'] = ( number1 + number2 + number3 + number4 + number5 + number6 + number7) / ( totalweight )

df['project_is_approved']  = df['project_is_approved'].apply(expit) 

# Any results you write to the current directory are saved as output.
df[['id', 'project_is_approved']].to_csv("ensembling_submission_ya_02.csv", index=False)


# What should be next for our kernel?
# Make it into a notebook?
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

blend_all = pd.read_csv("../input/fork-of-blend-it-al-ensamble/lazy_ensemble_submission_on_blend_sources.csv")
glove_and_fasttext = pd.read_csv("../input/glove-and-fasttext-blender/blend.csv")

blend = blend_all.copy()

blend[categories] = 0.67*blend_all[categories].values +0.33*glove_and_fasttext[categories].values

blend.to_csv("superblend.csv", index=False)
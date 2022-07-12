# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


#This kernel follows this one: 
#               https://www.kaggle.com/lesibius/crime-scene-exploration/editnb
# I initially intended to create the output directly in the exploration kernel
# but it turned out that the output file was too heavy.

# This one is basically the null model: take the average for each crime category and
# use it as the prediction

# Get data
df_train = pd.read_csv("../input/train.csv",index_col=None)
df_test = pd.read_csv("../input/test.csv",index_col=None)


number_of_crimes = df_train.Category.value_counts()
avg_crimes = number_of_crimes/sum(number_of_crimes)

model_null = df_test[["Id"]]
for crime in avg_crimes.index:
    model_null[crime] = round(100*avg_crimes[crime])/100

model_null.to_csv('null_submission.csv',index=False)



# Any results you write to the current directory are saved as output.
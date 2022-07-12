# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Based on : https://www.kaggle.com/vpaslay/lb-0-287-porto-seguro-mix

# submission_1 https://www.kaggle.com/aharless/exclude-same-wk-res-from-nitin-s-surpriseme2-w-nn
# submission_2 https://www.kaggle.com/meli19/surprise-me-h2o-automl-version-ver5-lb-0-479
# submission_3 https://www.kaggle.com/nitinsurya/surprise-me-2-neural-networks-keras
# https://www.kaggle.com/tejasrinivas/surprise-me-4-lb-0-479

# PLEASE Think about the overfitting problem !!!



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

submissions_path = "../input/PublicSubMissionFiles"
all_files = os.listdir(submissions_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(submissions_path, f), index_col=0)\
        for f in all_files]
concat_df = pd.concat(outs, axis=1)
cols = list(map(lambda x: "visitors_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

# Apply ranking, normalization and averaging
concat_df["visitors"] = (concat_df.rank() / concat_df.shape[0]).mean(axis=1)
concat_df.drop(cols, axis=1, inplace=True)

# Write the output
concat_df.to_csv("./ensemble.csv")
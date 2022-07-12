# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# All submission files were downloaded from different public kernels
# See the description to see the source of each submission file
submissions_path = "../input/lcfr-blend"
all_files = os.listdir(submissions_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(submissions_path, f), index_col=0)\
        for f in all_files]
concat_df = pd.concat(outs, axis=1)
cols = list(map(lambda x: "TARGET_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

# Apply ranking, normalization and averaging
concat_df["TARGET"] = (concat_df.rank() / concat_df.shape[0]).mean(axis=1)
concat_df.drop(cols, axis=1, inplace=True)

# Write the output
concat_df.to_csv("./LCFR_MIX.csv")
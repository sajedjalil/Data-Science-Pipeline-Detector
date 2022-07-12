ALPHA = 5e-3  # Regularizaiton parameter

import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.special import expit # sigmoid

# All submission files were downloaded from different public kernels
# See the description to see the source of each submission file
submissions_path = "../input/kaggleportosegurosubmissions"
all_files = os.listdir(submissions_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(submissions_path, f), index_col=0)\
        for f in all_files]
concat_df = pd.concat(outs, axis=1)
cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

# Apply log-odds transformation, regularize, and take standardized mean
logits = concat_df.applymap(lambda x: np.log(x/(1-x)))
logits *= (1 - ALPHA*logits**2)
stdevs = logits.std()  
w = .2/stdevs
wa = (w*logits).sum(axis=1)/w.sum()

# Convert back to probabilities
result = wa.apply(expit)
print( result.head() )

# Save result
pd.DataFrame(result,columns=['target']).to_csv("logitmix.csv",float_format='%.6f')
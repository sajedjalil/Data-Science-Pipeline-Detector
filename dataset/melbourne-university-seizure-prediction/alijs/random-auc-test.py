import numpy as np
import pandas as pd

print("Started")
np.random.seed(2017)
sub = pd.read_csv('../input/sample_submission.csv')
sub['Class'] = np.random.rand(sub.shape[0], 1)
sub['Class'] = sub['Class'] * (400 / sub['Class'].sum())
sub.to_csv('random.csv', index=False)
print(sub.head())
print("Done")

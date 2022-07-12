import numpy as np 
import pandas as pd

print("Started")
np.random.seed(2016)
def r(x):
    return "1 {} 2 {} 3 {} 4 {} 5 {}".format(np.random.random_sample(), np.random.random_sample(), np.random.random_sample(), np.random.random_sample(), np.random.random_sample())

s = pd.read_csv('../input/sample_submission.csv')
print(s.head(1))
s['LabelConfidencePairs'] = s['LabelConfidencePairs'].apply(r)
print(s.head(1))
s.to_csv('s.csv', index=False)
print("Done.")
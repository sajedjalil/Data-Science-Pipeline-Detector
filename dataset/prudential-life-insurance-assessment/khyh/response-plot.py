import pandas as pd 
import numpy as np
import matplotlib.pylab as plt

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

#print(train.shape)
#print(train.head(n=5))
#print(test.shape)
#print(test.head(n=5))

resp = np.array(train['Response'])

plt.hist(resp, bins=np.linspace(1,8,9))
plt.savefig('fig.png')
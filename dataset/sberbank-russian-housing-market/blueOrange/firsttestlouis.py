import pandas as pd
import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()

print ('Hello Louis')

train_df = pd.read_csv('../input/train.csv')
print (train_df.shape)


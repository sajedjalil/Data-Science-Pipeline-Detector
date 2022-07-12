import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../input/training.csv')

for n in range(2):
  fig = plt.figure(figsize=(35,20))

  for i in range(1,26):
    ax = fig.add_subplot(5, 5, i)
    col = train.columns[i + 25*n]
    ax.set_title(col)

    plt.hist([train[train['signal'] == 1][col], train[train['signal'] == 0][col]], bins=50, histtype='stepfilled', color=['r', 'b'], alpha=0.5, label=['signal', 'background'])
    
    if (i == 5): ax.legend()
        
  fig.tight_layout(pad=1, w_pad=1, h_pad=1)
  fig.savefig('hist'+str(n+1)+'.png', dpi=150)

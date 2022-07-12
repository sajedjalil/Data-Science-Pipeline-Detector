import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train = pd.read_csv("../input/train_1.csv")
train = train.fillna(0.)


# I'm gong to share a solution that I found interesting with you.
# The idea is to compute the median of the series in different window sizes at the end of the series,
# and the window sizes are increasing exponentially with the base of golden ratio.
# Then a median of these medians is taken as the estimate for the next 60 days.
# This code's result has the score of around 44.9 on public leaderboard, but I could get upto 44.7 by playing with it.

# r = 1.61803398875
# Windows = np.round(r**np.arange(0,9) * 7)
Windows = [6, 12, 18, 30, 48, 78, 126, 203, 329]


n = train.shape[1] - 1 #  550
Visits = np.zeros(train.shape[0])
for i, row in train.iterrows():
    M = []
    start = row[1:].nonzero()[0]
    if len(start) == 0:
        continue
    if n - start[0] < Windows[0]:
        Visits[i] = row.iloc[start[0]+1:].median()
        continue
    for W in Windows:
        if W > n-start[0]:
            break
        M.append(row.iloc[-W:].median())
    Visits[i] = np.median(M)

Visits[np.where(Visits < 1)] = 0.
train['Visits'] = Visits


test = pd.read_csv("../input/key_1.csv")
test['Page'] = test.Page.apply(lambda x: x[:-11])

test = test.merge(train[['Page','Visits']], on='Page', how='left')
test[['Id','Visits']].to_csv('sub.csv', index=False)
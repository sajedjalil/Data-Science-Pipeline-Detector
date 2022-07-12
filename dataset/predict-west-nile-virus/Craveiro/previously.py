import pandas as pd
import os

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")

for trap in set(train.Trap):
    dates = train[train.Trap == trap].Date
    for date in dates:
        print('trap:', trap, 'date:', date, 'check:',train[train.Trap == trap][train.Date == date].WnvPresent)


        
import pandas as pd
from random import randint
import random
sub = pd.read_csv('../input/sample_submission.csv')
def rand(day=[4,5,3,2,1]):
    res = ["4","5"]
    while len(res)<6:
        r =str(random.choice(day))
        if r not in res:
            res.append(r)
    return ' '.join([str(r) for r in  res ])
#sub['day'] = "4 5 3 2 1"
sub['day'] = rand(day="4 5 3 2 1")
sub.to_csv('naive_submit.csv', index=False)
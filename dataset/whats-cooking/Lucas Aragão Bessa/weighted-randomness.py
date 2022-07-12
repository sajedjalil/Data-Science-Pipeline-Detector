# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import random
from bisect import bisect

def weighted_choice(choices):
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)
    return values[i]

weighted_cuisines = [("italian",20), ("mexican",16), ("southern_us",11),
                     ("indian",8), ("chinese",7), ("french",7),
                     ("cajun_creole",4), ("thai",4), ("japanese",4),
                     ("greek",3), ("spanish",2), ("korean",2),
                     ("vietnamese",2), ("moroccan",2), ("british",2),
                     ("filipino",2), ("irish",2), ("jamaican",1),
                     ("russian",1), ("brazilian",1)]
                     

test = pd.read_json('../input/test.json')
test = test.drop('ingredients', axis = 1)

test['cuisine'] = test['id'].apply(lambda x: weighted_choice(weighted_cuisines))

test.to_csv('submission.csv',index = False)

# Any results you write to the current directory are saved as output.
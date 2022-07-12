# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import sklearn.feature_extraction.text as t
import scipy.spatial.distance as cos


te = pd.read_csv('../input/test.csv')
te = te.fillna("not a real entry")

u = t.CountVectorizer(lowercase = True, analyzer = "word")

print('vectorizer initialized')

def preds(dat):
    cur = dat[["question1","question2"]]
    if(len(cur[0]) <= 2 | len(cur[1]) <= 2):
        pred = 0
    else:    
        w = (pd.DataFrame(u.fit_transform(cur
        .tolist())
        .A,
        columns = u.get_feature_names()))
        pred = 1-cos.cosine(w.iloc[0], w.iloc[1])
    
    print(dat['test_id'])
    return pred
    

sub = pd.DataFrame(list(range(500)))
sub.columns = ['test_id']
is_duplicate = te.loc[range(500)].apply(preds,axis = 1)
sub["is_duplicate"] = is_duplicate

sub.to_csv('submission.csv', index = False)
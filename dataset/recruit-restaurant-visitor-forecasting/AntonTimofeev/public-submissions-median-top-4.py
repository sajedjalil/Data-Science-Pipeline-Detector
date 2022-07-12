# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# submission_1 https://www.kaggle.com/aharless/exclude-same-wk-res-from-nitin-s-surpriseme2-w-nn
# submission_2 https://www.kaggle.com/meli19/surprise-me-h2o-automl-version-ver5-lb-0-479
# submission_3 https://www.kaggle.com/nitinsurya/surprise-me-2-neural-networks-keras
# submission_4 https://www.kaggle.com/tejasrinivas/surprise-me-4-lb-0-479
# submission_5 https://www.kaggle.com/yixinsunn/simple-ridge-regression-lb-0-497

# PLEASE think about the overfitting problem !!! take your own risk of using this kernel.



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sub1 = pd.read_csv('../input/pubsubrrvf/submission_1.csv')
sub2 = pd.read_csv('../input/pubsubrrvf/submission_2.csv')
sub3 = pd.read_csv('../input/pubsubrrvf/submission_3.csv')
sub4 = pd.read_csv('../input/pubsubrrvf/submission_4.csv')
sub5 = pd.read_csv('../input/simple-ridge-regression-lb-0-497/submission.csv')

sub = pd.DataFrame({'sub1':sub1.visitors,
                    'sub2':sub2.visitors,
                    'sub3':sub3.visitors,
                    'sub4':sub4.visitors,
                    'sub5':sub5.visitors})

sub8 = pd.DataFrame()
sub8['id'] = sub1['id']
sub8['visitors'] = sub.median(axis=1)

sub8.to_csv('SubmissonK.csv',index=False)
# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import pandas as pd 

sub = pd.read_csv('../input/sample_submission.csv')
sub['Demanda_uni_equil'] = 3.88

sub.to_csv('submission.csv', index=False)

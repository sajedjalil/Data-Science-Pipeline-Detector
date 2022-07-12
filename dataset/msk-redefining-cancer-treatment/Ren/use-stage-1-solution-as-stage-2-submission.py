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
submission = pd.read_csv('../input/stage2_sample_submission.csv')
stage1_test = pd.read_csv('../input/test_variants')
stage2_test = pd.read_csv('../input/stage2_test_variants.csv')
stage1_solution = pd.read_csv('../input/stage1_solution_filtered.csv')

stage1_solution = stage1_solution.merge(stage1_test, how = 'left', on = 'ID')

stage2_test.merge(
        stage1_solution.drop('ID', axis = 1), 
        how = 'left', 
        on = ['Gene', 'Variation'])\
    .drop(['Gene', 'Variation'], axis = 1)\
    .fillna(1)\
    .to_csv('submission.csv', index = False)
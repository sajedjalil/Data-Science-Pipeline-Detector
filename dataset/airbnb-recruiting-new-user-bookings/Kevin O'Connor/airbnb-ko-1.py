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

# Importing data
age_gender_bkts = pd.read_csv('../input/age_gender_bkts.csv')
countries = pd.read_csv('../input/countries.csv')
sessions = pd.read_csv('../input/sessions.csv')
test_users = pd.read_csv('../input/test_users.csv')
train_users_2 = pd.read_csv('../input/train_users_2.csv')

# Possible locations: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF', and 'other'
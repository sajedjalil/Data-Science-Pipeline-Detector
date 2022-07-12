# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from compare import compare
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# -------- Read all input files to pandas df

dept_df = pd.read_csv('../input/departments.csv')
aisles_df = pd.read_csv('../input/aisles.csv')
order_products_df = pd.read_csv('../input/order_products__train.csv')
products_df = pd.read_csv('../input/products.csv')
orders_df = pd.read_csv('../input/orders.csv')
order_products_prior_df = pd.read_csv('../input/order_products__prior.csv')

# ---------------------------- EDA and Data checks ---------------------------- #


#print(orders_df.eval_set.value_counts())

#test = orders_df[orders_df['eval_set'].isin(['test'])]
#compare(test['order_id'] == order_products_df['order_id'], ignoreNames = TRUE, coerce = TRUE)


print(dept_df.department.nunique())

print(dept_df.department.value_counts())

print(len(dept_df.department))

print(dept_df.groupby(by = 'department'))


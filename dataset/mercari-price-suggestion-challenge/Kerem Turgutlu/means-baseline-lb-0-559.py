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


train = pd.read_csv('../input/train.tsv',
                    #nrows=10000, 
                    sep='\t')
                    
test = pd.read_csv('../input/test.tsv',
                   #nrows=10000,
                   sep='\t')

train.price = np.log(train.price + 1)

def get_means(train, test):
    test['price'] = np.nan
    cols = ['item_condition_id', 'category_name', 'brand_name', 'shipping', 'price']
    df = pd.concat([train, test])[cols]
    # replace NA to missing
    df.brand_name.fillna('missing', inplace=True)
    df.category_name.fillna('missing', inplace=True)
    # group stats by means
    df['price'] = df.groupby(['category_name','brand_name','item_condition_id', 'shipping']).transform(lambda x: x.mean())

    df['price'] = df.groupby(['category_name','brand_name','item_condition_id'])['price'].transform(lambda x: x.fillna(x.mean()))

    df['price'] = df.groupby(['category_name','brand_name'])['price'].transform(lambda x: x.fillna(x.mean()))

    df['price'] = df.groupby(['category_name'])['price'].transform(lambda x: x.fillna(x.mean()))

    df['price']  = df.price.fillna(df.price.mean())

    return df['price'][len(train):]
    
preds = np.exp(get_means(train, test)) - 1
pd.DataFrame({'test_id':test.test_id, 'price':preds}).to_csv('means_baseline.csv', index=False)








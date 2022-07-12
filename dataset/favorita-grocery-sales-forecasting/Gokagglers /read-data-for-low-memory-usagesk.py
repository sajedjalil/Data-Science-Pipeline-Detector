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


class Data(object):
    def __init__(self, data_folder, test_the_script = False):
        self.DATA_FOLDER = data_folder
        self.test_the_script = test_the_script
        self.read_data()
        # self.process_data()

        print('Train shape: ', self.train.shape)

    def read_data(self):
        self.nrows = None
        if self.test_the_script:
            self.nrows = 1000

        self.train = self.read_train_test_low_memory(train_flag = True)
        self.test = self.read_train_test_low_memory(train_flag = False)
        self.stores = self.read_stores_low_memory()
        self.items = self.read_items_low_memory()
        self.oil = self.read_oil_low_memory()
        self.transactions = self.read_transactions_low_memory()

        # if not self.test_the_script:
        #     self.train = self.train[self.train['date'] >= '2017-03-01']
            
    def read_train_test_low_memory(self, train_flag = True):
        filename = 'train'
        if not train_flag: filename = 'test'

        types = {'id': 'int64',
                'item_nbr': 'int32',
                'store_nbr': 'int8',
                'unit_sales': 'float32',
                'onpromotion': bool,
            }
        data = pd.read_csv(self.DATA_FOLDER + filename + '.csv', parse_dates = ['date'], dtype = types, 
                        nrows = self.nrows, infer_datetime_format = True)
        
        data['onpromotion'].fillna(False, inplace = True)
        data['onpromotion'] = data['onpromotion'].map({False : 0, True : 1})
        data['onpromotion'] = data['onpromotion'].astype('int8')
        
        return data

    def read_stores_low_memory(self):
        types = {'cluster': 'int32',
                'store_nbr': 'int8',
                }
        data = pd.read_csv(self.DATA_FOLDER + 'stores.csv', dtype = types)
        return data

    def read_items_low_memory(self):
        types = {'item_nbr': 'int32',
                'perishable': 'int8',
                'class' : 'int16'
                }
        data = pd.read_csv(self.DATA_FOLDER + 'items.csv', dtype = types)
        return data

    def read_oil_low_memory(self):
        types = {'dcoilwtico': 'float32',
                }
        data = pd.read_csv(self.DATA_FOLDER + 'oil.csv', parse_dates = ['date'], dtype = types, 
                                infer_datetime_format = True)
        return data
    
    def read_transactions_low_memory(self):
        types = {'transactions': 'int16',
                'store_nbr' : 'int8'
                }
        data = pd.read_csv(self.DATA_FOLDER + 'transactions.csv', parse_dates = ['date'], dtype = types, 
                                infer_datetime_format = True)
        return data

if __name__ == '__main__':
    
    data = Data(data_folder = '../input/', test_the_script = True)
    print(data.train.info())
    print(data.test.info())
    print(data.items.info())
    print(data.stores.info())
    print(data.transactions.info())
    print(data.oil.info())
        
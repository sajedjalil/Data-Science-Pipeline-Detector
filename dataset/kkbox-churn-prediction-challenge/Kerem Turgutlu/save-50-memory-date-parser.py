# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# You may want to install feather, a very fast serialization library, read in seconds !!!

#shell pip install feather-format (for pip)
#shell conda install feather-format -c conda-forge (for conda) 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import feather

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

dateparse = lambda x: pd.datetime.strptime(str(x), '%Y%m%d')

train = pd.read_csv('../data/train_v2.csv', dtype={'is_churn':'int16'})

logs = pd.read_csv('../data/user_logs_v2.csv', 
                  dtype={'date':'int32', 'num_25':'int16', 'num_50':'int16',
                         'num_75':'int16', 'num_985':'int16', 'num_100':'int16',
                        'num_unq':'int16', 'total_sec':'int32'})

members = pd.read_csv('../data/members_v3.csv', dtype={'city:':'int16', 'bd':'int16',
                                                       'registered_via':'int16', 
                                                      'registration_init_time':'int32'})

transactions = pd.read_csv('../data/transactions_v2.csv',
                          dtype = {'payment_method_id':'int16', 'payment_plan_days':'int16',
                           'plan_list_price':'int16', 'actual_amount_paid':'int16',
                           'is_auto_renew':'int16', 'transaction_date':'int16',
                            'membership_expire_date':'int16', 'is_cancel':'int16'})
                            
members.registration_init_time =  members.registration_init_time.apply(dateparse)
logs.date = logs.date.apply(dateparse)
transactions.transaction_date = transactions.transaction_date.apply(dateparse)
transactions.membership_expire_date = transactions.membership_expire_date.apply(dateparse)
                            
                            
train.to_feather('train')
logs.to_feather('logs')
members.to_feather('members')
transactions.to_feather('transcation')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
color = sns.color_palette()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


df_product  = pd.read_csv('../input/train_ver2.csv',usecols=['ncodpers'])
#print("Number of rows in train : ", df_product.shape[0])


'''for chunk in pd.read_csv('../input/train_ver2.csv',low_memory=False, chunksize=10):
    df=chunk[chunk['ind_cco_fin_ult1']==1]
    print(df.shape[0])'''

df=pd.read_csv('../input/train_ver2.csv',
     usecols=['ind_ahor_fin_ult1',
            'ind_aval_fin_ult1',
            'ind_cco_fin_ult1',
            'ind_cder_fin_ult1',
            'ind_cno_fin_ult1',
            'ind_ctju_fin_ult1'], 
     dtype='float16')

df.rename(columns={'ind_ahor_fin_ult1':'saving_account'},inplace=True)
df.rename(columns={'ind_aval_fin_ult1':'guarantees'},inplace=True)
df.rename(columns={'ind_cco_fin_ult1':'current_account'},inplace=True)
df.rename(columns={'ind_cder_fin_ult1':'derivatives_account'},inplace=True)
df.rename(columns={'ind_cno_fin_ult1':'payroll_account'},inplace=True)
df.rename(columns={'ind_ctju_fin_ult1':'junior_account'},inplace=True)

counts=df.astype('float64').sum(axis=0)
print(counts)


#print(list(df_product))

'''df=pd.read_csv('../input/test_ver2.csv', skiprows=1,
names=['As_of_date',
    'customer_id', 
    'Employee_ind',
    'country_of_residence', 
    'sex', 
    'age', 
    'Start_Date', 
    'New_Customer_less_6months_Flag', 
    'customer_seniority_months', 
    'primary_customer_flag', 
    'Last_date_Primary_Customer', 
    'customer_type_start_month', 
    'customer_relation_start_month',
    'resident_ind', 
    'Foriegn_birth_ind',
    'spouse_bank_employee_ind', 
    'Channel_to_join', 
    'Deceased_flag', 
    'Address_type', 
    'customer_address', 
    'province_name',  
    'Active_ind', 
    'Gross_Income',
    'segment_vip_ind_coll'],
dtype={'fecha_dato':str,'ncodpers':int}, 
low_memory=False)
df['Employee_indicator']=''
df.loc[df['Employee_ind']=='A','Employee_indicator']='Active'
df.loc[df['Employee_ind']=='B','Employee_indicator']='Ex-Employed'
df.loc[df['Employee_ind']=='N','Employee_indicator']='Not Employee'
df.loc[df['Employee_ind']=='P','Employee_indicator']='Pasive'

df_select= df[['As_of_date','customer_id', 
            'Employee_indicator'
            ]]'''



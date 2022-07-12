# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print("Reading files and getting top products..")
data_path="../input/"
dtype_dict = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}
train = pd.read_csv(data_path+"train_ver2.csv", dtype=dtype_dict, usecols=['ncodpers', 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1'])
count_dict = {}
for col_name in list(train.columns):
    if col_name != 'ncodpers':
        count_dict[col_name] = np.sum(train[col_name].astype('float64'))
        
top_products = sorted(count_dict, key=count_dict.get, reverse=True)

print("Drop duplicates and keep last one")
#train['ncodpers'] = train['ncodpers'].astype('int')
train = train.drop_duplicates('ncodpers', keep='last')


print("Read sample submission and merge with train..")
sub = pd.read_csv(data_path+"sample_submission.csv")
sub = sub.merge(train, on='ncodpers', how='left')
del train
sub.fillna(0, inplace=True)

print("Get the top products which are not already bought..")
ofile = open("simple_btb_v2.0.csv","w")
writer = csv.writer(ofile)
writer.writerow(['ncodpers', 'added_products'])
for ind, row in sub.iterrows():
    cust_id = row['ncodpers']
    top7_products = [] 
    for product in top_products:
        if int(row[product]) == 0:
            top7_products.append(str(product))
            if len(top7_products) == 7:
                break
    writer.writerow([cust_id, " ".join(top7_products)])
ofile.close()

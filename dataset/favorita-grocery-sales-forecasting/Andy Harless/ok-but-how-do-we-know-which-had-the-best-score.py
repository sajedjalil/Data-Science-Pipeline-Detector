# Weighted Ensemble of Two Public Kernels (LB:0.519)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import warnings
import time

start_time = time.time()
tcurrent   = start_time

warnings.filterwarnings("ignore")

# Ensemble of two public kernels
# Median-based from Paulo Pinto: https://www.kaggle.com/paulorzp/log-ma-and-days-of-week-means-lb-0-529
# LGBM from Ceshine Lee: https://www.kaggle.com/ceshine/lgbm-starter

filelist = ['../input/ensemble/LGBM.csv', '../input/ensemble/Median-based.csv']

outs = [pd.read_csv(f, index_col=0) for f in filelist]
concat_df = pd.concat(outs, axis=1)
concat_df.columns = ['submission1', 'submission2']

# original code using mean between the two submissions
#concat_df["unit_sales"] = concat_df.mean(axis=1)
#concat_df[["unit_sales"]].to_csv("ensemble.csv")



#------------- weighted ensemble approach 
print('Ensemble of Public Kernels with different weights\n')
print("AFAICT 'median' really isn't median at all; it's a combined moving average multiplied by .95")


v                       = 0.03                         # version
w                       = np.linspace(0.45,0.45,1)     # weighting

print('Adopted weights options:', w, '\n')


for i in range(len(w)):
     print ('Ensemble ' + str(i+1) + ': weight of LGBM = ', w[i], ', weight of Median = ', (1-w[i]))
     
     concat_df["unit_sales"] = w[i]*concat_df['submission1'] + (1-w[i])*concat_df['submission2'] 
     
     file_name = 'Ensemble using LGBM and Median - option ' + str(i+1) + ' - v' + str(v) + '.csv'
     print('File name', file_name, '\n')
     
     concat_df[["unit_sales"]].to_csv(file_name)
     
t = (time.time() - start_time)/60
print ("Total processing time %s min" % t)
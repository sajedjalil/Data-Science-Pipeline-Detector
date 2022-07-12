# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

"""
   Step 0: Import packages
   Step 1: Read Data / Transform Data Type / Visualise Data
   Step 2: Check for NA or Missing Data / Prepare for Testing of Stationarity
   Step 3: If the series is not Stationary, transform data and repeat Step 2
   Step 4: Fit ARIMA model at different orders / Choose the best model ARIMA(p,d,q)
   Step 5: Fit the best ARIMA Model
   Step 6: Predict Data 
   Step 7: Write in the file
   Step 8: Submit Prediction
"""

#Step 0: Import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime # manipulating date formats

#Step 0.1 (Visualisation)
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots

#Step 0.2 (Time Series)
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

#Step 0.3 (Settings)
import warnings
warnings.filterwarnings("ignore")

#Step 1 Read Data / Transform Data Type / Visualise Data

#Step 1.0 Files in hand
#print(os.listdir("../input"))

#Step 1.1 Read Data from files
items = pd.read_csv("../input/items.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
item_categories = pd.read_csv("../input/item_categories.csv")
shops = pd.read_csv("../input/shops.csv")
sales_test = pd.read_csv("../input/test.csv")
sales_train = pd.read_csv("../input/sales_train.csv")

#Step 1.2 Check Data and Data-type

sales_train_clean = sales_train.drop(labels = ['date', 'item_price'], axis = 1)

#sales_train_clean.head() >>> To check the data structure

# change the item count per day to item count per month by using group
sales_train_clean = sales_train_clean.groupby(["item_id","shop_id","date_block_num"]).sum().reset_index()
sales_train_clean = sales_train_clean.rename(index=str, columns = {"item_cnt_day":"item_cnt_month"})
sales_train_clean = sales_train_clean[["item_id","shop_id","date_block_num","item_cnt_month"]]

# intialising variables
group_list = [] # shop_id & item_id combination in test.csv which we need to predict
prediction_list = [] # predicted value for the above combination
month_34 = [] # check prediction vis-a-vis last month sale
yhat_print = 0 # print format for prediction
df_test = 0
#prediction_group = []
# iteration is to be done for all group_id 0 to 214199

""" Choose start_point and end_point of group_id, Total combination is 214200 Start = 0, End = 214199"""
start_point = 0
end_point = 100
for i in range(start_point,end_point): # actually this should be range(214200)
    group_id = i 
    shop_no = sales_test.loc[group_id]['shop_id']
    item_no = sales_test.loc[group_id]['item_id']
    
    """ This block is checking for a particular shop id + item id combination from test set """
    
    check = sales_train_clean[["shop_id","item_id","date_block_num","item_cnt_month"]]
    check = check.loc[check['shop_id'] == shop_no]     # checking for a specific shop id
    check = check.loc[check['item_id'] == item_no]     # checking for a specific item id
    
    """ Check if last 3 months sales is 0 also if 25th & 26th month is also 0"""
    check0_25 = check.loc[check['date_block_num'] == 25]
    check0_26 = check.loc[check['date_block_num'] == 26]
    check0_31 = check.loc[check['date_block_num'] == 31]
    check0_32 = check.loc[check['date_block_num'] == 32]
    check0_33 = check.loc[check['date_block_num'] == 33]
    x25 = pd.DataFrame(check0_25).empty
    x26 = pd.DataFrame(check0_26).empty
    x31 = pd.DataFrame(check0_31).empty
    x32 = pd.DataFrame(check0_32).empty
    x33 = pd.DataFrame(check0_33).empty
    y = x25 and x26 and x31 and x32 and x33
    if y == True:
        #print ("No prediction required for group", group_id)
        flag = 0
        df_test = 0
        yhat_print = float(df_test)
        #prediction_group.append(group_id)
    else:
        #print ("Need to predict for group", group_id)
        df_test = 1
        #prediction_group.append(group_id)
        #print ("Prediction List", prediction_group)
        #groups_dropped = i - len(prediction_group)
        #print ("Groups dropped", groups_dropped)

    
        """Step 2: Check for NA or Missing Data / Prepare for Testing of Stationarity""" 
        
        if df_test != 0:
            
            num_month = sales_train['date_block_num'].max()
            month_list=[i for i in range(num_month+1)]
            shop = []
            for j in range(num_month+1):
                shop.append(shop_no)
            item = []
            for k in range(num_month+1):
                item.append(item_no)
            months_full = pd.DataFrame({'shop_id':shop, 'item_id':item,'date_block_num':month_list})
                
            sales_34month = pd.merge(check, months_full, how='right', on=['shop_id','item_id','date_block_num'])
            sales_34month_month_sorted = sales_34month.sort_values(by=['date_block_num'])
            sales_34month.fillna(0.00,inplace=True)
            sales_34month_month_sorted.fillna(0.00,inplace=True)
            sales_34month_month_sorted['date_block_num'] = pd.to_datetime(sales_34month_month_sorted['date_block_num'], unit='m')
            
            """Step 3: Check If the series is Stationary, 
                       Run Dickey-Fuller test...If the p-value is <= 0.01; 
                       the series is stationary, else it is non-stationary"""
           
            diff_0 = sales_34month['item_cnt_month'] # Pass the series for which D-F test to run
            d = 0
            dftest = adfuller(diff_0, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            #for key,value in dftest[4].items(): 
            #    dfoutput['Critical Value (%s)'%key] = value
            check = dfoutput['p-value']
            if check <= 0.01:
                #print ("The p-value is",check)
                #print ("Series is Stationary")
                flag = 1
            else:
                #print ("The p-value is",check)
                #print ("Series is Non-Stationary >> Go for difference")
                
                for m in range (2):
                    d = m+1
                    diff_m = diff_0.diff(d)
                    diff_m.dropna(inplace=True)
            
                    dftest = adfuller(diff_m, autolag='AIC')
                    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
                    
                    check = dfoutput['p-value']
                    if check <= 0.01:
                        #print ("The p-value is",check)
                        #print ("Series is Stationary")
                        flag = 1
                        break
                    else:
                        if d == 2:
                            #print ("The series is non-stationary at d = 2, so no foreacast possible")
                            flag = 0
                            break
                        #print ("The p-value is",check)
                            
            if flag == 0:
                #print ("Skip forecast routine")
                yhat = 0
            else:
                #print ("The value of d is: ",d)
                dummy = 0
    
        """Step 4: Fit ARIMA model at different orders / Choose the best model ARIMA(p,d,q)"""
        
        while flag !=0:
            
            from itertools import product
            
            ps = range(0, 5) # Up to 5 AR terms
            ds = d           # Differencing Value
            qs = range(0, 5) # Up to 5 MA terms
            
            params = product(ps, qs)
            params_list = list(params)
                   
            def optimiseARIMA(ts, params_list, d):
                results = []
                best_aic = np.inf
                for param in params_list:
                    try:
                        arima = sm.tsa.ARIMA(ts.astype(float), freq = 'D',
                                            order=(param[0], d, param[1])).fit()
                    except:
                        continue
                    aic = arima.aic
                    if aic < best_aic:
                        best_model = arima
                        best_aic = aic
                        best_param = param
                    results.append([param, arima.aic])
                df_results = pd.DataFrame(results)
                df_results.columns = ['parameters', 'aic']
                df_results = df_results.sort_values(by='aic', ascending=True).reset_index(drop=True)
                return best_param
            
            """ Format Data Series to Time Series Format to find the best p,d,q"""
            
            ts_format = sales_34month_month_sorted[['date_block_num','item_cnt_month']]
            ts_format = ts_format.rename(columns = {'date_block_num':'month','item_cnt_month':'sales'})
            ts_format.index = ts_format['month']
            
            month_34_sales = ts_format.iloc[-1]['sales']
            
            """ Run OptimiseArima function """
            
            check1 = optimiseARIMA(ts_format['sales'], params_list, ds)
                    
            """Step 5: Fit the best ARIMA Model"""
            
            best_p=check1[0]
            best_d=ds
            best_q=check1[1]
                    
            bestARIMA_shop_item_fit = sm.tsa.ARIMA(ts_format["sales"],freq='m', order=(best_p, best_d, best_q)).fit()
            
            """Step 6: Predict Data"""
            
            yhat = bestARIMA_shop_item_fit.forecast()[0]
            yhat_element = yhat[0]
            yhat_print = abs(round(yhat_element,1))
            flag = 0
        #print ("The value of best (p,d,q) for Group_ID {} is: ({},{},{})".format(i,best_p,best_d,best_q))
    print('The predicted value for Group ID {Group_ID} ({Shop}, {Item}) is {predicted}'.format(Group_ID=i,Shop = shop_no, Item = item_no, predicted = yhat_print))
        
    group_list.append(group_id)
    prediction_list.append(yhat_print)

"""Step 7: Write in the file"""

final_data = {"ID": group_list, "item_cnt_month": prediction_list}
#print (final_data)
final_df = pd.DataFrame(final_data, columns = ["ID","item_cnt_month"])

"""Step 8: Submit Prediction"""

final_df.to_csv('final_submission_dc.csv',index=False)

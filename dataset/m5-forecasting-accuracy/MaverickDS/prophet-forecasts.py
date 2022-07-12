import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet
import multiprocessing
from joblib import Parallel, delayed

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Reading in the files
SalesTrain= pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
SampleSubmission= pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
SellPrices= pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
Calendar= pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

# Demand by Department per Store: Level to run forecast using Prophet
SalesDepStore = SalesTrain.groupby(['dept_id','store_id'], as_index=False).sum()

# Get the fraction of item sales in specific store - department combination
SalesTrainSub=SalesTrain[['id','dept_id','store_id']]
SalesDepStoreSub=SalesDepStore[['dept_id','store_id']]
SalesTrainSub['ItematStore1YrMean'] = SalesTrain.iloc[:,1550:1947].mean(axis=1)
SalesDepStoreSub['StoreDep1YrMean'] = SalesDepStore.iloc[:,1550:1947].mean(axis=1)
SalesTrainDepStore = pd.merge(SalesTrainSub,SalesDepStoreSub, on =['dept_id','store_id'])
SalesTrainDepStore['MeanFraction1Yr'] = SalesTrainDepStore['ItematStore1YrMean']/SalesTrainDepStore['StoreDep1YrMean']

# Subset to merge with Prices
Train_sub = SalesTrain[['item_id','dept_id','store_id']]
SellPrices = pd.merge(Train_sub,SellPrices,on=['item_id','store_id'])
# Price by Department per Store: Level to run forecast using Prophet
SellPricesDepStore = SellPrices[['dept_id','store_id','wm_yr_wk','sell_price']].groupby(['dept_id','store_id','wm_yr_wk'], as_index=False).mean()

CalendarSub=Calendar[['wm_yr_wk','d']]
SellPricesDepStore = pd.merge(SellPricesDepStore,CalendarSub,on='wm_yr_wk')

SellPricesRegs=SellPricesDepStore.pivot_table(index=['dept_id','store_id'], columns='d', values='sell_price')
SalesTrainDepStoreIndSub=SalesTrainDepStore[['dept_id','store_id']]
SellPricesRegs = pd.merge(SellPricesRegs,SalesTrainDepStoreIndSub,on=['dept_id','store_id'])
SellPricesRegs = SellPricesRegs.reset_index(drop=True)

HolidayDF1 = Calendar.iloc[858:1969,].loc[Calendar['event_name_1'].notnull()][['event_name_1','date']].rename(columns={'event_name_1':'holiday','date':'ds'})
HolidayDF2 = Calendar.iloc[858:1969,].loc[Calendar['event_name_1'].notnull()][['event_type_1','date']].rename(columns={'event_type_1':'holiday','date':'ds'})
HolidayDF3 = Calendar.iloc[858:1969,].loc[Calendar['event_name_2'].notnull()][['event_name_2','date']].rename(columns={'event_name_2':'holiday','date':'ds'})
HolidayDF4 = Calendar.iloc[858:1969,].loc[Calendar['event_name_2'].notnull()][['event_type_2','date']].rename(columns={'event_type_2':'holiday','date':'ds'})
holidays = pd.concat((HolidayDF1, HolidayDF2,HolidayDF3,HolidayDF4))

# Forecast function:
def ProphetFC(i):
    m = Prophet(yearly_seasonality=20, holidays=holidays)
    m.add_seasonality(name='monthly', period=28, fourier_order=10)
    m.add_seasonality(name='weekly', period=7, fourier_order=5)
    tsdf = pd.DataFrame({
      'ds': pd.to_datetime(Calendar.iloc[858:1941,]['date'].reset_index(drop=True)),
      'y': SalesDepStore.iloc[i,860:1943].reset_index(drop=True),
    })
    #tsdf['sell_price']=SellPricesRegs.iloc[i,860:1943].reset_index(drop=True)
    tsdf['wday']=Calendar.iloc[858:1941,]['wday'].reset_index(drop=True)
    tsdf['month']=Calendar.iloc[858:1941,]['month'].reset_index(drop=True)
    tsdf['year']=Calendar.iloc[858:1941,]['year'].reset_index(drop=True)
    #m.add_regressor('sell_price')
    m.add_regressor('wday')
    m.add_regressor('month')
    m.add_regressor('year')
    m.fit(tsdf)
    future = m.make_future_dataframe(periods=28)
    #future['sell_price']=SellPricesRegs.iloc[i,860:1972].reset_index(drop=True)
    future['wday']=Calendar.iloc[858:1969,]['wday'].reset_index(drop=True)
    future['month']=Calendar.iloc[858:1969,]['month'].reset_index(drop=True)
    future['year']=Calendar.iloc[858:1969,]['year'].reset_index(drop=True)
    fcst = m.predict(future)
    print("Iteration ", i, "Completed")
    FCAST = pd.DataFrame(fcst.iloc[1083:1112,]['yhat'])
    FCAST['dept_id']=SalesDepStore.iloc[i,]['dept_id']
    FCAST['store_id']=SalesDepStore.iloc[i,]['store_id']
    return(FCAST)

# Parallel jobs to Forecast
num_cores = multiprocessing.cpu_count()
if __name__ == "__main__":
    processed_FC = Parallel(n_jobs=num_cores)(delayed(ProphetFC)(i) for i in range(SalesDepStore.shape[0]))
    
# Combining obtained data frames
FCAST = pd.concat(processed_FC[0:70])
FCAST['Period']=FCAST.index
FCAST['Period']=FCAST['Period']-1082

FCASTPivot=FCAST.pivot_table(index=['dept_id','store_id'], columns='Period', values='yhat')
Submission = pd.merge(SalesTrainDepStore,FCASTPivot, on =['dept_id','store_id'])
for i in range(28):
    Submission.iloc[:,(6+i)] = Submission.iloc[:,(6+i)]*Submission['MeanFraction1Yr']

Submission_valid=Submission[['id', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
Submission_valid.columns = SampleSubmission.columns

SampleSubmission=SampleSubmission[['id']]
SubmissionFinal = pd.merge(SampleSubmission, Submission_valid, on = 'id', how = 'left')
SubmissionFinal = SubmissionFinal.fillna(0)
SubmissionFinal.to_csv('submission.csv',index=False)
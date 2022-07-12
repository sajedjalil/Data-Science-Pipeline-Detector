import pandas as pd
import numpy as np
import multiprocessing
from itertools import product
from statsmodels.tsa.api import ExponentialSmoothing

# %% [code]
#Reading required data
train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv',parse_dates=['date'])
sub = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

# I thought of trying some statistical method to do the prediction, 
# so with the help of this notebook https://www.kaggle.com/marisakamozz/m5-exponential-smoothing I tried exponential smoothing
# this script was executed on my local system. And whatever output getnerated, I used it to create submission file in another notebook.
# For complete details about the code, check this repo https://github.com/purnima291/M5-Forecasting-Accuracy 
# After submission I got the private score of 0.67114
# I am also learning, so if you find any mistake or if you have any question please feel free to comment.

#Creating empty dtataframe
columns = ['index']+list(sub.columns)
data = pd.DataFrame(columns=columns)

# 5 queues for storing output from each process function.
A = multiprocessing.Queue()
B = multiprocessing.Queue()
C = multiprocessing.Queue()
D = multiprocessing.Queue()
E = multiprocessing.Queue()

def my_product(inp):
    return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]
    
def prediction1(i):
	#global data
    best_score = np.inf
    best_model = None

    for param in params:
        model = ExponentialSmoothing(train.iloc[i],seasonal_periods=7, initialization_method='estimated', freq='D', **param).fit()
        if best_score > model.aicc:
            best_score = model.aicc
            best_model = model
            
    pred = model.forecast(28)
    df = pd.DataFrame({'forecast':pred})
    row = [i,sub.iloc[i+30490,0]]+list(df['forecast'].values)
    
   
    A.put(row)
    

def prediction2(i):
    best_score = np.inf
    best_model = None

    for param in params:
        model = ExponentialSmoothing(train.iloc[i],seasonal_periods=7, initialization_method='estimated', freq='D', **param).fit()
        if best_score > model.aicc:
            best_score = model.aicc
            best_model = model
            
    pred = model.forecast(28)
    df = pd.DataFrame({'forecast':pred})
    row = [i,sub.iloc[i+30490,0]]+list(df['forecast'].values)
    
    #print(row)
    B.put(row)

def prediction3(i):
    best_score = np.inf
    best_model = None

    for param in params:
        model = ExponentialSmoothing(train.iloc[i],seasonal_periods=7, initialization_method='estimated', freq='D', **param).fit()
        if best_score > model.aicc:
            best_score = model.aicc
            best_model = model
            
    pred = model.forecast(28)
    df = pd.DataFrame({'forecast':pred})
    row = [i,sub.iloc[i+30490,0]]+list(df['forecast'].values)
    
    #print(row)
    C.put(row)

def prediction4(i):
    best_score = np.inf
    best_model = None

    for param in params:
        model = ExponentialSmoothing(train.iloc[i],seasonal_periods=7, initialization_method='estimated', freq='D', **param).fit()
        if best_score > model.aicc:
            best_score = model.aicc
            best_model = model
            
    pred = model.forecast(28)
    df = pd.DataFrame({'forecast':pred})
    row = [i,sub.iloc[i+30490,0]]+list(df['forecast'].values)
    
    #print(row)
    D.put(row)

def prediction5(i):
    best_score = np.inf
    best_model = None

    for param in params:
        model = ExponentialSmoothing(train.iloc[i],seasonal_periods=7, initialization_method='estimated', freq='D', **param).fit()
        if best_score > model.aicc:
            best_score = model.aicc
            best_model = model
            
    pred = model.forecast(28)
    df = pd.DataFrame({'forecast':pred})
    row = [i,sub.iloc[i+30490,0]]+list(df['forecast'].values)
    
    #print(row)
    E.put(row)


if __name__ == "__main__": 
	
	pattern = {
    'trend': [None, 'add'],
    'seasonal': [None, 'add'],
}
	params = my_product(pattern)

	train.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace=True)
	train.set_index('id',inplace=True)
	train.columns = calendar.date[:len(train.columns)]

	#Total 30490 time-series is given. 

	for i in range(6098):

		process1 = multiprocessing.Process(target=prediction1, args=(i, ))
		process2 = multiprocessing.Process(target=prediction2, args=([i+6098, ]))
		process3 = multiprocessing.Process(target=prediction3, args=([i+12196, ]))
		process4 = multiprocessing.Process(target=prediction4, args=([i+18294, ]))
		process5 = multiprocessing.Process(target=prediction5, args=([i+24392, ]))
		

		process1.start()
		process2.start()
		process3.start()
		process4.start()
		process5.start()

		row1 = A.get()
		data = data.append(pd.DataFrame([row1],columns=columns))
		row2 = B.get()
		data = data.append(pd.DataFrame([row2],columns=columns))
		row3 = C.get()
		data = data.append(pd.DataFrame([row3],columns=columns))
		row4 = D.get()
		data = data.append(pd.DataFrame([row4],columns=columns))
		row5 = E.get()
		data = data.append(pd.DataFrame([row5],columns=columns))

		process1.join()
		process2.join()
		process3.join()
		process4.join()
		process5.join()
		
		#print('Done',i)
		
	data.to_csv('evaluation.csv',index = False)
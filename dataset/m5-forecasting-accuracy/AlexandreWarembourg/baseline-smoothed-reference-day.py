import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

days = range(1, 1913 + 1)
time_series_columns = [f'd_{i}' for i in days]
time_series_data = train_sales[time_series_columns]

for i in range(28):
    base_ = 1913
    reference_t1 = int(base_+i - 363)
    reference_t = int(base_+i - 364)
    reference_t2 = int(base_+i - 365)
    time_series_data['F'+str(i+1)] =   time_series_data[f'd_{reference_t1}']*0.25 + time_series_data[f'd_{reference_t}']*0.5 + time_series_data[f'd_{reference_t2}']*0.25 
    
forecast = time_series_data.iloc[:, -28:]
validation_ids = train_sales['id'].values
evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]
ids = np.concatenate([validation_ids, evaluation_ids])
predictions = pd.DataFrame(ids, columns=['id'])
forecast = pd.concat([forecast] * 2).reset_index(drop=True)
predictions = pd.concat([predictions, forecast], axis=1)
predictions.to_csv('submission.csv', index=False)
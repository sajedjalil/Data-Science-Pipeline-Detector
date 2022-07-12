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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline

from bokeh.plotting import figure, output_notebook, show, vplot, ColumnDataSource
from bokeh.charts import TimeSeries
from bokeh.models import HoverTool, CrosshairTool
from bokeh.palettes import brewer
import gc
import dask.dataframe as dd

output_notebook()
# ## This Notebook analyses trends of hotel bookings and clicks
# 
# we will start with reading the data, leaving just necessary columns, aggregating it to the day level and dropping the original dataframe
train  =  pd.read_csv('../input/train.csv', usecols = ('date_time', 'hotel_cluster','is_booking'), 
                      parse_dates = ['date_time'])
train['dow'] = train.date_time.dt.weekday
train['year'] = train.date_time.dt.year
train['month'] = train.date_time.dt.month
train['day'] = train.date_time.dt.day
train_agg = train.groupby(['dow','year','month','day', 'hotel_cluster']).agg(['sum', 'count'] )
train_agg.columns = ('bookings', 'total')
train_agg.head()
train_agg.info()

del(train)

gc.collect()
# ## Bookings per day of week
date_agg_1 = train_agg.groupby(level=0).agg(['sum'] )
date_agg_1.columns = ('bookings', 'total')
date_agg_1.head()
date_agg_1.plot( kind = 'bar', stacked = True )
# ## Bookings by year
date_agg_2 = train_agg.groupby(level=1).sum()
date_agg_2.columns = ('bookings', 'total')
date_agg_2.index.name = 'Year'
date_agg_2.plot(kind='bar', stacked='True')
# ## Bookings by month
date_agg_3 = train_agg.groupby(level=[1,2]).sum()
date_agg_3.columns = ('bookings', 'total')
date_agg_3.plot(kind='bar', stacked='True',figsize=(16,10))
# ## Interactive booking, click, and percentage of booking trends with Bokeh
date_agg_4 = train_agg.groupby(level=[1,2,3]).sum()
date_agg_4.columns = ('bookings', 'total')
date_agg_4.reset_index(inplace=True)
date_agg_4['dt'] = pd.to_datetime(date_agg_4.year*10000 + date_agg_4.month*100 + date_agg_4.day
                                  , format='%Y%m%d')
date_agg_4.head()
def make_plot(vals, title, ylab):
    hover = HoverTool(
        tooltips=[
            ("Date", "@day"),
            ("Day of week", "@dow"),
            ("clicks", "@clicks"),
            ("bookings", "@bookings"),
        ]
    )


    ch = CrosshairTool(dimensions = ['height'], line_color='red')

    src  = ColumnDataSource({'day': date_agg_4.dt.dt.strftime('%Y-%m-%d') ,
                             'dow': date_agg_4.dt.dt.weekday.tolist(),
                             'clicks': date_agg_4.total - date_agg_4.bookings,
                             'bookings': date_agg_4.bookings})

    p = figure(x_axis_type = 'datetime',plot_width=800, plot_height=400, tools=[hover, ch, 
                                                                                'pan,save, wheel_zoom,box_zoom,reset,resize'])
    p.title = title
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = ylab

    p.line((date_agg_4['dt']), vals, color='green',  source = src)
    
    return p
p = make_plot(date_agg_4['total'] - date_agg_4['bookings'], 'Expedia daily clicks', 'Clicks')
show(p)
p = make_plot(date_agg_4['bookings'], 'Expedia daily bookings', 'bookings')
show(p)
p = make_plot(date_agg_4['bookings'] / date_agg_4['total'] *100, 'Expedia daily bookings %', 'Bookings, %')
show(p)
# ## Building interactive charts for hotel clusters, 5 clusters per chart
pv_agg = train_agg.reset_index()
pv_agg['dt'] = pd.to_datetime( pv_agg.year*10000 + pv_agg.month*100 + pv_agg.day
                                  , format='%Y%m%d')
pv_agg = pv_agg.pivot(index = 'dt', columns = 'hotel_cluster', values = 'bookings')
pv_agg.columns = [str(i) for i in pv_agg.columns]
pv_agg['dt'] = pv_agg.index
pv_agg['dow'] = pv_agg.dt.dt.weekday
pv_agg.head()
def make_hc_plot(df, start, stop):
    hover = HoverTool(
        tooltips=[
            ("Date", "@day"),
            ("Day of week", "@dow"),
            ("cluster", "@cluster"),
            ("bookings", "@bookings"),
        ]
    )
    
    #colors = brewer['RdYlBu'][stop-start]
    colors = ['red', 'darkmagenta', 'green', 'darkorange', 'blue']
    ch = CrosshairTool(dimensions = ['height'], line_color='red')
    p = figure(x_axis_type = 'datetime',plot_width=800, plot_height=400, tools=[hover, ch, 'pan,wheel_zoom,save,box_zoom,reset,resize'])
    p.title = 'Expedia bookings for subset of clusters {} to {}'.format(start, stop)
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Number of bookings'
    
    for i in range(start, stop):
        src  = ColumnDataSource({'day': df.dt.dt.strftime('%Y-%m-%d'), 
                                 'dow': df.dow.tolist(),
                                 'bookings': df[str(i)],
                                 'cluster': [i]*df.shape[0]})
        
        p.line((df['dt']), df[str(i)], color=colors[i-start], legend = str(i), source = src)

    return p
tslines = []

for i in range(0,100,5):
    tsline = make_hc_plot(pv_agg, i, i+5)
    tslines.append(tsline)
    
show(vplot(*tslines))
print('done')
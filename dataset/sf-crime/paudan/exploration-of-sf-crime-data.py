# -*- coding: utf-8 -*-

__author__ = 'Paulius Danenas'

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Turn off warnings display
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../input/train.csv', header=0, sep=",", quotechar='"')
data["Dates"]= pd.to_datetime(data["Dates"])
rows = len(data.axes[0])

def make_pairplot(data, district, time_dim):
    sns.set(style="whitegrid")
    g = sns.PairGrid(data, x_vars=data.columns[:-1], y_vars=['Category'] ,size=8, aspect=.25)
    g.map(sns.stripplot, size=8, orient="h", palette="PuBu_r", edgecolor="gray")
    g.set(title="Distribution of crimes per %s%s" % (time_dim.lower(), 
                                                     "" if district is None else " in district %s" % district),
          xlabel="Crimes", ylabel="", )    
    titles = list(data.columns[:-1].values)
    for ax, title in zip(g.axes.flat, titles):
        ax.set(title=title)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)  
        _ = plt.setp(ax.get_xticklabels(), rotation=90)
    sns.despine(left=True, bottom=True)
    sns.plt.show()
    return g


def visualize_distribution(data, district, split_at=7, time_dim='Year'): 
    res = data[[time_dim, 'Category']].groupby([time_dim, 'Category'])[time_dim].count()
    res = res.unstack('Category').fillna(0)
    res = pd.concat([res,pd.DataFrame(res.sum(axis=0),columns=['Total']).T]).T
    res.sort(['Total'], ascending=False, inplace=True)
    res['Category'] = res.index
    ind = 1
    # fig = plt.figure()
    if split_at > 0:
        from_ = 0
        no_cols = len(res.columns)      
        while from_ < no_cols-1: 
            if  from_ + split_at < no_cols:           
                cols = list(range(from_, from_ + split_at)) + list([no_cols-1]) 
            else:
                cols = list(range(from_, no_cols))
            g = make_pairplot(res[res.columns[cols]], district, time_dim)
            # fig.add_axes(g)
            from_ = from_ + split_at
            ind += 1     
    else:      
        g = make_pairplot(res, district, time_dim)
        # fig.add_axes(g)
        ind += 1
    res = res.ix[:, :-2].T
    axes = plt.subplot(ind+1, 1, ind+1)
    axes.set_title("Boxplot of crimes per %s in district %s" % (time_dim.lower(), 
                                                                "" if district is None else " in district %s" % district))
    res.boxplot(ax=axes)
    # fig.add_axes(axes)
    plt.xticks(rotation=90)
    plt.show()
    

def draw_heatmap1(data):
    cdata = pd.crosstab(data.Category, data.Resolution)
    # Display statistics as percentages of total crimes
    cdata = cdata.apply(lambda x: x/float(x.sum())*100)
    fig, ax = plt.subplots()
    fig.set_size_inches(0.9*len(cdata.columns), 0.65*len(cdata.index))
    ax.set_title("Distribution of crime categories by their resolution")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Crime category")
    heat_map = sns.heatmap(cdata, square=True, linewidths=1, label='tiny', ax=ax, annot=True, fmt=".2f")
    plt.show()
    
    
def draw_heatmap2(data, time_dim='Year'):
    cdata = pd.crosstab(data.Category, data[time_dim])
    cdata = cdata.apply(lambda x: x/float(x.sum())*100)
    if time_dim == 'DayOfWeek':
        cdata = cdata[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
    fig, ax = plt.subplots()
    fig.set_size_inches(0.9*len(cdata.columns), 0.65*len(cdata.index))
    ax.set_title("Distribution of crime categories by the district")
    ax.set_xlabel(time_dim)
    ax.set_ylabel("Crime category")
    heat_map = sns.heatmap(cdata, square=True, linewidths=1, label='tiny', ax=ax, annot=True, fmt=".2f")
    plt.show()
  
  
def draw_all(data, district, split_at=7, time_dim='Year'):
    visualize_distribution(data, district, split_at, time_dim)
    draw_heatmap1(data)
    draw_heatmap2(data, time_dim)


# Plot distributions per year    
data['Year'] = data['Dates'].dt.year
# Plot overall distributions
visualize_distribution(data, district=None)
draw_heatmap1(data)
draw_heatmap2(data)
# Plot distributions in districts
for district in data["PdDistrict"].unique():
    draw_all(data[(data["PdDistrict"] == district) & (data['Year'] <= 2014)], district)

# Plot distributions per month
data['Month'] = data['Dates'].dt.month
# Plot overall distributions
visualize_distribution(data, district=None, time_dim='Month', split_at=6)
draw_heatmap1(data)
draw_heatmap2(data, time_dim="Month")
# Plot distributions in districts
for district in data["PdDistrict"].unique():
    draw_all(data[data["PdDistrict"] == district], district)
    
# Plot distributions per weekday
visualize_distribution(data, district=None, time_dim='DayOfWeek', split_at=0)
draw_heatmap1(data)
draw_heatmap2(data, time_dim="DayOfWeek")
# Plot distributions in districts
for district in data["PdDistrict"].unique():
    draw_all(data[data["PdDistrict"] == district], district)



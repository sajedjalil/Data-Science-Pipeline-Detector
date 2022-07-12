# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import pi

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore') # silence annoying warnings

#setup path variables
path = "../"
app_input="../input"
app_output="../output"

# aisles
aisles = pd.read_csv('../input/aisles.csv', engine='c')
print('Total aisles: {}'.format(aisles.shape[0]))
aisles.head()

# departments
departments = pd.read_csv('../input/departments.csv', engine='c')
print('Total departments: {}'.format(departments.shape[0]))
departments.head()

# products
products = pd.read_csv('../input/products.csv', engine='c')
print('Total products: {}'.format(products.shape[0]))
products.head(5)

# combine aisles, departments and products (left joined to products)
goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')
# to retain '-' and make product names more "standard"
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 

goods.head()



#bokeh script to generate html file (credits udemy course "Interactive Data Visualization with Python & Bokeh" by ardit )
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import all_palettes as alp
from bokeh.models import Range1d, PanTool, ResetTool, HoverTool,ColumnDataSource

#create column data source for bokeh
df = pd.DataFrame(goods.groupby(['department']).count()['product_id'].sort_values(ascending=False)).reset_index()
source = ColumnDataSource(df)



#x_range displays the labels for the x axis
p1 = figure(title="No of Products by Department",background_fill_color="#E8DDCB",x_range=list(df["department"]),
            tools="pan,box_select,lasso_select,reset", active_drag="lasso_select")

#Style the tools
p1.toolbar_sticky=False
p1.toolbar_location='below'
p1.toolbar.logo=None

#very imp concept here. We are placing the major ticks using the width list
width = [index+.5 for index,dep in enumerate(df.department)]
p1.quad(top="product_id", bottom=0, left=width[:-1], right=width[1:],
        fill_color="#036564", legend="department",source=source)

#style the axis
p1.xaxis.major_label_orientation = pi/2 #we want to display the vertical text using this 
p1.xaxis.axis_label = 'Departments'
p1.yaxis.axis_label = 'Number of Products'

#style the legend 
p1.legend.location = "top_right"
#p1.legend.location=(575,555)
p1.legend.background_fill_alpha=0
p1.legend.border_line_color=None
p1.legend.legend_margin=10
p1.legend.legend_padding=10
p1.legend.label_text_color='olive'
p1.legend.label_text_font='times'

output_file('histogram.html', title="Histogram Using Bokeh.Plotting")

show(gridplot(p1, ncols=2, plot_width=800, plot_height=600))


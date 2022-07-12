# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import animation
from matplotlib import cm
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
plt.style.use('ggplot')

train=pd.read_csv('../input/train.csv')

#Sort places by popularity
sorted_visits=train.groupby('place_id')['time'].count().sort_values(ascending=False)

n=10 #number of most popular places to plot
decay=20 #If not False older check ins will disappear from the plot


timemin=99999999999
timemax=0

plots=[] #Animated objects to update
c=[cm.jet(x) for x in np.linspace(0.0, 1.0, n)] #let's borrow colors from everyone's dear Jet

data=[] #Temporary data storage

fig = plt.figure()
ax=fig.add_subplot(111)

for i  in range(n):
    place=train[train.place_id==sorted_visits.index[i]]
    data.append(place)
    
    timemin=min(timemin,place.time.min())
    timemax=max(timemax,place.time.max())

    ax.set_xlim([-2.0,12.0])
    ax.set_ylim([-2.0,12.0])
    
    plots.append(plt.scatter(x=[],y=[],c=c[i], alpha=0.1))
plots.append(ax.text(0.1,0.9,'',transform=ax.transAxes))
timeinc=(timemax-timemin)/600.0# 30fps 20 seconds


def init():    
    return tuple(plots)

def redraw(frame):
    current_time=timemin+timeinc*frame
    for i in range(n):
        if decay:
            place=data[i][(data[i].time<=current_time)&(data[i].time>=current_time-decay*timeinc)]
        else:
            place=data[i][data[i].time<=current_time]
        #print len(place)
        
        plots[i].set_offsets(np.hstack((place.x.values[:,np.newaxis],place.y.values[:,np.newaxis])))
        plots[i]._sizes = place.accuracy.values
    txt='Day:{} Weekday:{} Hour:{}'.format(int(current_time/1440), int(current_time/1440)%7, int(current_time/60)%24)
    plots[-1].set_text(txt)
    return tuple(plots)

        

anim = animation.FuncAnimation(fig, redraw, init_func=init, blit=True,
                               frames=600, interval=1, repeat=True)
anim.save('PopularCheckins.gif', writer='imagemagick', fps=30)
plt.show()
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import animation

train = pd.read_csv('../input/train.csv')

train.pickup_datetime = pd.to_datetime(train.pickup_datetime)
train.dropoff_datetime = pd.to_datetime(train.dropoff_datetime)
train['hour_pick'] = train.pickup_datetime.dt.hour
train['hour_drop'] = train.dropoff_datetime.dt.hour

#
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]
train = train[(train.pickup_longitude> xlim[0]) & (train.pickup_longitude < xlim[1])]
train = train[(train.dropoff_longitude> xlim[0]) & (train.dropoff_longitude < xlim[1])]
train = train[(train.pickup_latitude> ylim[0]) & (train.pickup_latitude < ylim[1])]
train = train[(train.dropoff_latitude> ylim[0]) & (train.dropoff_latitude < ylim[1])]

# set init for animation
period = np.sort(train.hour_pick.unique())

def draw_day_night():
    # draw plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    #set up map
    m = Basemap(projection='merc',
                 llcrnrlat=train['dropoff_latitude'].min(),
                 urcrnrlat=train['dropoff_latitude'].max(),
                 llcrnrlon=train['dropoff_longitude'].min(),
                 urcrnrlon=train['dropoff_longitude'].max(),
                 resolution='f')
    m.drawmapboundary()

    #animation
    x, y = m(0, 0)
    point_pick = m.plot(x, y, '.', c="#1292db", alpha=0.2)[0]
    x1, y1 = m(0, 0)
    point_drop = m.plot(x1, y1, '.', c="#fd3096", alpha=0.2)[0]

    def animate(i):
        pickup = train[train.hour_pick==i]

        lon_pick = pickup['pickup_longitude'].values
        lat_pick = pickup['pickup_latitude'].values
        lon_drop = pickup['dropoff_longitude'].values
        lat_drop = pickup['dropoff_latitude'].values

        x, y = m(lon_pick, lat_pick)
        point_pick.set_data(x,y)
        x1, y1 = m(lon_drop, lat_drop)
        point_drop.set_data(x1,y1)

        fig.suptitle('New York City Taxi Trip at %2d:00' % (i), fontsize=18)

        # day/night imitation
        if i >= 3 and i <= 6:
            m.drawmapboundary(fill_color='#1e384d')
            m.fillcontinents(color = '#4d4d4d')
        elif i >= 7 and i <= 9:
            m.drawmapboundary(fill_color='#2c5170')
            m.fillcontinents(color = '#b3b3b3')
        elif i >= 10 and i <= 15:
            m.drawmapboundary(fill_color='steelblue')
            m.fillcontinents(color = '#bfbfbf')
        elif i >= 16 and i <= 18:
            m.drawmapboundary(fill_color='#2c5170')
            m.fillcontinents(color = '#b3b3b3')
        elif i >= 19 and i <= 22:
            m.drawmapboundary(fill_color='#1e384d')
            m.fillcontinents(color = '#4d4d4d')
        else:
            m.drawmapboundary(fill_color='#080045')
            m.fillcontinents(color='#191919')

        return point_pick, point_drop,

    output = animation.FuncAnimation(fig, animate, period, interval=500, repeat=True)
    output.save('day_night.gif', writer='imagemagick')


def draw_pick_drop():
    # draw plot
    fig = plt.figure(4, figsize=(20,10))
    fig.suptitle('New York City Taxi Trip', fontsize=18)

    #set up two map
    ax1 = fig.add_subplot(121)
    m1 = Basemap(projection='merc',
                 llcrnrlat=train['pickup_latitude'].min(),
                 urcrnrlat=train['pickup_latitude'].max(),
                 llcrnrlon=train['pickup_longitude'].min(),
                 urcrnrlon=train['pickup_longitude'].max(),
                 resolution='h')
    m1.fillcontinents(color='#000026')
    m1.drawmapboundary(fill_color='#030040')

    x, y = m1(0, 0)
    point_pick = m1.plot(x, y, '.', c="#1292db", alpha=0.2)[0]
    ax1.set(title="pickup")


    ax2 = fig.add_subplot(122)
    m2 = Basemap(projection='merc',
                 llcrnrlat=train['dropoff_latitude'].min(),
                 urcrnrlat=train['dropoff_latitude'].max(),
                 llcrnrlon=train['dropoff_longitude'].min(),
                 urcrnrlon=train['dropoff_longitude'].max(),
                 resolution='h')
    m2.fillcontinents(color='#000026')
    m2.drawmapboundary(fill_color='#030040')

    x1, y1 = m2(0, 0)
    point_drop = m2.plot(x1, y1, '.', c="#fd3096", alpha=0.2)[0]
    ax2.set(title="dropoff")

    #animation
    def animate(i):
        pickup = train[train.hour_pick==i]
        dropoff = train[train.hour_drop==i]

        lon_pick = pickup['pickup_longitude'].values
        lat_pick = pickup['pickup_latitude'].values
        lon_drop = dropoff['dropoff_longitude'].values
        lat_drop = dropoff['dropoff_latitude'].values

        x, y = m1(lon_pick, lat_pick)
        point_pick.set_data(x,y)

        x1, y1 = m2(lon_drop, lat_drop)
        point_drop.set_data(x1,y1)

        ax1.set(title='pickup at %2d:00' % (i))
        ax2.set(title='dropoff at %2d:00' % (i))

        return point_pick, point_drop,

    output = animation.FuncAnimation(fig, animate, period, interval=500, repeat=True)
    output.save('pick_drop.gif', writer='imagemagick')


if __name__ == '__main__':
    draw_day_night()
    draw_pick_drop()

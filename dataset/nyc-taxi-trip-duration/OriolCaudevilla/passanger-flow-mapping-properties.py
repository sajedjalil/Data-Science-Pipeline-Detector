import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import neighbors
from PIL import Image,ImageDraw

# LOAD NYC OUTLINE
#mask = np.load('NYmask.npy')

## KNN PARAMETERS:
n_neighbors =30 
#weights = 'distance'
weights = 'uniform'

## LOAD DATA:
fields = ['pickup_datetime','pickup_latitude','pickup_longitude','dropoff_datetime','dropoff_latitude','dropoff_longitude','trip_duration']
parse_dates = ['pickup_datetime','dropoff_datetime']
#df = pd.read_csv('~/kaggle/taxi/3march.csv',usecols=fields,parse_dates=parse_dates)
df = pd.read_csv('../input/train.csv',usecols=fields,parse_dates=parse_dates)
df.dropna(how='any',inplace=True)
dfsave = df

# DEFINE THE SPATIAL GRID
#ymax = 40.85
#ymin = 40.65
#xmin = -74.06
ymax = 40.8
ymin = 40.7
xmin = -74.02
xmax = xmin +(ymax-ymin)
X,Y = np.mgrid[xmin:xmax:512j,ymin:ymax:512j]
#X,Y = np.mgrid[xmin:xmax:256j,ymin:ymax:256j]
positions = np.vstack([X.ravel(),Y.ravel()])

# RECONSTRUCT THE MAP FOR EVERY TIME FRAME 
movie = []
Zs = []
time_step = 30 #time gates of 30 minutes
for h in range(24):
    print(h)
    for m in np.arange(0,60,time_step).astype(int):
        print(m)
        df = dfsave[dfsave.pickup_datetime.dt.weekday<5] # select only weekdays
        df = dfsave.groupby(dfsave.pickup_datetime.dt.hour).get_group(h)
        df = df[(df.pickup_datetime.dt.minute>=m) & (df.pickup_datetime.dt.minute<(m+time_step))]
        print(df.count())
        values_pickup = np.vstack([df.pickup_longitude.values,df.pickup_latitude.values])
        values_dropoff = np.vstack([df.dropoff_longitude.values,df.dropoff_latitude.values])
        values = np.hstack([values_pickup,values_dropoff])
        #values = np.hstack([values_dropoff])
        targets = np.hstack([np.ones((1,values_pickup.shape[1])),-np.ones((1,values_dropoff.shape[1]))])
        #targets = np.hstack([df.passenger_count.values,-df.passenger_count.values])
        #targets = np.hstack([df.trip_duration])
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        Z = np.reshape(knn.fit(values.T,targets.T).predict(positions.T),X.shape)
        Zs.append(Z)

Writer = animation.writers['imagemagick']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

#font = ImageFont.truetype("arial.ttf",32)
fig = plt.figure()
movie = []
h = 4
m = -time_step
for i in range(len(Zs)):
    m = m + time_step
    if m==60:
    	h = h+1
    	m = 0
    Z = 0.25*Zs[i-1]+0.5*Zs[i]   # temporal smoothing
    if (i+1)==len(Zs):
    	Z = Z+0.25*Zs[0]
    else:
    	Z = Z+0.25*Zs[i+1]
    #img = Image.fromarray(np.rot90(Z*mask))
    img = Image.fromarray(np.rot90(Z))
    draw = ImageDraw.Draw(img)
    draw.text((5,5),"%02d:%02d" % (h, m),1)
    Z = np.array(img)
    frame = plt.imshow(Z,extent=[xmin,xmax,ymin,ymax],clim = [-1, 1],cmap='RdBu',animated=True)
    movie.append([frame])

ani = animation.ArtistAnimation(fig, movie, interval=100, blit=False,repeat_delay=0)
ani.save('./animation_out.gif', writer=writer)


import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fi(df,pre):
    latlong = np.array([[p[0][1], p[0][0]] for p in df['POLYLINE'] if len(p)>0])
    # cut off long distance trips
    lat_low, lat_hgh = np.percentile(latlong[:,0], [2, 98])
    lon_low, lon_hgh = np.percentile(latlong[:,1], [2, 98])
    
    # create image
    bins = 513
    lat_bins = np.linspace(lat_low, lat_hgh, bins)
    lon_bins = np.linspace(lon_low, lon_hgh, bins)
    H2, _, _ = np.histogram2d(latlong[:,0], latlong[:,1], bins=(lat_bins, lon_bins))
    
    img = np.log(H2[::-1, :] + 1)
    
    plt.figure()
    ax = plt.subplot(1,1,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Taxi trip end points')
    plt.savefig(pre+"taxi_trip_end_points.png")
    return H2


# reading training data
zf = zipfile.ZipFile('../input/train.csv.zip')
df = pd.read_csv(zf.open('train.csv'), 
    nrows=2500,
    usecols =['POLYLINE'], 
    converters={'POLYLINE': lambda x: json.loads(x)})

h=fi(df,'train')


# reading training data
zf = zipfile.ZipFile('../input/train.csv.zip')
df = pd.read_csv(zf.open('train.csv'), 
    nrows=5000,
    skiprows=range(1,2500),
    usecols = ['POLYLINE'], 
    converters={'POLYLINE': lambda x: json.loads(x)})

h = fi(df,'test')
img = np.log(h[::-1, :] + 1)

plt.figure()
ax = plt.subplot(1,1,1)
plt.imshow(img)
plt.axis('off')
plt.title('Taxi trip end points diff')
plt.savefig("diff taxi_trip_end_points.png")

exit()

print("computing stats")
df["len"]=df["POLYLINE"].apply(lambda p: len(p))
df=df[df["len"]>0]
print ("num of trips"),
print (df["len"].count())
print ("mean trip length"),
print (df["len"].mean())
print ("median trip length"),
print (df["len"].median())
print ("trip length std"),
print (df["len"].std())
print ("trip length variance"),
print (df["len"].var())
print ("trip length min"),
print (df["len"].min())
print ("trip length max"),
print (df["len"].max())

df=df[df["len"]<200]
plt.figure()
df.plot(kind='hist', bins=100)
plt.title('Taxi trip len')
plt.savefig("taxi_trip_len.png")

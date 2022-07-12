import cv2
import io
import numpy as np
import random
import math
import csv as csv
import json
import zipfile

def getDist(posStart,posEnd):
    dispair = (posStart-posEnd)*[83.818,111.23] # distance in kilometers for longitude and latitude
    return np.sqrt(np.power(dispair[0],2)+np.power(dispair[1],2))

def drawTrips(image,data,border):
    minlat,maxlon,maxlat,minlon = border
    absLat = abs(minlat-maxlat)
    absLon = abs(minlon-maxlon)
    for i in range(len(data)):
        if (data[i][7] != False): # only if there are no missing gps datas
            gpsTrack = np.array(json.loads(data[i][8]))
            if len(gpsTrack) > 0: # the gpsTrack must be longer than 0s
                for p in range(len(gpsTrack)-1):
                    # get the correct px values for the current gps position
                    pxXS = np.rint((gpsTrack[p][0]-minlon)*(np.shape(image)[1]/absLon))-1
                    pxYS = np.rint((maxlat-gpsTrack[p][1])*(np.shape(image)[0]/absLat))-1

                    # and for the following one
                    pxXE = np.rint((gpsTrack[p+1][0]-minlon)*(np.shape(image)[1]/absLon))-1
                    pxYE = np.rint((maxlat-gpsTrack[p+1][1])*(np.shape(image)[0]/absLat))-1

                    # px is an array of all pixels that are between the current gps position and the following one
                    px = [[pxYS,pxXS]]
                    # calculate the shortest path between the current pos and the following one
                    while(px[-1] != [pxYE,pxXE]):
                        # calculate the distance between the last point and the end pos (the next real gps pos)
                        dX = pxXE-px[-1][1]
                        dY = pxYE-px[-1][0]
                        if abs(dY) < abs(dX):
                            if (dX > 0):
                                px.append([px[-1][0],px[-1][1]+1])
                            else:
                                px.append([px[-1][0],px[-1][1]-1])
                        else:
                            if (dY > 0):
                                px.append([px[-1][0]+1,px[-1][1]])
                            else:
                                px.append([px[-1][0]-1,px[-1][1]])

                    # calculate the distance of the current gpsTrack part (in km)
                    dist = getDist(gpsTrack[p],gpsTrack[p+1])
                    # the distance shouldn't be longer than 0.5 km cause that would be 120km/h
                    if (dist > 0.5):
                        break
                    else:
                        # save the distance for each pixel in the px array
                        for pi in range(len(px)):
                            pxY = px[pi][0]
                            pxX = px[pi][1]
                            # we are only analysing the points which are near the city of Porto (ignore the other ones)
                            if pxX >= 0 and pxX < np.shape(image)[1] and pxY >= 0 and pxY < np.shape(image)[0]:
                                image[pxY,pxX]+=[0,1,dist]
    return image

# generate a 4000 x 4000 image in RGB mode
image = np.zeros((4000,4000,3))

minlat = 41 # most southern part
minlon = -8.8
maxlat = 41.3 # most northern part
maxlon = -8.4

zf = zipfile.ZipFile("../input/train.csv.zip")
f = io.TextIOWrapper(zf.open("train.csv", "rU"))
r = csv.reader(f)
header = r.__next__()

data=[] 	# Create a variable to hold the data
# use only the first 5000 trips
for i in range(5000):
    row=r.__next__()
    data.append(row[0:]) 								# adding each row to the data variable
    
    if i%1000==999:
        data = np.array(data) 								
        # do the cool stuff here  
        image = drawTrips(image,data,[minlat,maxlon,maxlat,minlon])
        data = []

f.close()

old_settings = np.seterr(divide='ignore',invalid='ignore')

image[:,:,2] = np.nan_to_num(image[:,:,2]/image[:,:, 1])
image[:,:,1] = 0
image[:,:,0] = 0

# get the maximum accumlated distance and set that one to 255
maxF = np.max(image[:,:,2])
image[:,:,2] *= 255/maxF
image[:,:,1]  = image[:,:,2]
image[:,:,0] = image[:,:,2]
image[:,:,1][image[:,:,1] < 50] = 0 # really slow => change to red
image[:,:,0][image[:,:,0] < 150] = 0 # kind of slow => orange/yellow
cv2.imwrite("image.png", image)
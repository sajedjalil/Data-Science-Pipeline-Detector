# Copy and Run code in your local machine with your own path. Images will not be resized. Resize later in the challenge part!
# Dataset has 1048575 images(~22GB) close and rerun code anytime to resume download


import numpy as np 
import pandas as pd 
import sys, requests, shutil, os
from urllib import request, error
# Input data files are available in the "../input/" directory.
print(os.listdir("../input"))
data=pd.read_csv('../input/train.csv')
data.head(5)

def fetch_image(path):
    url=path
    response=requests.get(url, stream=True)
    with open('../input/train_images/image.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
links=data['url']
i=0

for link in links:              #looping over links to get images
    if os.path.exists('../input/train/'+str(i)+'.jpg'):
        i+=1
        continue
    fetch_image(link)
    os.rename('../input/train_images/image.jpg','../input/train_images/'+ str(i)+ '.jpg')
    i+=1
    #if(i==15):   #uncomment to test in your machine
    #    break
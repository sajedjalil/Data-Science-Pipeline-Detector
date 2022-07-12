#Library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/train_v2.csv")
tags = df["tags"].apply(lambda x: x.split(' '))
   
end = len(tags)
id_haze = []
id_cloudy = []
id_partly = []
id_clear = []

for i in range (0,end):
    for x in tags[i]:
        if x == 'haze':
            id_haze.append(i)
        elif x == 'cloudy':
            id_cloudy.append(i)
        elif x == 'partly_cloudy':
            id_partly.append(i)
        elif x == 'clear':
            id_clear.append(i)
print (len(id_haze))
print (len(id_cloudy))
print(len(id_partly))
print (len(id_clear))

import cv2
import matplotlib.pyplot as plt
import random

new_style = {'grid': True}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))
index = []
for i in range(0,9):
 
    if i <3:
        l = random.choice(id_cloudy)
        index.append(l)
    elif (i>=3 and i<6):
        l = random.choice(id_partly)
        index.append(l)
    elif (i>=6 and i<9):
        l = random.choice(id_haze)
        index.append(l)
    
    img = cv2.imread('../input/train-jpg/train_'+str(l)+'.jpg')
    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()
print (index)

import cv2
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
l = random.choice(id_cloudy)
im = cv2.imread('../input/train-jpg/train_'+str(l)+'.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]
    
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.hist(r.ravel(), bins=256, range=(0., 255))
plt.hist(g.ravel(), bins=256, range=(0., 255))
plt.hist(b.ravel(), bins=256, range=(0., 255))
plt.show()
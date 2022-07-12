#Python 2.7 script
#Download and save the last downloaded image
#re-run and resume from the last dwonloaded image

import os
import requests
import json
import shutil
train_data = json.load(open('train.json'))

images_arr = train_data['images']
annotations_arr = train_data['annotations']
os.makedirs('train_data')

last = None
with open('last.txt','r') as l:
    last = int(l.readline())
    print type(last)
    print last
l.close()

for img in images_arr[last:]:
    try:
        r = requests.get(img['url'][0],timeout=10)
        folder_path = 'train_data/'
        with open(folder_path+str(img['image_id'])+'.jpg','wb') as f:
            f.write(r.content)
        f.close()

        with open('last.txt','w') as l:
            l.write(str(img['image_id']))
        l.close()        
    except Exception as e:
        print e
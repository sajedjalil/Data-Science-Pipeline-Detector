"""
Created on Fri Feb 26 16:06:40 2016

@author: Run2
@modified : vrajs5
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import os
import io
from PIL import Image
import datetime
from multiprocessing import Process

os.chdir('../input')

def dhash(image,hash_size = 16):
    image = image.convert('LA').resize((hash_size+1,hash_size),Image.ANTIALIAS)
    difference = []
    for row in xrange(hash_size):
        for col in xrange(hash_size):
            pixel_left = image.getpixel((col,row))
            pixel_right = image.getpixel((col+1,row))
            difference.append(pixel_left>pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index%8)
        if (index%8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))
            decimal_value = 0
    
    return ''.join(hex_string)

    
def StartHashing(zip_counter):
    imgzipfile = zipfile.ZipFile('Images_'+str(zip_counter)+'.zip')
    print ('Doing zip file ' + str(zip_counter))
    namelist = [i for i in imgzipfile.namelist() if '.jpg' in i]
    print ('Total elements ' + str(len(namelist)))

    img_id_hash = []
    counter = 1
    for name in namelist:
        #print name
        try:
            imgdata = imgzipfile.read(name)
            if len(imgdata) >0:
                img_id = name[:-4]
                stream = io.BytesIO(imgdata)
                img = Image.open(stream)
                img_hash = dhash(img)
                img_id_hash.append([img_id,img_hash])
                counter+=1
            if counter%10000==0:
                print ('Done ' + str(counter) + datetime.datetime.now())
        except:
            print ('Could not read ' + str(name) + ' in zip file ' + str(zip_counter))
    df = pd.DataFrame(img_id_hash,columns=['image_id','image_hash'])
    df.to_csv('image_hash_' + str(zip_counter) + '.csv')

    
paths = [0,1,2,3,4,5,6,7,8,9]
if __name__ == '__main__':
    procs = []
    for pt in paths:
        p = Process(target = StartHashing, args=(pt,))
        procs.append(p)
        
    for p in procs:
        p.start()
        
    for p in procs:
        p.join()
        
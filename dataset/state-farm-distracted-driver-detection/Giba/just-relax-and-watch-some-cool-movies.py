# -*- coding: utf-8 -*-
"""
@author: Giba1
"""
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
from skimage import io, transform
import matplotlib.animation as animation

train = pd.read_csv( '../input/driver_imgs_list.csv' )
train['id'] = range( train.shape[0] )
fig = plt.figure()
subj = np.unique( train['subject'])[0]

for subj in np.unique( train['subject'])[:2]:

    imagem = train[ train['subject']==subj ]
    
    imgs = []
    t = imagem.values[0]
    for t in imagem.values:
        img = cv2.imread('../input/train/'+t[1]+'/'+t[2],3)
        img = cv2.resize(img, (160, 120))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append( img )
        
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1,wspace=None, hspace=None)  # removes white border
    fname = 'MOVIE_subject_'+subj+'.gif'
    imgs = [ (ax.imshow(img), 
              ax.set_title(t[0]), 
              ax.annotate(n_img,(5,5))) for n_img, img in enumerate(imgs) ] 
    img_anim = animation.ArtistAnimation(fig, imgs, interval=125, 
                                repeat_delay=1000, blit=False)
    print('Writing:', fname)
    img_anim.save(fname, writer='imagemagick', dpi=20)
    fig.clf()
print ('Now relax and watch some movies!!!')


# Any results you write to the current directory are saved as output.
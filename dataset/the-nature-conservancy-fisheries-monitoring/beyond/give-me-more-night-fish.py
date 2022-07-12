### This kernel helps create more night fish by shifting the hue channel to 
### make the images look greener
### More details on https://github.com/netbeast/colorsys

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import colorsys

from PIL import Image
import matplotlib
matplotlib.use("Agg")

####################################
im = Image.open("../input/train/ALB/img_00003.jpg")
ld = im.load()
width, height = im.size
for y in range(height):
    for x in range(width):
        r,g,b = ld[x,y]
        h,s,v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
        h = (h + -90.0/360.0) % 1.0   # hue
        s = s**0.25                   # saturation
        r,g,b = colorsys.hsv_to_rgb(h, s, v)
        ld[x,y] = (int(r * 255.9999), int(g * 255.9999), int(b * 255.9999))
        ld.show()
####################################
# To save the image:
# ld.save(yourpath + '.jpg')

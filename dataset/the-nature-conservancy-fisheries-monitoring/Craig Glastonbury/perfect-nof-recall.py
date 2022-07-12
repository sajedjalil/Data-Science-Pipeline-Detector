# The following script will take 4 random crops from each of the NoF images.
# Feeding a CNN bounding boxes of fish and these random crops achieves excellent NoF recall

import random, os, time
from PIL import Image

INPATH = r"NoF"
OUTPATH = r"NoF_fish"

dx = dy = 299
tilesPerImage = 4

files = os.listdir(INPATH)
numOfImages = len(files)


# img_04052_resized.jpg
t = time.time()
for file in files:
   with Image.open(os.path.join(INPATH, file)) as im:
     for i in range(1, tilesPerImage+1):
       newname = file.split('_r')[0]+'_box'+str(i)+'.jpg'
       w, h = im.size
       x = random.randint(0, w-dx-1)
       y = random.randint(0, h-dy-1)
       print("Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
       im.crop((x,y, x+dx, y+dy))\
         .save(os.path.join(OUTPATH, newname))

t = time.time()-t
print("Done {} images in {:.2f}s".format(numOfImages, t))
print("({:.1f} images per second)".format(numOfImages/t))
print("({:.1f} tiles per second)".format(tilesPerImage*numOfImages/t))
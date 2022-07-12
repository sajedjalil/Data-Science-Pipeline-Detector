import matplotlib.pyplot as plt
import glob, os

ultrasounds = [img for img in glob.glob("../input/train/*.tif") if 'mask' not in img]
    
for file in ultrasounds[0:1]:
    im = plt.imread(file)
    plt.figure(figsize=(15,20))
    plt.imshow(im, cmap="Greys_r")
    plt.show()

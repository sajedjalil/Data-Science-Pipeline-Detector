# Hi all,
# Please note that this is my first Kaggle kernel ;)
# The idea of this approach is to use color distances to convert the rgb-image
# to a grey-scale image. Using this image one can make use of finding local maxima
# to find the dots.

# This script is somehow careful to not produce FPs. So not all TPs are found.
# Maybe you have any ideas to improve this? Feel free to get in touch!

# Have fun and any helpful comment is appreciated

# Sometimes "check_output" does not list all files present in the directory. Why?

# Expected output is something like:
#0.9s
#0
#0.jpg
#1.jpg
#10.jpg
#2.jpg
#3.jpg
#4.jpg
#41.jpg
#42.jpg
#43.jpg
#44.jpg
#45.jpg
#46.jpg
#47.jpg
#48.jpg
#49.jpg
#5.jpg
#50.jpg
#6.jpg
#7.jpg
#8.jpg
#9.jpg

#1.1s
#1
#Fontconfig warning: ignoring C.UTF-8: not a valid language tag
#2.4s
#2
#41.jpg
#10.9s
#3
#Expected: 177, actual: 146
#42.jpg
#18.1s
#4
#Expected: 22, actual: 20
#43.jpg
#32.1s
#5
#Expected: 606, actual: 518
#44.jpg
#39.3s
#6
#Expected: 45, actual: 37
#45.jpg
#47.2s
#7
#Expected: 138, actual: 117
#46.jpg
#56.1s
#8
#Expected: 5, actual: 10
#47.jpg
#65.6s
#9
#Expected: 113, actual: 110
#48.jpg
#72.9s
#10
#Expected: 105, actual: 94
#49.jpg
#79.2s
#11
#Expected: 19, actual: 10
#50.jpg
#85.5s
#12
#Expected: 1, actual: 1






# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/TrainDotted"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from skimage import color, io, img_as_float
from skimage.feature import corner_peaks
from PIL import Image, ImageFont, ImageDraw

# counts extracted from the csv file
expected_counts = {
    "41": 15 + 0 + 85 + 18 + 59,
    "42": 7 + 4 + 10 + 1 + 0,
    "43": 28 + 4 + 338 + 47 + 189,
    "44": 3 + 2 + 25 + 15 + 0,
    "45": 4 + 7 + 100 + 27 + 0,
    "46": 1 + 4 + 0 + 0 + 0,
    "47": 13 + 16 + 48 + 3 + 33,
    "48": 5 + 10 + 66 + 24 + 0,
    "49": 0 + 0 + 4 + 15 + 0,
    "50": 1 + 0 + 0 + 0 + 0
    }
    
# colors extracted using paint - maybe there are more accurate color values - please let me know    
colors = {
    "red": np.array([231, 8, 9]),
    "green": np.array([56, 161, 33]),
    "brown": np.array([87, 46, 10]),
    "blue": np.array([37, 51, 141]),
    "magenta": np.array([244, 8, 242]),
    }
    
path_dotted = "../input/TrainDotted"

# just converting the list to a tuple after normalization
def color_normalized(name):
    return tuple(colors[name]/255.)

# iterate over the files with known expected counts
for file in expected_counts.keys():
    filename = file + ".jpg"
    print(filename)
    
    # load the file for calculations
    image_dotted = img_as_float(io.imread(path_dotted + "/" + filename))
    
    # load the file for drawing
    image_dotted_pil = Image.open(path_dotted + "/" + filename)
    dr = ImageDraw.Draw(image_dotted_pil)
    coordinates_count = 0
    
    # iterate over each color
    for color_key in colors:
        # calculate the distances of each pixel to the current color
        distances = color.rgb2gray(1 - np.abs(image_dotted - color_normalized(color_key)))
        
        # throw away all non-maximas
        distances[distances<0.99] = 0
        
        # calculate the peaks 
        coordiantes = corner_peaks(distances, threshold_rel=0.95, min_distance=20)
        
        # draw rectangles on images
        coordinates_count += len(coordiantes)
        for coordinate in coordiantes:
            dr.rectangle(((coordinate[1]-20,coordinate[0]-20),(coordinate[1]+20,coordinate[0]+20)), outline = tuple(colors[color_key]))

    # compare the count
    print("Expected: " + str(expected_counts[file]) + ", actual: " + str(coordinates_count))
    
    
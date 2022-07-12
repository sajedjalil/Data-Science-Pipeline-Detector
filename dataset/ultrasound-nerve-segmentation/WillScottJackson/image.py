import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import glob, os

ultrasounds = [img for img in glob.glob("../input/train/*.tif") if 'mask' not in img]

import cv2
img = cv2.imread(ultrasounds[0])
cv2.imwrite('img.png', img)
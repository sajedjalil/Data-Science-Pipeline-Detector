#! /usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

for bgtype in range(4):
    bgimg = np.max(np.asarray([cv2.imread('../input/test/{}.png'.format(n)) for n in range(73+3*bgtype, 217, 12)]), axis=0)
    cv2.imwrite('bgimg{}.png'.format(bgtype), bgimg)

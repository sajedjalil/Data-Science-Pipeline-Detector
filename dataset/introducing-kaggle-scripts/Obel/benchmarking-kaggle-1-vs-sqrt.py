# There did seem to be a general agreement from other tests, but that seemed to vary from python version to version.
# http://stackoverflow.com/questions/327002/which-is-faster-in-python-x-5-or-math-sqrtx

import time
import math

def timeit1():
    s = time.time()
    for i in range(7500000):
        z=i**.5
    print (time.time() - s)

def timeit2(arg=math.sqrt):
    s = time.time()
    for i in range(7500000):
        z=arg(i)
    print (time.time() - s)

timeit1()
timeit2()

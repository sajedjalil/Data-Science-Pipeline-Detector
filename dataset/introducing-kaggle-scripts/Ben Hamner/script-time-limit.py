import sys
import time 

for i in range(10000):
    print("Minute %d" % i)
    sys.stdout.flush()
    time.sleep(60)

from threading import Thread
import threading
import multiprocessing
from queue import Queue, Empty
import time, os
import logging
startT = time.time()
class Test(multiprocessing.Process):
    def __init__(self, id, num):
        self._num = num
        self._id = id
    def run(self):
        self._test()
    def _test(self, ):
        for i in range(self._num):
            aaa = i*i
        print("[{}] {} done".format(time.time()-startT, self._id))
def test_process():
    num_thr = 4#
    tot = 1e9
    num = int(tot/num_thr)
    print("num is {}".format(num))
    threads = []
    for i in range(num_thr):
        #thread = Thread(target = test, name = str(i), args = (num,))
        thread = Test(i,num)
        #thread.setDaemon(True)
        thread.run()
        threads.append(thread)
    time.sleep(10)

def test_util():
    import util
    util.test()



test_process()
# ----------------------------------------------------------------------------
# primes.py 
#
# This python script attempts to find as many primes as possible
# within 10 seconds. It uses a segmented Sieve of Erathosthenes 
# and the Python multiprocessing library. Seperate processes work on 
# different segments of the sieve. Primes are written to a .CSV file. 
#
# This script can find all primes <1 billion (about 50M of them) in much 
# less than 10 seconds (typically 3 to 4 sec) running on Kaggle Scripts. 
# However, to stay within the Kaggle Scripts 512MB disk limit,
# only primes below about 945M can be found (48M primes in all).
#
# Initialy written by Chris Hefele July, 2015
# #ND WhizWilde: this is a different attempt at optimizing the sieve
# ---------------------------------------------------------------------------


import numpy
import math
import sys
from time import time
from contextlib import contextmanager
import multiprocessing
import ctypes
#import pickle #didn't do any good

def sieve_primes(n):

    # Find all primes n > prime > 2 using the Sieve of Eratosthenes 
    # For efficiency, track only odd numbers (evens are nonprime)

    sieve = numpy.ones(n/2, dtype=numpy.bool) 
    #limit = int(math.sqrt(n)) + 1 
    limit= math.ceil(math.sqrt(n)) 
    
    #sieve[4 :: 3]=False
    #sieve[7 :: 5]=False
    #sieve[4 :: 3]=False
    #Tests showed it didn't improved processsing time, but rather degraded it by more than 4 seconds? May it be that the fact of skipping the first two steps of i degrade time so much? I will test changing the lower bound for i
    
    
    for i in range(3, limit, 2): 
        #print(str(doublei))
        if sieve[i/2] :
    #integer division or math.floor doesn't seem to work better -_-  
    
            sieve[i*i/2 :: i] = False
            
            #an attempt to increase speed, I am not sure if I am right about pair steps but I believe they are useless, I can't get my maths on it.
            
            
    prime_indexes = numpy.nonzero(sieve)[0][1::]
    #prime_indexes =sieve[numpy.where(sieve!=False)]# 22,355 s !:(
    
  #  direct_primes  = [(2 * index+ 1) for index, element in enumerate(sieve) if element is not False]
   #I'll use a comprehension instead of numpy method
   #seems I don't need .astype(ctypes.c_uint) anymore with a list comprehension, since enumerate/where returns ints
   #str can't be used here, too much memory
    #direct_primes=[(2 * index+ 1) for index in sieve[numpy.where(sieve!=False)]]
   #seems it is very slow like that, so I will make another try:
    #prime_indexes =sieve[numpy.where(sieve!=False)]
    #prime_indexes=sieve[sieve !=False] #other alternative
    #prime_indexes=sieve[sieve !=0]
    
    #primes=2*prime_indexes.astype(ctypes.c_uint)+1#for original nonzero method
    primes=2*prime_indexes.astype(ctypes.c_uint)+1
    #firstprime=primes[0].astype(ctypes.c_uint)
    firstprime=primes[0]
    #lastprime=primes[-1].astype(ctypes.c_uint)
    lastprime=primes[-1]
    prime_found=str(len(primes))
    print ("First prime found: "+ str(firstprime) )
    print ("Last prime found: "+ str(lastprime) )
    print ("Number of primes found: "+ prime_found )
    #Used to evaluate how far can it get 
    
    return primes
    
    #prime_array=numpy.nonzero(sieve)[0]#includes 1 which is not a prime, in order to avoid a costly slice operation
    #last_prime_index = prime_array[-1]
    #lastprime  = str(2 * last_prime_index.astype(ctypes.c_uint) + 1 )
    #primes_found=str(len(prime_array)-1)#corrects the length error (len is good, but there is 1 which is not a prime)
    #print ("Last prime found: "+ lastprime )
    #print ("Number of primes found: "+ primes_found )
    #return lastprime
#This is an attempt to be faster by getting rid of the last array operations, just keeping the last prime





def ints_tostring(ints_arr, line_chars=10):

    # Converts an array of ints to ASCII codes in a bytestring.
    # This is ugly but faster than numpy's .tofile()

    buf  = numpy.zeros(shape=(len(ints_arr), line_chars), dtype=numpy.int8) 
    buf[:, line_chars-1] = 10   # 10 = ASCII linefeed
    for buf_ix in range(line_chars-2, 0-1, -1):
        numpy.mod(ints_arr, 10, out=buf[:, buf_ix])
        buf[:, buf_ix] += 48    # 48 = ASCII '0'
        ints_arr /= 10        
    return buf.tostring()



def singleprocess_calc_primes(n, file_name):
    fout = open(file_name, "wb")
    prime_string = ints_tostring(sieve_primes(n))
    #prime_string = (sieve_primes(n)) #now this is an array of strings
    fout.write(prime_string)
    #pickle.dump(sieve_primes(n), fout)
    fout.close()





@contextmanager
def timer(label):
    # timer(), as suggested by Sang Han 
    output = '{label}: {time:03.3f} sec'
    start = time()
    try:
        yield
    finally:
        end = time()
    print(output.format(label=label, time=end-start))
    sys.stdout.flush()


def main():

    # Using 945M as a bound yields a CSV of primes that is 
    # just under the 512MB Kaggle Scripts disk limit
    #mil = 1000000
    #upper_bound = (1000 - 55)*mil  
    #upper_bound=1000000000000 #Too much
    #upper_bound=5500000000 #7*10^9 Too much too,5*10^9 was ok, 
    upper_bound=10000000
    file_singleprocess = 'primes_singleprocess.csv'

    n_CPUs = multiprocessing.cpu_count()
    n_processes  = n_CPUs

    print("\n>>> PRIME NUMBER CALCULATION <<<")
    print("\nNumber of CPUs detected:", n_CPUs)
    print("Now finding all primes less than :", upper_bound)

   
    with timer("Single-process primes calculation"):
       singleprocess_calc_primes(upper_bound, file_singleprocess)
       # sieve_primes(upper_bound)
        # I will use this instead to see how far the script can go when it doesn't have to build a file
    print()


if __name__ == '__main__':
    main()
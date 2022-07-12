from numpy import savetxt, flatnonzero, ones, linspace, bool, uint64, frombuffer, ulonglong, int8, zeros, mod
from multiprocessing import Pool
from multiprocessing.sharedctypes import RawArray
from math import sqrt, exp, log, ceil, floor
from ctypes import c_bool
from contextlib import closing

def initSieve(sharedArray):
    global primeArray
    primeArray = frombuffer(sharedArray, dtype=bool)
    primeArray[:] = True
    primeArray[0] = False
    #primes = [0,1,1,1,0,1,1,0,1,1,0,1,0,0,1]
    #primeArray[:startingPrimes] = primes[:startingPrimes]

def segsieve(inTuple):
    
    tIdx, sIdx, eIdx = inTuple
    #sqeIdx = sqrt(2*eIdx+1)
    
    for ii in range(3, tIdx, 2):#(primeArray[:tIdx]):
        if primeArray[i//2]:
            primeArray[sIdx//2:eIdx//2] = False
    '''
        # If it is not prime, skip it
        if not ii: continue

        # Actual number
        nVal = 2*nIdx + 1
        
        # Break if we're above the upper limit for a factor
        if nVal > sqeIdx: break

        # Find the offset from the start point
        off = (nIdx - sIdx%nVal)
        
        # Set all multiples as non-prime
        primeArray[sIdx + off:eIdx:nVal] = 0
    '''

if __name__ == '__main__':
    global PrimeArray
    numproc = 16

    m = 10
    po = 8
    mult = 1
    hl = mult*m**po

    #targetPo = 16
    #base = floor(exp(log(hl)/targetPo))
    #base += base%2
    #print("base", base)

    #rns = Array(c_bool, hl//2)
    rns = RawArray(c_bool, hl//2)
    
    cIdx = int(sqrt(hl))
    cIdx = cIdx + (1 - cIdx%2)

    #pownew = 1
    #powold = 1

    #start = base

    #pownew = powold
    
    #primes = ones(hl//2, dtype=bool)
    #primes[0] = 0
    
    #for i in range(3,int(sqrt(hl)),2):
    #    if primes[i//2]:
    #        primes[i**2//2::i] = 0
    
    
    with closing(Pool(initializer=initSieve, initargs=(rns,))) as p:
        
        #while base**pownew < hl:
            
        #powold = pownew
        #pownew *= 2
        
        sliceIdx = linspace(cIdx, hl, numproc + 1, endpoint=True, dtype=uint64)
        print(sliceIdx)
        
        slices = tuple((sliceIdx[i], sliceIdx[i+1]) for i in range(numproc))
        sieveargs = [(cIdx, s[0] + (1 - s[0]%2), s[1] + (1 - s[1]%2)) for s in slices]

        #p.map(segsieve, sieveargs)
        p.imap_unordered(segsieve, sieveargs)
    
    
    #savetxt("out.csv", (2*flatnonzero(frombuffer(rns, dtype=bool))+1), delimiter=",", fmt="%0u")
    primeArray = frombuffer(rns, dtype=bool)
    primeArray = 2*flatnonzero(primeArray) + 1
    #savetxt("out.csv", (flatnonzero(primeArray)), delimiter=",", fmt="%u")
    line_chars = 10
    buf  = zeros(shape=(len(primeArray), line_chars), dtype=int8) 
    buf[:, line_chars-1] = 10   # 10 = ASCII linefeed
    for buf_ix in range(line_chars-2, 0-1, -1):
        mod(primeArray, 10, out=buf[:, buf_ix])
        buf[:, buf_ix] += 48    # 48 = ASCII '0'
        primeArray /= 10        
    #return buf.tostring()
    
    with open("out.csv", 'wb') as f:
        f.write(buf.tostring())

    print("Done!")
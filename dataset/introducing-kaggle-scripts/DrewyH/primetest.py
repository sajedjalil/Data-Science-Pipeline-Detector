from numpy import ones, arange, bool, uint32
import cProfile

def run(hl):
    nos = arange(3, hl, 2, dtype=uint32) #from blog for uint32
    ps = ones(len(nos), dtype=bool)#nes(int(hl/(log(hl)-4)), dtype=bool)
    
    for i,n in enumerate(nos):
        if not ps[i]:
            continue
    
        ps[i+n::n] = False
    
    #print(len(nos[ps]))
    
cProfile.run('run(10**7)')
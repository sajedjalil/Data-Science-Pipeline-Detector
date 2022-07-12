from heapq import heappush as add, heappop as minim

heap=[]
root = 2003;
upbound = root*root;
cnt =2;
add(heap, (9, 6))
with open ("primes.csv", 'w') as f:
    f.write('primes\n2\n3\n');

    i = 5

    while  (i<upbound):
        if (heap[0][0] == i):
            while (heap[0][0] == i):
                extr = minim(heap)
                add(heap, (extr[0]+extr[1], extr[1]))
        else:
            ++cnt
            f.write('%d\n'%i);
            if  (i<root):
                add(heap, (i*i, 2*i))
                
        i+=2
print('%d primes found\n'%cnt);
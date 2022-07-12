numhorse = 1000
numball = 1100
numbike = 500
numtrain = 1000
numbook = 1200
numdoll = 1000
numblocks = 1000
numgloves = 200

outfile = open( 'submission_naive2.csv', 'w' )
outfile.write( 'Gifts\n' )
bags = 0

for bags in range(bags,1000):
    if (numball>=23):
        s = ''
        for z in range(23):
            s = s + 'ball_%d' % (numball-1) + ' '
            numball -= 1
        outfile.write( s+'\n' )
    else:
        break
    
for bags in range(bags,1000):
    if (numhorse>=8):
        s = ''
        for z in range(8):
            s = s + 'horse_%d' % (numhorse-1) + ' '
            numhorse -= 1
        outfile.write( s+'\n' )
    else:
        break
    
for bags in range(bags,1000):
    if (numbook>=2) and (numblocks>=3):
        s = ''
        for z in range(2):
            s = s + 'book_%d' % (numbook-1) + ' '
            numbook -= 1
        for z in range(3):
            s = s + 'blocks_%d' % (numblocks-1) + ' '
            numblocks -= 1
        outfile.write( s+'\n' )
    else:
        break

for bags in range(bags,1000):
    if (numdoll>=7) and (numbook>=2):
        s = ''
        for z in range(7):
            s = s + 'doll_%d' % (numdoll-1) + ' '
            numdoll -= 1
        for z in range(2):
            s = s + 'book_%d' % (numbook-1) + ' '
            numbook -= 1
        outfile.write( s+'\n' )
    else:
        break    

for bags in range(bags,1000):
    if (numbook>=3) and (numtrain>=3):
        s = ''
        for z in range(3):
            s = s + 'book_%d' % (numbook-1) + ' '
            numbook -= 1
        for z in range(3):
            s = s + 'train_%d' % (numtrain-1) + ' '
            numtrain -= 1
        outfile.write( s+'\n' )
    else:
        break    
 
for bags in range(bags,1000):
    if (numtrain>=3) and (numgloves>=5):
        s = ''
        for z in range(3):
            s = s + 'train_%d' % (numtrain-1) + ' '
            numtrain -= 1
        for z in range(5):
            s = s + 'gloves_%d' % (numgloves-1) + ' '
            numgloves -= 1
        outfile.write( s+'\n' )
    else:
        break    

for bags in range(bags,1000):
    if (numtrain>=4):
        s = ''
        for z in range(4):
            s = s + 'train_%d' % (numtrain-1) + ' '
            numtrain -= 1
        outfile.write( s+'\n' )
    else:
        break 

for bags in range(bags,1000):
    if (numbike>=2) and (numball>=1):
        s = ''
        for z in range(2):
            s = s + 'bike_%d' % (numbike-1) + ' '
            numbike -= 1
        for z in range(1):
            s = s + 'ball_%d' % (numball-1) + ' '
            numball -= 1
        outfile.write( s+'\n' )
    else:
        break   
outfile.close()


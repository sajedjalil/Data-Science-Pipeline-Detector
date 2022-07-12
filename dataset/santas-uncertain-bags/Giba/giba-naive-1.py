qtty = []
qtty.append( 1100 )#ball
qtty.append( 1000 )#horse
qtty.append( 1000 )#doll
qtty.append( 200 )#gloves
qtty.append( 1000 )#blocks
qtty.append( 1200 )#book
qtty.append( 1000 )#train
qtty.append( 500 )#bike

perbag = []
perbag.append( 23 )#ball
perbag.append( 8 )#horse
perbag.append( 8 )#doll
perbag.append( 28 )#gloves
perbag.append( 3 )#blocks
perbag.append( 19 )#book
perbag.append( 4 )#train
perbag.append( 2 )#bike

BAG = []
s = ['ball']*perbag[0]
BAG.append(" ".join(s)+'\n')
s = ['horse']*perbag[1]
BAG.append(" ".join(s)+'\n')
s = ['doll']*perbag[2]
BAG.append(" ".join(s)+'\n')
s = ['gloves']*perbag[3]
BAG.append(" ".join(s)+'\n')
s = ['blocks']*perbag[4]
BAG.append(" ".join(s)+'\n')
s = ['book']*perbag[5]
BAG.append(" ".join(s)+'\n')
s = ['train']*perbag[6]
BAG.append(" ".join(s)+'\n')
s = ['bike','bike','ball']#not allowed less than 2 gifts per bag
BAG.append(" ".join(s)+'\n')

outfile = open( 'submission_naive1.csv', 'w' )
outfile.write( 'Gifts\n' )
bags = 0

while (qtty[7]>=perbag[7]) and (qtty[0]>0) and (bags<56):
    qtty[0] -= 1
    qtty[7] -= perbag[7]
    outfile.write( BAG[7] )
    bags+=1

for g in range(7):
    while (qtty[g]>=perbag[g]) and (bags<1000):
        qtty[g] -= perbag[g]
        outfile.write( BAG[g] )
        bags+=1
        
outfile.close()

counter = 0

with open("../input/trainSearchStream.tsv") as infile:
    for line in infile:
        counter += 1

print ('Total number of records = ' + str(counter))
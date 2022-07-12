from csv import DictReader
from collections import defaultdict
from datetime import datetime

start = datetime.now()

destination_clusters = defaultdict(set)

for i, row in enumerate(DictReader(open("../input/train.csv"))):
    if row["orig_destination_distance"] != '':
        destination_clusters[(row["user_location_city"],row["orig_destination_distance"])].add(int(row["hotel_cluster"]))
 
    if i % 1000000 == 0:
        print("%s\t%s"%(i, datetime.now() - start))
    

with open("sub.csv", "w") as outfile:
	outfile.write("id,hotel_cluster\n")
	for i, row in enumerate(DictReader(open("../input/test.csv"))):
		outfile.write("%d,%s\n"%(i,' '.join(str(s) for s in destination_clusters[(row["user_location_city"],row["orig_destination_distance"])])))
		if i % 1000000 == 0:
			print("%s\t%s"%(i, datetime.now() - start))  
from csv import DictReader
from collections import defaultdict
from datetime import datetime

start = datetime.now()

def get_top5(d):
    return " ".join(sorted(d, key=d.get, reverse=True)[:5])

destination_clusters = defaultdict(lambda: defaultdict(int))

for i, row in enumerate(DictReader(open("../input/train.csv"))):
	destination_clusters[row["srch_destination_id"]][row["hotel_cluster"]] += 1
	if i % 1000000 == 0:
		print("%s\t%s"%(i, datetime.now() - start))

most_frequent = defaultdict(str)

for k in destination_clusters:
	most_frequent[k] = get_top5(destination_clusters[k])

with open("pred_sub.csv", "w") as outfile:
	outfile.write("id,hotel_cluster\n")
	for i, row in enumerate(DictReader(open("../input/test.csv"))):
		outfile.write("%d,%s\n"%(i,most_frequent[row["srch_destination_id"]]))
		if i % 1000000 == 0:
			print("%s\t%s"%(i, datetime.now() - start))
from collections import defaultdict
from csv import DictReader

clusters_per_user_id = defaultdict(lambda: defaultdict(int))
popular_clusters = defaultdict(int)

for e, row in enumerate(DictReader(open("../input/train.csv"))):
	clusters_per_user_id[ row["user_id"] ][ row["hotel_cluster"] ] += 1
	popular_clusters[ row["hotel_cluster"] ] += 1
	if e % 100000 == 0:
		print(e)

fixed_guesses = ""
for w in sorted(popular_clusters, key=popular_clusters.get, reverse=True)[:5]:		
	fixed_guesses += " %s"%w
		
with open("submission.csv", "w") as outfile:
	outfile.write("id,hotel_cluster\n")
	for e, row in enumerate(DictReader(open("../input/test.csv"))):
		user_favs = clusters_per_user_id[ row["user_id"] ]
		guesses = ""
		for w in sorted(user_favs, key=user_favs.get, reverse=True)[:5]:
			guesses += " %s"%w
		if guesses == "":
			guesses = fixed_guesses
		outfile.write( "%s,%s\n"%(e, guesses) )
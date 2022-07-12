# Adds proper .csv structure for Page column.
# 
# Some Page column values contain `"` or `,` without proper escaping.
# This script fixes that.

with open("train_2.csv", "w") as outfile:
	for i, line in enumerate(open("../input/train_1.csv")):
		r = line.strip().split(",")
		line = ','.join(['"%s"'%(','.join(r[:-550]).replace('"', '""'))] + r[-550:]) + '\n'
		outfile.write(line)
		if i % 10000 == 0:
			print(i)
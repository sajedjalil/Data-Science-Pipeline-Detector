# Are the ids sorted?
def is_sorted(f):
    prev = -float("inf")
    with open(f, "r") as fin:
        fin.readline()
        for linenum, l in enumerate(fin):
            if linenum % 1000 == 0:
                print("processing line {}".format(linenum))
            idx = int(l.split(",")[0])
            if idx < prev:
                return False
    return True, linenum
    
s = []
for f in ["../input/train_numeric.csv", "../input/train_categorical.csv"]:
    truth, linenum = is_sorted(f)
    s.append((truth, linenum))
print(s)

"""
This result is useful, because now we can easily construct all flow paths.
"""

# A related question: are the ids identical between the two files?
def are_the_ids_identical(f1, f2):
    with open(f1, "r") as fin1:
        with open(f2, "r") as fin2:
            fin1.readline()
            fin2.readline()
            for l1, l2 in zip(fin1, fin2):
                id1 = int(l1.split(",")[0])
                id2 = int(l2.split(",")[0])
                if id1 != id2:
                    return False
    return True
print(are_the_ids_identical("../input/train_numeric.csv", "../input/train_categorical.csv"))
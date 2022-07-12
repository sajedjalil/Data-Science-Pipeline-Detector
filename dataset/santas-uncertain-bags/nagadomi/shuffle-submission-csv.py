# Shuffle gift IDs while preserving gift types.

# update1: Fix an error on python3
# update2: Add seed option

import numpy as np
import sys
import warnings
import argparse

def load_submission(filename):
    with open(filename) as f:
        lines = f.read().split("\n")[1:]
        if lines[-1] == "":
            lines = lines[:-1]
    bags = []
    ids = {}
    for line in lines:
        gifts = line.split(" ")
        bag = []
        for gift in gifts:
            name_id = gift.split("_")
            if name_id[0] not in ids:
                ids[name_id[0]] = [name_id[1]]
            else:
                ids[name_id[0]].append(name_id[1])
            bag.append(name_id[0])
        bags.append(bag)
    return bags, ids

def save_submission(filename, bags, ids):
    id_ind = {}
    id_seq = {}
    for t, i in ids.items():
        id_ind[t] = np.random.permutation(len(i))
        id_seq[t] = 0

    with open(filename, "w") as f:
        f.write("Gifts\n")
        for bag in bags:
            line = []
            for gift in bag:
                line.append(gift + "_" + ids[gift][id_ind[gift][id_seq[gift]]])
                id_seq[gift] += 1
            f.write(" ".join(line) + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-s", "--seed", action="store", nargs="?", type=int)
opt = parser.parse_args()
assert(opt.input != opt.output)
if opt.seed is None:
    opt.seed = np.random.randint(0, 0xffffffff)
    warnings.warn("set seed = {}, you can reproduce shuffling with `--seed {}` option".format(opt.seed, opt.seed))
np.random.seed(opt.seed)

bags, ids = load_submission(opt.input)
save_submission(opt.output, bags, ids)

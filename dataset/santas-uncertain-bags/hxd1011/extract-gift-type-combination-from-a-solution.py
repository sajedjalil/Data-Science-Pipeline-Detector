import collections
import argparse

def parseToyBagConfig(inputFile):
    f=open(inputFile)
    f.readline()
    total_config=[]
    for line in f:
        gift=line.split()
        toy_config=[]
        for i in gift:
            toy=i.split("_")[0]
            toy_config.append(toy)
        cnt=collections.Counter(toy_config)
        total_config.append(str(cnt)[9:-2])

    f.close()

    total_cnt=collections.Counter(total_config).most_common()
    for v in total_cnt:
        print(v)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
opt = parser.parse_args()

parseToyBagConfig(opt.input)


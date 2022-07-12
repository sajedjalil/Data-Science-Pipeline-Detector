import csv

sub_files = ['../input/mobilenet-lb-0-900-forked-from-beluga/gs_mn_submission_8518.csv',
             '../input/mobilenet-lb-0-895-please-upvote/gs_mn_submission_8465.csv',
             '../input/greyscale-mobilenet-lb-0-892/gs_mn_submission_8513.csv']

sub_weight = [2, 1.8, 1.6] ## Weights of the individual subs ##

Hlabel = 'key_id' 
Htarget = 'word'
npt = 3 # number of places in target

place_weights = {}
for i in range(npt):
    place_weights[i] = 5-i

lg = len(sub_files)
sub = [None]*lg
for i, file in enumerate( sub_files ):
    ## input files ##
    print("Reading {}: w={} - {}". format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file,"r"))
    sub[i] = sorted(reader, key=lambda d: float(d[Hlabel]))

## output file ##
out = open("sub_ens.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel,Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt,0) + (place_weights[ind]*sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()

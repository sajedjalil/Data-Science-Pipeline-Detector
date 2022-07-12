import csv
import time
def hk(x,y):
    return (x,y)
def update(hs,hsc,key,val):
    if(key in hs):
        hs[key]  += val
        hsc[key] += 1
    else:
        hs[key] = val
        hsc[key] = 1
def update1(hs,hsc):
    for i in hs.keys():
        x = hsc[i]
        y = hs[i]
        z = (y+x-1)/x
        hs[i] = int(z)
def find(pid,h1,h2):
    x = pid in h1.keys()
    y = pid[0] in h2.keys()
    if(x == True):
        return h1[pid]
    if(y == True):
        return h2[pid[0]]
    else:
        return 7
start = time.time()
res_file = open("sol_br_two.csv",'w')
fieldnames = ['id','Demanda_uni_equil']
writer = csv.DictWriter(res_file, fieldnames=fieldnames)
writer.writeheader()
train_file = "../input/train.csv"
test_file = "../input/test.csv"
train = open(train_file,)
train_reader = csv.reader(train)
spid = set()
spr = set()
test = open(test_file)
test_reader = csv.reader(test)
check = 0
for row in test_reader:
    if(check==0):
        check = 1
        continue
    cid = int(row[-2])
    pid = int(row[-1])
    pc = hk(pid,cid)
    spid |= {pc}
    spr |= {pid}
test.close()
dict_pid = {}
dict_pid_cnt = {}
pr = {}
prc = {}
check = 0
for row in train_reader:
    if(check==0):
        check = 1
        continue
    cid = int(row[4])
    pid = int(row[5])
    pc = hk(pid,cid)
    tar = int(row[-1])
    if(pc in spid):
        update(dict_pid,dict_pid_cnt,pc,tar)
    if(pid in spr):
        update(pr,prc,pid,tar)
update1(dict_pid,dict_pid_cnt)
update1(pr,prc)
test = open(test_file)
test_reader = csv.reader(test)
check = 0
x = 0
check = 0
t1 = time.time()
for row in test_reader:
    if(check==0):
        check = 1
        continue
    cid = int(row[-2])
    pid = int(row[-1])
    pc = hk(pid,cid)
    xy = find(pc,dict_pid,pr)
    r = {'id':str(x),'Demanda_uni_equil':str(xy)}
    writer.writerow(r)
    check=1
    x+=1
stop = time.time()
train.close()
test.close()

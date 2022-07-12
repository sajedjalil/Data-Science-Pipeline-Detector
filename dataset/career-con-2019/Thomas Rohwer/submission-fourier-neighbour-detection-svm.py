import numpy as np
import pandas as pd
import math as math

import os
if os.fstat(0) != os.fstat(1):
    pd.options.display.max_rows= None
    pd.options.display.max_columns= None
    pd.options.display.width= None


def deterministic():
    # determinism
    np.random.seed(42)
    import random as rn
    rn.seed(12345)

deterministic()

train_x = pd.read_csv('../input/X_train.csv')
train_y = pd.read_csv('../input/y_train.csv')
test_x = pd.read_csv('../input/X_test.csv')

fast= False
# fast= True

#num_test=500
num_test=0

def prepare_neigh(t):
    import pyquaternion as pq

    q= [ [pq.Quaternion(x2) for x2 in zip(*x) ] for x in zip(t['ox'].values, t['oy'].values, t['oz'].values, t['ow'].values) ]
    
    def metric(p0,p1,p2,p3): return min( pq.Quaternion.distance(p1,p2),
        max(pq.Quaternion.distance(2*p1-p0,p2), pq.Quaternion.distance(2*p2-p3,p1)) )

    def dist(a0,a,b,b0):
        return max(-(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w), metric(a0,a,b,b0))

    # compute probability dist of metric for adjacent
    d_values= []
    for e in q:
        for i in range(8):
            d= metric(e[i*16],e[i*16+1],e[i*16+2],e[i*16+3])
            d_values.append(d)

    thresh=np.quantile(d_values, 0.999)

    mesh=100

    def pos(x):
        return math.floor((x+1)/2*mesh)
    def pos2(x):
        return np.unique([ max(math.floor((x+1)/2*mesh-0.5),0), min(math.floor((x+1)/2*mesh+0.5),mesh-1) ])
    def ibucket(q):
       return (pos(q.x),pos(q.y),pos(q.z),pos(q.w))
    def gbuckets(q):
       for i in pos2(q.x):
           for j in pos2(q.y):
               for k in pos2(q.z):
                   for l in pos2(q.w):
                       yield i,j,k,l
    
    ind=[{},{}]
    for i in range(len(q)):
        for j in [0,1]:
            h= ind[j]
            k= ibucket(q[i][j*127])
            if k in h:
                h[k].append(i)
            else:
                h[k]=[i]

    neigh_fwd=[ [] for x in q ]
    neigh_rev=[ [] for x in q ]
    # in first round, store only distance; afterwards compute probabilities
    for i in range(len(q)):
        for k in gbuckets(q[i][127]):
            if k in ind[0]:
                for j in ind[0][k]:
                    if j==i: continue
                    d= dist(q[i][126],q[i][127],q[j][0],q[j][1])
                    if d>=thresh: continue
                    neigh_fwd[i].append((j, d))
                    neigh_rev[j].append((i, d))


    bandwidth= 0.00005
    
    from sklearn.neighbors.kde import KernelDensity
    # connected case
    kd= KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kd= kd.fit([[x] for x in d_values])
    delta0=-math.log(len(q)) # log probability of being connected
    p_connected= lambda x:kd.score_samples([[x]])[0]+delta0
    
    # all (conn+ not conn)
    total= len(q)*len(q)
    d_values2= []
    for i in range(len(q)):
        for n,d in neigh_fwd[i]: d_values2.append(d)

    delta= math.log(len(d_values2)/total)
    kd2= KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kd2= kd2.fit([[x] for x in d_values2])
    p_all= lambda x:kd2.score_samples([[x]])[0]+delta
    
    prob= lambda x:math.exp(p_connected(x)-p_all(x))

    print("thresh:",thresh)

    for neigh in [neigh_fwd, neigh_rev]:
        for i in range(len(q)):
            na=[ (n,prob(d)) for n,d in neigh[i] ]
            na.sort(key=(lambda x: x[1]), reverse=True)
            neigh[i]= na

    # simple algo first: follow most likely path until x vertices connected to initial
    neigh=[neigh_fwd,neigh_rev]
    neigh_discrete= [ [] for x in q]

    p_cutoff= 0.5
    len_cutoff= 30
    for i in range(len(q)):
        visited={i}
        visited_o=[i]
        front=[i,i]
        front_p=[1.0,1.0]
        while (len(visited))<len_cutoff:
            # find biggest weigh in forward, backward neighbours
            pmax=-math.inf
            nmax= None
            for d in [0,1]:
                for n in neigh[d][front[d]]:
                    if n[0] in visited: continue
                    p=n[1]*front_p[d]
                    if p<=pmax: continue
                    pmax= p
                    nmax= n[0]
                    dmax= d
            if nmax is None: break
            l= len(visited)
            if pmax<p_cutoff: break
            visited.add(nmax)
            visited_o.append(nmax)
            front[dmax]= nmax
            front_p[dmax]= pmax
        neigh_discrete[i]= visited_o

    return neigh_discrete

def add_neigh(t):
    neigh= prepare_neigh(t)
    t['neigh']= neigh
    t['stable_id']=[i for i in range(len(neigh))]

def prepare_base(t,ty=None):
    def f(d):
        d=d.sort_values(by=['measurement_number'])
        return pd.DataFrame({
         'len':[ len(d['measurement_number']) ],
         'lx':[ d['linear_acceleration_X'].values ],
         'ly':[ d['linear_acceleration_Y'].values ],
         'lz':[ d['linear_acceleration_Z'].values ],
         'ax':[ d['angular_velocity_X'].values ],
         'ay':[ d['angular_velocity_Y'].values ],
         'az':[ d['angular_velocity_Z'].values ],
         'ox':[ d['orientation_X'].values ],
         'oy':[ d['orientation_Y'].values ],
         'oz':[ d['orientation_Z'].values ],
         'ow':[ d['orientation_W'].values ],
        })

    # implicit copy; otherwise do so, because t gets modified later
    t= t.groupby('series_id').apply(f)
    t.reset_index()

    # merge
    l= range(len(t))
    if ty is None:
        # generate pseude
        ty= pd.DataFrame({'surface':[-1 for i in l],'group_id':['?' for i in l],'series_id':[i for i in l],'source':['te' for i in l]})
    else:
        ty=ty.copy()
        ty['source']= ['tr' for i in l]

    t=pd.merge(t,ty[['series_id','group_id','surface']],on='series_id')
    t=t.rename(columns={"surface": "y"})

    if fast:
        t= t[:100]

    return t

def prepare_data_svm(t, smoothing= True):
    def mfft(x):
        return [ np.fft.fft(v)[1:65]/math.sqrt(128.0) for v in x ]

    features= ['ax','ay','az','lx','ly','lz']
    
    tlen= len(t)

    fft= {f:mfft(t[f]) for f in features}

    # nfft is still complex
    nfft= {}
    
    for f in features:
        v=fft[f]

        lf= range(len(v[0]))
        norm= [ sum([np.absolute(v[i][j]) for i in range(tlen) ])/tlen for j in lf ]
        nfft[f]= [ [ l[j]/norm[j] for j in lf] for l in v ]

    def trans(x):
        return np.absolute(x)

    neigh= t['neigh'].values

    r=[ 0 for i in range(tlen) ]

    for i in range(tlen):
        rline=[[], [], []]
        for f in features:
            if smoothing:
                nd= [ nfft[f][k] for k in neigh[i] ]
                line= [ sum([ trans(v[j])/len(neigh[i]) for v in nd]) for j in range(len(nd[0])) ]
            else:
                line= [ trans(v) for v in nfft[f][i] ]

            # normalize
            avg= sum(line)/len(line)
            scale=1/(2*avg)
            
            rline[0].append([avg])
            rline[1].append([ scale*(line[2*i]+line[2*i+1]) for i in range(len(line)//2) ])

        r[i]= rline

    return r

def split_shuffle_groups(num_test, t):
    t= t.copy()

    # select randomly some groups (should be weighted by # of samples)

    aggcol='y' # arbitrary; just to get size
    gstat= t.groupby('group_id')[aggcol].agg(np.size)
    gstat= gstat.reset_index()

    import random

    groups= [t for t in zip(gstat['group_id'],gstat[aggcol]) ]
    random.shuffle(groups)
    
    test_groups= set()
    c=0
    for gid,len in groups:
        if gid=='?': continue
        if c>=num_test: break
        c+=len
        test_groups.add(gid)
    print("test groups:", test_groups)

    ctest = [ i for i,gid in enumerate(t['group_id']) if (gid in test_groups) ]
    ctrain = [ i for i,gid in enumerate(t['group_id']) if not (gid in test_groups) ]

    random.shuffle(ctrain)
    random.shuffle(ctest)

    return t.iloc[ctrain], t.iloc[ctest]

def split_shuffle(num_test, t):
    t= t.copy()
    ind= [ i for i,r in t.iterrows() if r['group_id']!='?' ]
    import random
    random.shuffle(ind)
    return t.iloc[ind[num_test:]], t.iloc[ind[:num_test]]



def folds_method(num, method):
    for i in range(num):
        yield (i,) + method()

def folds_groups(t, groups):
    import random

    for i,gid0 in enumerate(groups):
        ctest = [ i for i,gid in enumerate(t['group_id']) if gid==gid0 ]
        ctrain = [ i for i,gid in enumerate(t['group_id']) if (not gid==gid0) and (gid!='?') ]
        random.shuffle(ctrain)
        random.shuffle(ctest)
        yield i,t.iloc[ctrain],t.iloc[ctest]


def folds_allgroups(t):
    groups= set(t['group_id'].values).difference({'?'})
    return folds_groups(t, groups)


def folds_all(t):
    yield (0,) + split_shuffle(0, t)


def confusion_matrix(t, y_pred):
    y_real= t['y'].values

    confmat= pd.DataFrame.from_records([
        (l1,l2,0) for l1 in labels for l2 in labels ], columns=('pred','real','count'))
    confmat= confmat.set_index(['pred','real'])

    for i,v in enumerate(y_pred):
        confmat.at[(labels[y_pred[i]],labels[y_real[i]]),'count']= confmat.at[(labels[y_pred[i]],labels[y_real[i]]),'count']+1
    confmat= confmat.unstack()
    return confmat

def confusion_matrix_group(t, y_pred):
    # get groups from t
    group_at= t['group_id'].values
    groups= set(group_at)
    confmat= pd.DataFrame.from_records([
        (l,g,0) for l in labels for g in groups], columns=('pred','group','count'))
    confmat= confmat.set_index(['pred','group'])

    y_real= t['y'].values

    group_size= {g:0 for g in groups}
    group_type= {}
    group_predok= {g:0 for g in groups}
    for i,pred in enumerate(y_pred):
        g= group_at[i]
        pred= labels[y_pred[i]]
        real= labels[y_real[i]]
        confmat.at[(pred,g),'count']+=1
        group_size[g]+= 1
        group_type[g]= real
        if pred==real: group_predok[g]+=1
    
    confmat['count']= [ 1.0*confmat.at[(l,g),'count']/group_size[g] for l,g in confmat.index]

    # move count in index
    confmat= confmat.stack()
    confmat= confmat.reset_index(-1, drop=True)
    confmat= confmat.unstack()
    
    # add additional rows for: type, 
    
    def union(d1,d2): return dict(i for d in (d1,d2) for i in d.items())
    
    headers= pd.DataFrame()
    headers= headers.append(union({'pred':'type'},group_type), ignore_index=True)
    headers= headers.append(union({'pred':'samples'},group_size), ignore_index=True )
    headers= headers.append(union({'pred':'error'},{ g:1.0-1.0*group_predok[g]/group_size[g] for g in groups}), ignore_index=True)
    headers= headers.set_index('pred',drop=True)

    confmat= pd.concat([headers,confmat])

    # confmat= confmat.sort_index('error', axis=1, ascending=False)
    groups_sorted= [g for g in groups]
    groups_sorted.sort(key=(lambda g: group_predok[g]/group_size[g]), reverse=True)
    confmat= confmat.reindex(columns=groups_sorted)

    return confmat


total_okcount= 0.0
total_count= 0
total_confusion= pd.DataFrame()

def eval_test(model, test):
    y_test= test['y'].values

    y_pred0= model.predict(test)
    
    current_count= test.shape[0]

    y_pred= y_pred0
    y_pred= np.argmax(y_pred, axis=1)
    current_okcount= sum([ (1.0 if y_pred[i]==int(y_test[i]) else 0.0) for i in range(current_count) ])
    print("accuracy:", current_okcount/current_count)

    global total_okcount, total_count, total_confusion

    total_okcount+= current_okcount
    total_count+= current_count

    current_confusion= confusion_matrix_group(test, y_pred)
    for c in current_confusion.columns:
        if not c in total_confusion:
            total_confusion[c]= current_confusion[c]
            continue
        # merge columns according to sample #
        l1= total_confusion.at['samples',c]
        l2= current_confusion.at['samples',c]
        total_confusion.at['len',c]= l1+l2
        for l in labels:
            total_confusion.at[l,c]= (total_confusion.at[l,c]*l1 + current_confusion.at[l,c]*l2)/(l1+l2)

    print("confusion matrix test")
    print(current_confusion)
    print("current average score:", total_okcount/total_count)


def target_frequencies():
    tf= {
        'carpet':0.06,
        'concrete':0.16,
        'fine_concrete':0.09,
        'hard_tiles':0.06,
        'hard_tiles_large_space':0.10,
        'soft_pvc':0.17,
        'soft_tiles':0.23,
        'tiled':0.03,
        'wood':0.06,
    }
    s= sum([tf[k] for k in tf])

    return [ tf[l]/s for i,l in enumerate(labels)]

    # in train

def create_svm():

 def cn(a):
     # flatten arrays systematically
     try:
        for b in iter(a): yield from cn(b)
     except TypeError:
        yield a

 def prepare_tensors(t):
     x= [ [x for x in cn(v)] for v in t['svm'].values ]
     y= t['y'].values if 'y' in t else None
     return x,y

 def class_weight(y_train):
     # count samples
     count=[0 for l in labels]
     for y in y_train:
         count[y]+=1

     # error in fit, when one class is missing
     if min(count)==0: return None

     if num_test>0: return { i:1.0 for i,l in enumerate(labels) }

     tf= target_frequencies()
 
     return { i:tf[i]*len(y_train)/count[i] if count[i]>0 else 0.0 for i,l in enumerate(labels) }

 class svm_base:
    def prepare_data(self, t):
        t['svm']= prepare_data_svm(t)

    def fit(self, train, test):
        x_train, y_train= prepare_tensors(train)
        self.dim= max(y_train)+1

        from sklearn import svm
        self.model= svm.SVC(cache_size=2000, probability=False, decision_function_shape='ovo', random_state=2109,
            class_weight=class_weight(y_train),
            C=1.2, gamma='scale')

        self.model.fit(x_train, y_train)

    def predict(self, test):
        x_test, y_test= prepare_tensors(test)
        pred= self.model.predict(x_test)
        return [ np.array([1.0 if p==i else 0.0 for i in range(self.dim)]) for p in pred ]

 return svm_base()

if fast: num_test=min(num_test,10)

labels=None
labels_dict=None
def prepare_train():
    t=prepare_base(train_x, train_y)
    global labels, labels_dict
    labels=t['y'].unique()
    labels_dict={val:i for i, val in enumerate(labels)}
    t['y']=t['y'].apply(lambda x:labels_dict[x])

    add_neigh(t)
    
    return t

def prepare_test(both=False):
    t=prepare_base(test_x)
    add_neigh(t)
    return t
    

t= prepare_train()

# folds= folds_method(40, lambda: split_shuffle_groups(num_test, t))
folds= folds_allgroups(t)
# folds= folds_groups(t,{5})
# folds= folds_groups(t,range(6))

# problematic_groups=[13,16,27,33,35,51,60,25,72,40,10]
# folds= folds_groups(t,problematic_groups)


if num_test==0: folds= folds_all(t)


model= create_svm()

model.prepare_data(t)
for fold, train, test in folds:
    print("start fold", fold)
    
    model.fit(train, test)

    print("confusion matrix train")
    y_pred= model.predict(train)
    y_pred= np.argmax(y_pred, axis=1)
    print(confusion_matrix(train, y_pred))

    if num_test>0: eval_test(model, test)

if num_test>0:
    print("total confusion")

    groups_sorted= [g for g in total_confusion.columns]
    groups_sorted.sort(key=(lambda g: total_confusion.at['error', g]), reverse=True)
    total_confusion= total_confusion.reindex(columns=groups_sorted)

    total_confusion= total_confusion.transpose()
    total_confusion.index.rename('group', inplace=True)

    print(total_confusion)

    print("total test score:",total_okcount/total_count," total instances:",total_count)
    exit(0)

t=prepare_test()

model.prepare_data(t)

yraw= model.predict(t)
y= np.argmax(yraw, axis=1)

"""
# print detailed statistics
def union(d1,d2): return dict(i for d in (d1,d2) for i in d.items())
print(pd.DataFrame(union(
    {
    'series_id':t['series_id'],
    'surface': [ labels[y[i]] for i in range(0,len(y)) ]
    },
    {l:[ p[li] for p in yraw] for li,l in enumerate(labels) }
)))
"""

submission=pd.DataFrame({
    'series_id': t['series_id'],
    'surface': [ labels[y[i]] for i in range(0,len(y)) ],
})

submission.to_csv('submission.csv',index=False)

# submission stats
stat= submission.copy()
stat['count']=[ 0 for i,r in stat.iterrows() ]

stat= stat.groupby('surface').count()

stat= stat[['count']]
stat['count']= [ 1.0*v/len(submission) for v in stat['count'].values ]
print(stat)

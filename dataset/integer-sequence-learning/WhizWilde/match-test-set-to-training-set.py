import numpy as np
import pandas as pd
import math
from functools import reduce
#from numba import jit, int32, int64
"""testing on my personal computer was disappointing,
while I intended this to be quite faster by precompiling some functions related to numpy operations..."""
import time

"""Script based on Nina Chen's (https://www.kaggle.com/ncchen)
Integer Sequence Learning Competition
'Match test set to training set'
(https://www.kaggle.com/ncchen/integer-sequence-learning/match-test-set-to-training-set/comments)

Fork by WhizWilde:
I did not really created change in the functions' logic apart specific points,
I merely tried to optimize a few things using numpy. But it doesn't seem to be a success.
I don't know if it comes from using numpy.ndarray with python longs or if there's something else I am missing.
I will try to test later (mostly at the end of competition, still have to make real submissions;)
"""

t0=time.time()

trainfile='../input/train.csv'
testfile='../input/test.csv'

train_df= pd.read_csv(trainfile, index_col="Id")
test_df = pd.read_csv(testfile, index_col="Id")


def preprocess(dataframe)    :
    df_seqs= dataframe['Sequence'].to_dict()    
    new_seqs={}
    for key in df_seqs:
        seq=df_seqs[key]
        seq=np.array ([int(x) for x in seq.split(',')],dtype=object)
        new_seqs[key]=seq
    return dataframe, new_seqs    

train_final, train_seqs=preprocess(train_df)    
test_final, test_seqs=preprocess(test_df)   


MIN_LENGTH = 10  #Ignore sequences with length<10

print ("Time Elapsed since the beginning:  %.2f seconds" %(time.time()-t0))

#@jit
def findGCD(seq):
    """ Compute the greatest common divisor of a list of numbers. """
    gcd = seq[0]
    
    gcd=reduce (math.gcd, seq)
    #this was faster individually, but seems actually slow?    
    return gcd

#@jit
def findSignature2(seq, n = MIN_LENGTH):
    """NC: Compute the signature of the sequence using the first n elements
        if the length of sequence is less than n, return the empty tuple. """
#    print("finding difference")
    
    if len(seq)<n:
        return tuple([])
    seq_up=np.array (seq[1:n], dtype=object)#these will be float
    seq_dn=np.array (seq[0:n-1], dtype=object)
    difference = np.array(seq_up-seq_dn, dtype=object)
   
    if np.any(difference):
        sign = np.sign(difference[difference.nonzero()[0][0]])
        gcd = findGCD(difference)
        signature=tuple(sign*(difference/gcd))
    else:
        signature = tuple(np.sign(difference))
    return signature

#@jit
def findLine(x,y, n, requireInteger=True, useNumpy=True):  
    """NC: Find [m,b] so that y=mx+b holds for the first n points: (x1,y1), (x2,y2),...(xn,yn)

    Args:
        x,y: list[int]
        n: int, number of points fitted
        requireInteger: boolean, whether m,b must be integers
        
    Returns:
        [m,b]: int m, int b
    
    Remark:
        This should be faster than numpy.polyfit(x,y,1) 
    """
     
    #  Find m,b use the first two points (x0,y0),(x1,y1) 
    #  Formula: m = (y1-y0)/(x1-x0).
    #  If the denominator becomes zero, use the next points.   
    x0 = x[0]
    i = 1
    while(i<n-1 and x[i]==x[0]):
        i+=1
    x1=x[i]
    if x1==x0:
        return None
    else:
        y0,y1 = y[0],y[i]
    m =1.0*(y1-y0)/(x1-x0)
    #WW: with type casting, this should be automatically converted to float if needed?
    b = y[0]-m*x[0]
    # Check if m,b are integers
    m_int = int(round(m))
    b_int = int(round(b))
    
    if abs(m-m_int)>10**(-2) or abs(b-b_int)>10**(-2):
        return None
    else:
            m, b = m_int, b_int
    # Check if the next points satisfty y=mx+b
    #y_predict = m*np.array(x)+b
    #this is broken in the original script.Will cause value error because of impossibility of broadcasting

    y_predict = np.add(np.multiply(np.array(x[0:n]),m),b)
#    y_predict = m*np.array(x[0:n])+b

    substract = np.abs(np.array(y[0:n])-y_predict)
#    changed name to avoid confusion with "difference" used for making the signature
    error = np.max(substract)   
    if error<10**(-2):
        return [m,b]

    
# Compute signatures using the first 10 elements.
minlength = MIN_LENGTH
t3=time.time()
train_df['Signature'] = [findSignature2(train_seqs[id][:minlength], minlength) for id in train_df.index]
#train_df['Signature'] = [findSignature(train_seqs[i][:minlength], minlength) for i in train_df.index]
t4=time.time()
print ("Time Elapsed between t4 and t3:  %.2f seconds" %(t4-t3))
t5=time.time()
test_df['Signature'] = [findSignature2(test_seqs[id][:minlength], minlength) for id in test_df.index]
t6=time.time()
print ("Time Elapsed between t6 and t5:  %.2f seconds" %(t6-t5))
t7=time.time()
# Group data frames by signatures
train_gb = train_df.groupby(['Signature'], sort=True)
test_gb = test_df.groupby(['Signature'], sort=True)
t8=time.time()
print ("Time Elapsed between t8 and t7:  %.2f seconds" %(t8-t7))

# Find signatures that appear in both train/test sets
t8=time.time()
commonSignatures = list(set(test_gb.groups.keys()).intersection(train_gb.groups.keys()))
t9=time.time()
print ("Time Elapsed between t9 and t8:  %.2f seconds" %(t9-t8))
t10=time.time()
commonSignatures = list(set(test_gb.groups.keys()).intersection(train_gb.groups.keys()))
#should be tried?
#commonSignatures = list(set(test_gb.groups.keys()).intersection(set(train_gb.groups.keys())))
commonSignatures.remove(tuple([]))
t11=time.time()
print ("Time Elapsed  between t11 and t10:  %.2f seconds" %(t11-t10))

#Find match (train, test) pairs

# For every (test, train) pair of sequences with the same signature,
# Let (x,y)= (test, train) or (x,y)=(train, test),
# verify whether y=mx+b.
# Requirement: Train sequence must be longer than test sequence to make prediction
result={}
t12=time.time()
for signature in commonSignatures:
    for test_id in test_gb.groups[signature]:
        test_seq = test_seqs[test_id]
        n = len(test_seq)
        train_candidates = train_gb.groups.get(signature)

        for train_id in train_candidates:
            train_seq=train_seqs[train_id]
            if len(train_seq)<=n: # too short to  make prediction
                continue
             
            # Check if train = m*test + b
            line = findLine(train_seq,test_seq, n)  
            if line:
                [m,b] = line
                predict = str(m*train_seq[n]+b)
                result[test_id] = (train_id, [m,b], '(train,test)', predict)
                break
            
            # Check if test = m*train + b
            line = findLine(test_seq,train_seq, n)
            if line:
                [m,b] = line
                if m!=0:
                    predict = str((train_seq[n]-b)/m)
                    result[test_id] = (train_id, [m,b], '(test,train)', predict)
                    break
t13=time.time()
print ("Time Elapsed between t13 and t12: %.0f seconds" %(t13-t12))
print ("Time Elapsed between t13 and t0: %.0f seconds" %(t13-t0))
#Save the result to a data frame: match_df
match_df = pd.DataFrame.from_dict(result, orient='index', dtype=None)
match_df.columns=['TrainID', '[m,b]','(x,y)', 'Prediction']
match_df.index.name="TestID"
match_df=match_df.sort_index()

match_df.to_csv("matchPairs.csv")
print ("Sample output, rows 25-30: ")
match_df[25:30]
print ("Total Time Elapsed: %.0f seconds" %(time.time()-t0))

"""Note that conclusions are similar, at least for the example presented in original script.
I will make more comparisons later. Feel free to comment on any aspects you feel worth of it"""



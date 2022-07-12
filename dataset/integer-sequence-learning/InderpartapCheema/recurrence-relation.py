import pandas as pd
import numpy as np

testfile='../input/test.csv'
data = open(testfile).readlines()

sequences={}   #(key, value) = (id , sequence)
for i in range(1,len(data)): 
    line=data[i]
    line =line.replace('"','')
    line = line[:-1].split(',')
    id = int(line[0])
    sequence=[int(x) for x in line[1:]];
    sequences[id]=sequence

def checkRecurrence(seq, order= 2, minlength = 7):
    """
    :type seq: List[int]
    :type order: int
    :type minlength: int 
    :rtype: List[int]
    
    Check whether the input sequence is a recurrence sequence with given order.
    If it is, return the coefficients for the recurrenec relation.
    If not, return None.
    """     
    if len(seq)< max((2*order+1), minlength):
        return None
    
    ################ Set up the system of equations 
    A,b = [], []
    for i in range(order):
        A.append(seq[i:i+order])
        b.append(seq[i+order])
    A,b =np.array(A), np.array(b)
    try: 
        if np.linalg.det(A)==0:
            return None
    except TypeError:
        return None
   
    #############  Solve for the coefficients (c0, c1, c2, ...)
    coeffs = np.linalg.inv(A).dot(b)  
    
    ############  Check if the next terms satisfy recurrence relation
    for i in range(2*order, len(seq)):
        predict = np.sum(coeffs*np.array(seq[i-order:i]))
        if abs(predict-seq[i])>10**(-2):
            return None
    
    return list(coeffs)


def predictNextTerm(seq, coeffs):
    """
    :type seq: List[int]
    :type coeffs: List[int]
    :rtype: int
    
    Given a sequence and coefficienes, compute the next term for the sequence.
    """
    
    order = len(coeffs)
    predict = np.sum(coeffs*np.array(seq[-order:]))
    return int(round(predict))
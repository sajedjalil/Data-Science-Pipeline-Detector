# # Summary:#
# 
#    This notebook uses elementary math methods to detect linear recurrence sequences.  Machine learning methods are *not* used here. 
#     
#    Among the 113,849 sequenes in the test set, we found more than 5000 of them satify recurrence relations of order 2,3 or 4. For those sequences, we computed the recurrence relations and predict the next terms. Moreover, we found recurrence relations that are not described in the OEIS. 
# 
# ## Linear Recurrence Relations ##
# 
# (Remark: This notebook only considers homogeneous linear recurrence relations with constant coefficients. Nonlinear or non-homogeneous relations are not investigated here.)
# 
# ### 2nd order recurrence relation ###   
# A second order recurrence relation is of the form:  $$ a_{n+2} = c_{0}a_{n}+ c_{1}a_{n+1}, $$
# where the coefficients $c_{0}$ and $c_{1}$ are constant.
# 
# For example, the Fibonacci sequence $a_{n+2}= a_{n}+a_{n+1}$ is a second order recurrence sequence with coefficients $(1,1)$.
# 
# ### 3rd order recurrence relation ###   
# A second order recurrence relation is of the form:  $$ a_{n+3} = c_{0}a_{n}+ c_{1}a_{n+1}+c_{2}a_{n+2}, $$
# where the coefficients $c_{0},c_{1},c_{2}$ are constant.
# 
# 
# ## Detect Recurrence Relations ##
# 
# Given a sequence $a_{n}$, let's say we want to verify whether it's given by a 3rd order recurrence relation. In other words, we check if it's possible to find constants $c_{0},c_{1},c_{2}$ so that $$a_{n+3} = c_{0}a_{n}+ c_{1}a_{n+1}+c_{2}a_{n+2}$$ is satified. To find possible $c_{0},c_{1},c_{2}$, since there are 3 unknowns, we need at least 3 equations. Let's set the equations using $a_{3},a_{4},a_{5}$ as follows:
# 
# $$ a_{3} = c_{0}a_{0}+ c_{1}a_{1}+c_{2}a_{2} $$
# $$ a_{4} = c_{0}a_{1}+ c_{1}a_{2}+c_{2}a_{3} $$
# $$ a_{5} = c_{0}a_{2}+ c_{1}a_{3}+c_{2}a_{4}. $$
# 
# Writting these equations in matrix form, we obtain
# 
# $$\begin{bmatrix}
# a_{0} & a_{1} & a_{2} \\ 
# a_{1} & a_{2} & a_{3} \\
# a_{2} & a_{3} & a_{4}
# \end{bmatrix}
# \begin{bmatrix}
# c_{0} \\ 
# c_{1}\\
# c_{2}
# \end{bmatrix}=
# \begin{bmatrix}
# a_{3} \\ 
# a_{4}\\
# a_{5}
# \end{bmatrix},
# $$ 
# then we solve for $(c_{0},c_{1},c_{2})$. Once the coefficients $(c_{0},c_{1},c_{2})$ are found, we check whether the next terms $a_{6},a_{7},\cdots$ satisfy the recurrence relation.
# 
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

# ## Example: ##
# * Given a sequence [1,5,11,21,39,73,139,269,527].
# * We verify if it's 3rd order recurrence sequence and find the coefficients (2,-5,4).
# * We then predict the next term using the last 3 terms and the relation $a_{n+3} = 2a_{n}-5a_{n+1}+4a_{n+2}$. 
seq = [1,5,11,21,39,73,139,269,527]
print (checkRecurrence(seq,3))
print (predictNextTerm(seq, [2,-5,4]))
# # Find 2nd order sequeneces in the test set #
order2Seq={}   #(key, value) = (sequence id, [prediction, coefficients])
for id in sequences:  
    seq = sequences[id]
    coeff = checkRecurrence(seq,2)
    if coeff!=None:
        predict = predictNextTerm(seq, coeff)
        order2Seq[id]=(predict,coeff)

print ("We found %d sequences\n" %len(order2Seq))

print  ("Some examples\n")
print ("ID,  Prediction,  Coefficients")
for key in sorted(order2Seq)[0:5]:
    value = order2Seq[key]
    print ("%s, %s, %s" %(key, value[0], [int(round(x)) for x in value[1]]))
# # Find 3rd order sequeneces in the test set #
order3Seq={}
for id in sequences:
    if id in order2Seq:
        continue
    seq = sequences[id]
    coeff = checkRecurrence(seq,3)
    if coeff!=None:
        predict = predictNextTerm(seq, coeff)
        order3Seq[id]=(predict,coeff)

print ("We found %d sequences\n" %len(order3Seq))

print  ("Some examples\n")
print ("ID,  Prediction,  Coefficients")
for key in sorted(order3Seq)[0:5]:
    value = order3Seq[key]
    print ("%s, %s, %s" %(key, value[0], [int(round(x)) for x in value[1]]))
# # Find 4th order sequeneces in the test set #
order4Seq={}
for id in sequences:  
    if id in order2Seq or id in order3Seq:
        continue
    seq = sequences[id]
    coeff = checkRecurrence(seq,4)
    if coeff!=None:
        predict = predictNextTerm(seq, coeff)
        order4Seq[id]=(predict,coeff)

print ("We found %d sequences \n" %len(order4Seq))
print  ("Some examples\n")
print ("ID,  Prediction,  Coefficients")
for key in sorted(order4Seq)[4:5]:
    value = order4Seq[key]
    print ("%s, %s, %s" %(key, value[0], [int(round(x)) for x in value[1]]))

print (sequences[239][0:17])
# ## Recurrence relations not included in OEIS ##
# In the previous cells,
#     * We find that Sequence 239 is a 4th order sequence and predict the next term as 5662052980.
#     * We check OEIS https://oeis.org/A000773, which confirms the prediction is correct.
#     * We observe that this recurrence relation is not described in OEIS. (There are more such sequences.)
print("Conclusion:")
print("Number of sequences in the test set:", len(sequences))
print("Number of 2nd order sequences:", len(order2Seq))
print("Number of 3rd order sequences:", len(order3Seq))
print("Number of 4th order sequences:", len(order4Seq))
import theano
import theano.tensor as T
import numpy as np

a = np.array([[1,2,3], [1,2,3], [2,3,1], [3,2,1]], dtype=np.float)
print(a.shape)
b = np.array([[1,2,3,4,5], [5,1,2,3,0], [2,3,5,1,4]], dtype=np.float)
print(b.shape)
s = T.dot(theano.shared(a), theano.shared(b))
f = theano.function([], s)
print(f())
r = T.argmax(s, axis=1)
f = theano.function([], r)
print(f())
m = theano.function([], T.nnet.softmax(T.dot(a, b)).shape)
print(m())
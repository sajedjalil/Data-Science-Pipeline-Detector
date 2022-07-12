import theano
#theano.config.device = 'gpu'
import numpy
from theano import tensor

x = tensor.matrix('x')
y = 2 * x
f = theano.function([x], y)

print(f(numpy.ones((100, 100))))

print(theano.config)
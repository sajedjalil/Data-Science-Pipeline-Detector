import numpy as np
import timeit
import math

# vectorized error calc
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

#looping error calc
def rmsle_loop(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

# create random values to demonstrate speed difference
y1 = np.random.rand(1000000)
y2 = np.random.rand(1000000)

t0 = timeit.default_timer()
err = rmsle_loop(y1,y2)
elapsed = timeit.default_timer()-t0
print('Using loops:')
print('RMSLE: {:.3f}\nTime: {:.3f} seconds'.format(err, elapsed))

t0 = timeit.default_timer()
err = rmsle(y1,y2)
elapsed = timeit.default_timer()-t0
print('\nUsing vectors:')
print('RMSLE: {:.3f}\nTime: {:.3f} seconds'.format(err, elapsed))
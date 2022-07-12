
from matplotlib import cm
from matplotlib.pyplot import subplots, show
from numpy import linspace, bincount, hstack, cumsum
from numpy.random import rand, random_integers


def confusion_probs(predictions, labels, ax = None, vmargin = 0.1, equalize = False, cmap = cm.jet):
	"""
		Make a confusion-matrix-like scatter plot
	"""
	N, D = predictions.shape
	bins = bincount(labels)
	assert labels.min() >= 0 and labels.max() < D, 'labels should be between 0 and {0:d}'.format(D - 1)
	assert predictions.min() >= 0 and predictions.max() <= 1, 'predictions should be normalized (to [0, 1])'
	""" Set up graphical stuff. """
	fig = None
	if ax is None:
		fig, ax = subplots(figsize = (6, 6))
		fig.tight_layout()
	if equalize:
		xticks = [x + .5 for x in range(D)]
	else:
		xticks = (float(D) / N) * (cumsum(bins) - bins / 2)
	ax.set_xticks(xticks)
	ax.set_xticklabels(range(D))
	ax.set_xlim([0, D])
	ax.set_yticks([(x + .5) * (1 + vmargin) for x in range(D)])
	ax.set_yticklabels(range(D))
	ax.set_ylim([0, D * (1 + vmargin) - vmargin])
	""" Sort predictions by label. """
	indices = labels.argsort()
	X = predictions[indices, :]
	y = labels[indices]
	""" Add more vertical offset to higher labels. """
	X += linspace(0, D - 1, D) * (1 + vmargin)
	""" Make a new horizontal axis to give all samples equal width. """
	if equalize:
		z = hstack([k + linspace(0, 1, t) for k, t in enumerate(bins)])
	else:
		z = linspace(0, D, N)
	""" Plot the results with class-coding. """
	for q in range(D):
		colors = cmap(y / float(D))
		ax.scatter(z, X[:, q], c = colors, edgecolors = 'none')
	return (fig, ax) if fig else ax


if __name__ == '__main__':
	"""
		Simple demonstration.
	"""
	predictions = rand(10000, 6)
	labels = hstack((random_integers(0, 5, size = (5000,)), random_integers(2, 3, size = (2500,)), [5] * 2500))
	predictions[range(10000), labels] *= 100
	predictions /= predictions.sum(1, keepdims = True)
	fig, ax = confusion_probs(predictions, labels, equalize = True)
	fig.savefig('equalized.png')
	fig, ax = confusion_probs(predictions, labels, equalize = False)
	fig.savefig('unbalanced.png')



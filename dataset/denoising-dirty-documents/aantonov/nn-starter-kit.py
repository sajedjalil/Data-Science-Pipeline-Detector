"""
A simple feed-forward neural network that denoises one pixel at a time

author: Rangel Dokov
"""
import numpy as np
import theano
import theano.tensor as T
import cv2
import os
import itertools

theano.config.floatX = 'float32'

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def feature_matrix(img):
    """Converts a grayscale image to a feature matrix

    The output value has shape (<number of pixels>, <number of features>)
    """
    # select all the pixels in a square around the target pixel as features
    window = (5, 5)
    nbrs = [cv2.getRectSubPix(img, window, (y, x)).ravel()
            for x, y in itertools.product(range(img.shape[0]), range(img.shape[1]))]

    # add some more possibly relevant numbers as features
    median5 = cv2.medianBlur(img, 5).ravel()
    median25 = cv2.medianBlur(img, 25).ravel()
    grad = np.abs(cv2.Sobel(img, cv2.CV_16S, 1, 1, ksize=3).ravel())
    div = np.abs(cv2.Sobel(img, cv2.CV_16S, 2, 2, ksize=3).ravel())
    misc = np.vstack((median5, median25, grad, div)).transpose()

    # compose the full feature matrix
    features = np.hstack((np.asarray(nbrs), misc))
    return (features / 255.0).astype('float32')


def target_matrix(img):
    """Converts a grayscale image to a target matrix

    The output has shape (<number of pixels>, 1). It is basically a vector, but
    it is converted to a matrix to make theano work.
    """
    return (img / 255.0).astype('float32').ravel()[:, None]


def load_train_set(file_list):
    xs = []
    ys = []
    for fname in file_list:
        xs.append(feature_matrix(load_image(os.path.join('../input/train/', fname))))
        ys.append(target_matrix(load_image(os.path.join('../input/train_cleaned/', fname))))

    return np.vstack(xs), np.vstack(ys)


class Layer(object):
    """Representation of a network layer

    It contains a weight matrix `W` and a bias vector `b` and computes the
    function:

        activation(input * W + b)

    where `activation` is typically some sort of sigmoid function.
    """
    def __init__(self, rng, inp, n_in, n_out, activation=T.tanh):
        W_values = rng.uniform(low=-(1.0 / (n_in + n_out)),
                               high=(1.0 / (n_in + n_out)),
                               size=(n_in, n_out)).astype('float32')
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = np.zeros((n_out,), dtype=np.float32)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.input = inp
        self.output = activation(T.dot(self.input, self.W) + self.b)
        self.params = [self.W, self.b]


class Model(object):
    """A neural network with a single hidden layer """
    def __init__(self, rng, inp, n_in, n_hidden, n_out):
        self.layer1 = Layer(
            rng=rng,
            inp=inp,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.layer2 = Layer(
            rng=rng,
            inp=self.layer1.output,
            n_in=n_hidden,
            n_out=n_out,
            activation=lambda x: T.clip(x + 0.5, 0.0, 1.0)
        )
        # NOTE: using `clip` as the activation function is probably a bad idea,
        # because once it satures you are stuck with a zero gradient. On the
        # other hand it produced good results...

        self.input = inp
        self.output = self.layer2.output
        self.params = self.layer1.params + self.layer2.params

    def cost(self, y):
        return T.mean((self.output - y)**2)


def sgd_train(model, training_set, batch_size, learning_rate, n_epochs):
    """Naive gradient descent"""
    train_x, train_y = training_set

    n_train_batches = train_x.shape[0] // batch_size
    x = model.input
    y = T.matrix('y')
    y.tag.test_value = np.random.rand(batch_size, train_y.shape[1]).astype('float32')

    gparams = [T.grad(model.cost(y), param) for param in model.params]
    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(model.params, gparams)]

    train_model = theano.function(
        inputs=[x, y],
        outputs=[],
        updates=updates
    )
    # theano.printing.pydotprint(train_model, outfile="train_graph.png", var_with_name_simple=True)

    predict = theano.function(
        inputs=[x],
        outputs=[model.output]
    )
    # theano.printing.pydotprint(predict, outfile="predict_graph.png", var_with_name_simple=True)
    valid_x, valid_y = load_train_set(['3.png'])

    for epoch in range(n_epochs):
        for batch_id in range(n_train_batches):
            index = list(range(batch_id*batch_size, (batch_id + 1)*batch_size))
            train_model(train_x[index], train_y[index])

        y_pred, = predict(valid_x)
        print('Epoch {}, validation error: {}'.format(
            epoch,
            np.sqrt(np.mean((valid_y - y_pred)**2))
        ))


def main():
    rng = np.random.RandomState(12345)

    N_HIDDEN = 10
    BATCH_SIZE = 20
    N_EPOCHS = 100
    RATE = 0.1

    TRAIN_IMAGES = ['2.png']
    # NOTE: the data loading is really inefficient and trying to use all of the
    # training images is likely going to run out of RAM
    # TRAIN_IMAGES = os.listdir('../input/train')

    train_x, train_y = load_train_set(TRAIN_IMAGES)

    x = T.matrix('x')
    x.tag.test_value = np.random.rand(BATCH_SIZE, train_x.shape[1]).astype('float32')

    model = Model(
        rng=rng,
        inp=x,
        n_in=train_x.shape[1],
        n_hidden=N_HIDDEN,
        n_out=train_y.shape[1]
    )

    sgd_train(model, (train_x, train_y), batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, learning_rate=RATE)

    predict = theano.function(
        inputs=[x],
        outputs=[model.output]
    )

    # for fname in os.listdir('../input/test/'):
    for fname in ['1.png']:
        test_image = load_image(os.path.join('../input/test', fname))
        test_x = feature_matrix(test_image)

        y_pred, = predict(test_x)
        output = y_pred.reshape(test_image.shape)*255.0

        cv2.imwrite('original_' + fname, test_image)
        cv2.imwrite('cleaned_' + fname, output)


if __name__ == '__main__':
    main()

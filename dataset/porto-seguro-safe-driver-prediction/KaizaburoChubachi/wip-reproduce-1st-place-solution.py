import numpy as np
import pandas as pd

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import training
from chainer.training import extensions

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score

from scipy.special import erfinv


class MLP(chainer.ChainList):

    def __init__(self, topology, dropout=-1, dropout_input=-1):
        super(MLP, self).__init__()
        self.total_hidden_dim = sum(topology[1:-1])
        self.dropout = dropout
        self.dropout_input = dropout_input

        n_in = topology[0]
        for n_out in topology[1:]:
            self.add_link(L.Linear(n_in, n_out))
            n_in = n_out

    def get_hidden_concat(self, X):
        hidden_outputs = np.zeros((X.shape[0], self.total_hidden_dim), dtype=np.float32)
        child_links = list(self.children())

        next_start = 0
        for f in child_links[:-1]:
            X = F.relu(f(X))

            next_end = next_start + X.shape[1]
            hidden_outputs[:, next_start:next_end] = chainer.cuda.to_cpu(X.data)
            next_start = next_end

        return hidden_outputs

    def __call__(self, X):
        child_links = list(self.children())

        if self.dropout_input > 0:
            X = F.dropout(X, self.dropout_input)

        for f in child_links[:-1]:
            X = F.relu(f(X))
            if self.dropout > 0:
                X = F.dropout(X, self.dropout)

        y = child_links[-1](X)
        return y


class MSERegressor(chainer.Chain):

    def __init__(self, base_model):
        super(MSERegressor, self).__init__()
        with self.init_scope():
            self.base_model = base_model

    def __call__(self, X, y):
        y_hat = self.base_model(X)
        loss = F.mean_squared_error(y, y_hat)
        chainer.report({'loss': loss}, self)
        return loss


class SCEClassifier(chainer.Chain):

    def __init__(self, base_model):
        super(SCEClassifier, self).__init__()
        with self.init_scope():
            self.base_model = base_model

    def __call__(self, X, y):
        y_hat = self.base_model(X)
        loss = F.sigmoid_cross_entropy(y_hat[:, 0], y)
        chainer.report({'loss': loss}, self)
        return loss


class SwapNoiseIterator(chainer.iterators.SerialIterator):

    def __init__(self, X, noise_rate, batch_size, repeat=True, shuffle=True):
        self.X = X
        self.noise_rate = noise_rate
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle

        self.dataset = None
        self.reset_noise()
        self.reset()

    def reset_noise(self):
        X = self.X.copy()

        swap_idx = (np.random.uniform(0, 1, X.shape) < self.noise_rate)
        swap_nums = swap_idx.sum(axis=0)
        for i in range(X.shape[1]):
            X[swap_idx[:, i], i] = np.random.choice(self.X[:, i], swap_nums[i])

        self.dataset = chainer.datasets.TupleDataset(X, self.X)

    def __next__(self):
        # All lines are the same as the original SerialIterator
        # except the line `self.reset_noise()`

        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        if self._order is None:
            batch = self.dataset[i:i_end]
        else:
            batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    np.random.shuffle(self._order)

                self.reset_noise()

                if rest > 0:
                    if self._order is None:
                        batch.extend(self.dataset[:rest])
                    else:
                        batch.extend([self.dataset[index]
                                      for index in self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__

    def reset(self):
        super(SwapNoiseIterator, self).reset()


def fit(model, iterator, settings, gpu):
    if gpu >= 0:
        model.to_gpu(gpu)

    optimizer = chainer.optimizers.SGD(lr=settings['learning_rate'])
    optimizer.setup(model)
    if 'reg_l2' in settings:
        optimizer.add_hook(chainer.optimizer.WeightDecay(settings['reg_l2']))
    print(optimizer.lr)

    updater = training.StandardUpdater(iterator, optimizer, device=gpu)
    trainer = training.Trainer(updater, (settings['nb_epochs'], 'epoch'))

    trainer.extend(extensions.ExponentialShift('lr', settings['learning_rate_decay']),
                   trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    print(optimizer.lr)


def predict(model, iterator, gpu):
    iterator.reset()

    pred = None
    next_start = 0

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for batch in iterator:
            X_batch = chainer.dataset.concat_examples(batch, gpu)[0]
            y_hat = model(X_batch)

            if isinstance(y_hat, chainer.Variable):
                y_hat = y_hat.data

            y_hat = chainer.cuda.to_cpu(y_hat)

            if pred is None:
                pred = np.zeros((len(iterator.dataset), y_hat.shape[1]), dtype=np.float32)

            next_end = next_start + y_hat.shape[0]
            pred[next_start:next_end] = y_hat
            next_start = next_end

    return pred


if __name__ == '__main__':
    # general settings
    seed = 1024
    gpu = 0
    submission_name = 'submission.csv'

    np.random.seed(seed)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()

    # load data
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    X = pd.concat([train.drop(['id', 'target'], axis=1),
                   test.drop(['id'], axis=1)], axis=0).reset_index(drop=True)

    calc_columns = [c for c in X.columns if 'calc' in c]
    X = X.drop(calc_columns, axis=1)

    # one hot encoding
    cat_columns = [c for c in train.columns if 'cat' in c]
    for col in cat_columns:
        labels, uniques = pd.factorize(X[col])
        dummy = pd.get_dummies(labels).values
        for i in range(dummy.shape[1]):
            X[col + '_one_hot_' + str(i)] = dummy[:, i]

    trafo_columns = [c for c in X.columns if len(X[c].unique()) != 2]

    # Gauss Rank transformation
    for col in trafo_columns:
        values = sorted(set(X[col]))
        # Because erfinv(1) is inf, we shrink the range into (-0.9, 0.9)
        f = pd.Series(np.linspace(-0.9, 0.9, len(values)), index=values)
        f = np.sqrt(2) * erfinv(f)
        f -= f.mean()
        X[col] = X[col].map(f)

    X = X.values.astype(np.float32)

    # train denoising autoencoder
    settings = {
        'topology': [221, 1500, 1500, 1500, 221],
        'learning_rate': 3e-3,
        'batch_size': 128,
        'learning_rate_decay': 0.995,
        'swap_noise': 0.15,
        'nb_epochs': 1000
    }
    assert len(settings['topology']) % 2 == 1
    assert settings['topology'][0] == settings['topology'][-1]

    autoencoder = MLP(settings['topology'])
    model = MSERegressor(autoencoder)
    iterator = SwapNoiseIterator(X, settings['swap_noise'], settings['batch_size'])
    fit(model, iterator, settings, gpu)
    chainer.serializers.save_npz('autoencoder.npz', autoencoder)

    # If you want to load trained weights rather than train from scratch,
    # comment out above two lines, and use below
    # chainer.serializers.load_npz('autoencoder.npz', autoencoder)
    # if gpu >= 0:
    #     model.to_gpu(gpu)

    # extract hidden layer's outputs.
    iterator = chainer.iterators.SerialIterator(
        chainer.datasets.TupleDataset(X),
        settings['batch_size'], repeat=False, shuffle=False)
    transformed = predict(autoencoder.get_hidden_concat, iterator, gpu)

    # train nn classifier
    settings = {
        'topology': [4500, 1000, 1000, 1],
        'dropout': 0.5,
        'dropout_input': 0.1,
        'reg_l2': 0.05,
        'learning_rate': 1e-4,
        'batch_size': 128,
        'learning_rate_decay': 0.995,
        'nb_epochs': 150
    }

    X, y = transformed[:train.shape[0]], train['target'].values
    X_test = transformed[train.shape[0]:]

    pred_valid = np.zeros(X.shape[0])
    pred_test = np.zeros(X_test.shape[0])

    test_iter = chainer.iterators.SerialIterator(
        chainer.datasets.TupleDataset(X_test),
        settings['batch_size'], repeat=False, shuffle=False)

    kf = KFold(5, True, seed)

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        nn = MLP(settings['topology'], settings['dropout'], settings['dropout_input'])
        model = SCEClassifier(nn)

        train_iter = chainer.iterators.SerialIterator(
            chainer.datasets.TupleDataset(X_train, y_train), settings['batch_size'])
        fit(model, train_iter, settings, gpu)
        chainer.serializers.save_npz('nn_{}.npz'.format(fold_idx), autoencoder)

        valid_iter = chainer.iterators.SerialIterator(
            chainer.datasets.TupleDataset(X_valid),
            settings['batch_size'], repeat=False, shuffle=False)
        pred_valid[valid_idx] = F.sigmoid(predict(nn, valid_iter, gpu)).data[:, 0]

        auc = roc_auc_score(y_valid, pred_valid[valid_idx])
        logloss = log_loss(y_valid, pred_valid[valid_idx])
        print('[{}-fold] auc: {:.6f}, logloss: {:.6f}'.format(fold_idx, auc, logloss))

        pred_test += F.sigmoid(predict(nn, test_iter, gpu)).data[:, 0] / 5

    submission = pd.DataFrame({'id': test['id'], 'target': pred_test})
    submission.to_csv(submission_name, index=False)
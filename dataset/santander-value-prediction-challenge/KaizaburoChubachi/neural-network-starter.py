import numpy as np
import pandas as pd

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import training
from chainer.training import extensions

from sklearn.model_selection import KFold


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


class RMSERegressor(chainer.Chain):

    def __init__(self, base_model):
        super(RMSERegressor, self).__init__()
        with self.init_scope():
            self.base_model = base_model

    def __call__(self, X, y):
        y_hat = self.base_model(X)
        loss = F.sqrt(F.mean_squared_error(y.reshape(y_hat.shape), y_hat))
        chainer.report({'loss': loss}, self)
        return loss


def fit(model, iterator, settings, gpu, valid_iterator=None):
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
    # trainer.extend(extensions.ProgressBar())

    if valid_iterator is None:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))
    else:
        evaluator = extensions.Evaluator(valid_iterator, model, device=gpu)
        trainer.extend(evaluator, trigger=(1, 'epoch'))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

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
    gpu = -1
    eps = 1e-8
    settings_pred = {
        'topology': [-1, 192, 64, 64, 1],
        'dropout': 0.5,
        'dropout_input': 0.1,
        'reg_l2': 0.05,
        'learning_rate': 1e-2,
        'batch_size': 256,
        'learning_rate_decay': 0.995,
        'nb_epochs': 200
    }

    np.random.seed(seed)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()

    # load data
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    nunique = train.nunique()
    drop_cols = nunique[nunique == 1].index.tolist()

    X = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
    X = X.drop(drop_cols + ['ID', 'target'], axis=1).values.astype(np.float32)

    # normalize
    positive_index = X >= eps
    X[positive_index] = np.log1p(X[positive_index])
    mean_posi, std_posi = X[positive_index].mean(), X[positive_index].std()
    X[positive_index] = (X[positive_index] - mean_posi) / std_posi
    X[~positive_index] = 0

    # train nn regressor
    X_test = X[train.shape[0]:]
    X = X[:train.shape[0]]
    y = train['target'].values.astype(np.float32)
    settings_pred['topology'][0] = X.shape[1]

    # normalize target
    y = np.log1p(y) - mean_posi

    pred_valid = np.zeros(X.shape[0])
    pred_test = np.zeros(X_test.shape[0])

    test_iter = chainer.iterators.SerialIterator(
        chainer.datasets.TupleDataset(X_test),
        settings_pred['batch_size'], repeat=False, shuffle=False)

    kf = KFold(5, True, seed)

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        nn = MLP(settings_pred['topology'], settings_pred['dropout'], settings_pred['dropout_input'])
        model = RMSERegressor(nn)

        train_iter = chainer.iterators.SerialIterator(
            chainer.datasets.TupleDataset(X_train, y_train), settings_pred['batch_size'])
        valid_iter = chainer.iterators.SerialIterator(
            chainer.datasets.TupleDataset(X_valid, y_valid),
            settings_pred['batch_size'], repeat=False, shuffle=False)

        fit(model, train_iter, settings_pred, gpu, valid_iter)
        chainer.serializers.save_npz('nn_{}.npz'.format(fold_idx), nn)

        valid_iter = chainer.iterators.SerialIterator(
            chainer.datasets.TupleDataset(X_valid),
            settings_pred['batch_size'], repeat=False, shuffle=False)
        pred_valid[valid_idx] = predict(nn, valid_iter, gpu)[:, 0]

        rmse = np.mean((y_valid - pred_valid[valid_idx]) ** 2) ** 0.5
        print(f'[{fold_idx}-fold] rmse: {rmse:.6f}')

        pred_test += predict(nn, test_iter, gpu)[:, 0] / 5

    pred_test = np.expm1(pred_test + mean_posi)

    submission = pd.DataFrame({'ID': test['ID'], 'target': pred_test})
    submission.to_csv('submission.csv', index=False)

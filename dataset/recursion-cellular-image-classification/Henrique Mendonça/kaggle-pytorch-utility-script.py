## Utility script for kagglers using PyTorch
## to add it to your kernel click in 'File -> Add utility script'
## usage can be seen in https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50

import os, sys, gc
import random, math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import itertools as it
import scipy
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Optimizer

from tqdm.notebook import tqdm

import ignite
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import RunningAverage

from collections import OrderedDict

from sklearn import preprocessing


### General purpose:

def is_interactive():
    ''' Return True if inside a notebook/kernel in Edit mode, or False if committed '''
    try:
        from IPython import get_ipython
        return 'runtime' in get_ipython().config.IPKernelApp.connection_file
    except:
        return False

def filter_alphanum(string):
    ''' Remove non-alphanumerical characters from string '''
    return ''.join((filter(lambda x: x.isalnum(), string)))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_n_remap(model, model_path, device='cpu'):
    ''' Load a pytorch model remapping key names (keeping param order), useful if loading a model from another source '''
    model_state = torch.load(model_path, map_location=device)

    # A basic remapping is required
    mapping = { k:v for k,v in zip(model_state.keys(), model.state_dict().keys()) }
#         print(mapping)
    mapped_model_state = OrderedDict([
        (mapping[k], v) for k,v in model_state.items() if k in mapping.keys()
    ])

    model.load_state_dict(mapped_model_state, strict=False)
    return model


class Lookahead(Optimizer):
    ''' https://github.com/lonePatient/lookahead_pytorch '''
    def __init__(self, base_optimizer, alpha=0.5, k=5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                                for group in self.param_groups]
        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p,q in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha, p.data - q.data)
                p.data.copy_(q.data)
        return loss



### Model Visualizations

def normalize_channels(img):
    _min, _max = img.min(axis=(0,1)), img.std(axis=(0,1))
    img = (img - _min) / (_max - _min)
    return img

def plot_first_kernels(weight):
    ''' plot first filters of a model '''
    with torch.no_grad():
        filters = weight.detach().cpu().float().numpy().transpose([0,2,3,1])  # channels last
        filters = normalize_channels(filters)
        filters /= filters.max()
    n = filters.shape[0]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axs = plt.subplots(rows, cols)
    for c in range(cols):
        for r in range(rows):
            idx = r + c * rows
            if idx < n:
                axs[r,c].imshow(filters[idx])
            axs[r,c].set_axis_off()
    return fig, axs


def plot_norms(named_parameters, figsize=None):
    ''' plot l2 norm distribution for each layer of the given named parameters. e.g. model.named_parameters() '''
    from matplotlib import cm
    p = [0, 25, 50, 75, 100]
    with torch.no_grad():
        norms, names = [], []
        for name, param in named_parameters:
                param_flat = param.view(param.shape[0], -1)
                norms.append(np.percentile(torch.norm(param_flat, p=2, dim=1).cpu().numpy(), p))
                names.append(name)

    n = len(norms)
    inv_p = np.arange(len(p)-1,-1,-1)
    norms = np.array(norms)
    if figsize is None:
        figsize = (np.min([16, n]), 6)
    plt.figure(figsize=figsize)
    plt.yscale('log')
    for i,c in zip(inv_p, cm.get_cmap('inferno')(0.1+inv_p/len(p))):
        plt.bar(np.arange(n), norms[:,i], lw=1, color=c)
    plt.xticks(range(n), names, rotation="vertical")
    plt.xlabel("layers")
    plt.ylabel("norm distribution")
    plt.title("Kernel L2 Norms")
    plt.grid(True)
    plt.legend(labels=[f'{i}%' for i in p[::-1]])


def plot_grad_flow(named_parameters, figsize=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8#post_10'''
    from matplotlib.lines import Line2D
    avg_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if (p.grad is not None) and ("bias" not in n):
            layers.append(n)
            avg_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    if figsize is None:
        figsize = (np.min([16, len(avg_grads)]), 6)
    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), avg_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(layers)+1, lw=2, color="k" )
    plt.xticks(range(0, len(layers), 1), layers, rotation="vertical")
    plt.xlim(left=-1, right=len(layers))
    plt.xlabel("Layers")
    plt.ylabel("Gradient Magnitude")
    plt.yscale('log')
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def cosine_distance_heatmap(model, x0, x1, y):
    ''' plot cosine distances between samples of 2 batches '''
    with torch.no_grad():
        typ = next(iter(model.parameters()))[0].type()
        f0 = model.features(x0.type(typ))
        f1 = model.features(x1.type(typ))
        cosX = 1 - torch.mm(f0, f1.t()).cpu().numpy()
        del f0; del f1
        n = len(y)
        print('all-mean:', cosX.mean(), 'twins-mean:', cosX.trace()/n)

    idx = [f'{c}:{i:0>4}' for c,i in enumerate(y)]
    hm = pd.DataFrame(cosX, columns=idx, index=idx)
    plt.figure(figsize=(n, n//2))
    sns.heatmap(hm, annot=True, fmt=".2f")



### Loss functions:

class FocalLoss(nn.Module):
    ''' cross entropy focal loss '''
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            self.alpha = self.alpha.type(inputs.type(), non_blocking=True) # fix type and device
            alpha = self.alpha[targets]
        else:
            alpha = 1.

        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = alpha * (1-pt)**self.gamma * CE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()
        return F_loss


def cont_kappa(input, targets, activation=None):
    ''' continuos version of quadratic weighted kappa '''
    n = len(targets)
    y = targets.float().unsqueeze(0)
    pred = input.float().squeeze(-1).unsqueeze(0)
    if activation is not None:
        pred = activation(pred)
    wo = (pred - y)**2
    we = (pred - y.t())**2
    return 1 - (n * wo.sum() / we.sum())

kappa_loss = lambda pred, y: 1 - cont_kappa(pred, y)  # from 0 to 2 instead of 1 to -1


### PyTorch Ignite:

class Metrics(RunningAverage):
    "Metrics and history handler for ignite"
    def __init__(self, evaluator, eval_loader, output_transform, interactive=False):
        super().__init__(alpha=0.9, output_transform=output_transform)
        self.evaluator = evaluator
        self.eval_loader = eval_loader
        self.interactive = interactive
        self.validation_history = {}
        self.loss_history = []

    def attach(self, engine, name):
        super().attach(engine, name)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.run_evaluation)

    def compute(self):
        loss = super().compute()
        self.loss_history.append(loss)
        return loss

    def run_evaluation(self, engine):
        self.evaluator.run(self.eval_loader)
        if self.interactive:
            print(engine.state.epoch, self.evaluator.state.metrics)
        # save validation_history
        for k,v in self.evaluator.state.metrics.items():
            if k not in self.validation_history.keys():
                self.validation_history[k] = [v]
            else:
                self.validation_history[k].append(v)

    def plot(self, epoch_size=None, figsize=(16, 4)):
        plt.figure(figsize=figsize)
        ax = plt.subplot()
        ax.set_yscale('log')
        ax.set_xlabel('Batches processed')
        ax.set_ylabel("loss")
        ax.plot(self.loss_history, label='Train Loss')
        ax2 = ax.twinx()
        ax2.set_ylabel("score")
        for k,v in self.validation_history.items():
            if epoch_size is None:
                epoch_size = len(self.loss_history) // len(v)
            iters = np.arange(1, len(v)+1) * epoch_size
            if k == 'Loss':
                  ax.plot(iters, v, label='Valid Loss')
            else: ax2.plot(iters, v, label=k, ls='--')
        ax.legend(frameon=False, loc='upper left')
        ax2.legend(frameon=False, loc='lower left')




if __name__ == "__main__":
    pass

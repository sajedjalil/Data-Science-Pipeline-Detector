import os, sys, random, math
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

from tqdm import tqdm_notebook as tqdm

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import RunningAverage

from collections import OrderedDict

from sklearn import preprocessing
import cv2

from albumentations import *
from albumentations.pytorch import ToTensor
from collections import OrderedDict

### General purpose:

def is_interactive():
   return 'runtime' in get_ipython().config.IPKernelApp.connection_file


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_n_remap(model, model_path):
    model_state = torch.load(model_path)

    # A basic remapping is required
    mapping = {
        k: v for k, v in zip(model_state.keys(), model.state_dict().keys())
    }
    mapped_model_state = OrderedDict([
        (mapping[k], v) for k, v in model_state.items()
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

    
### RCIC dataset:

def get_fold(df, fold, kfolds=4):
    exp_counts = df.groupby('cell_type').experiment.nunique()
    valid = []
    for t,c in zip(exp_counts.index, exp_counts.values):
        for i in range(1-fold, -c, -kfolds):
            valid.append(f'{t}-{(c+i):02}')
    return valid


def get_rcic_exp_stats(path_data):
    ''' Get pixel stats group by experiment '''
    stats = pd.read_csv(path_data+'/pixel_stats.csv')
    stats['var'] = stats['std']**2.
    exp_stats = stats.groupby(['experiment', 'channel']).mean()
    ## fix non-linear std
    exp_stats['std'] = np.sqrt(exp_stats['var'])
    return exp_stats


def normalize_channels(img):
    _min, _max = img.min(axis=(0,1)), img.std(axis=(0,1))
    img = (img - _min) / (_max - _min)
    return img

def show_norm_images(images):
    plt.figure(figsize=(16, 8))
    plt.axis("off")
    plt.title("Training Images")
    _ = plt.imshow( # show every second channel
        normalize_channels(
            vutils.make_grid(images[...,::2], padding=2, normalize=False).cpu().numpy().transpose((1, 2, 0)),
        )
    )


class ExpNormRCICdataset(Dataset):
    ''' Multi channel datatset normalised by experiment image stats '''
    def __init__(self, df, img_dir, target, df_ctrl=None, mode='train', sites=[1,2], channels=[1,2,3,4,5,6], img_stats=None, transform=None):
        self.df = df
        self.df_ctrl = df_ctrl
        self.channels = channels
        self.sites = sites
        self.target = target
        self.mode = mode
        self.img_dir = img_dir
        self.stats = img_stats
        self.transform = transform
        
    @staticmethod
    def _load_channel(file_name):
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        return np.float32(img)

    def _load_img(self, experiment, paths):
        img = [self._load_channel(img_path) for img_path in paths]

        ## norm
        if self.stats is not None:
            stats = self.stats.loc[experiment, ['mean', 'std']]
            # mean subtract
            img = [i-m for i,m in zip(img, stats['mean'].values)]
            # norm to 1 std
            img = [i/s for i,s in zip(img, stats['std'].values)]
        
        img = np.stack(img, axis=-1)
#         print(stats)
#         print(img.shape, img.mean(axis=(1,0)).tolist(), img.std(axis=(1,0)).tolist(), float(img.min()), float(img.max()))
        if self.transform:
            img = self.transform(image=img)['image']
        return img

    def _get_ctrl_img(self, index, site):
        experiment, plate = self.df.iloc[index].experiment, self.df.iloc[index].plate
        # get a random control well in the same plate
        well = self.df_ctrl[(experiment == self.df_ctrl.experiment) & (plate == self.df_ctrl.plate)].sample(1).well.values[0]
        paths = [os.path.join(self.img_dir, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png')
                 for channel in self.channels]
        return self._load_img(experiment, paths)
        
    def _get_img(self, index, site):
        experiment, well, plate = self.df.iloc[index].experiment, self.df.iloc[index].well, self.df.iloc[index].plate
        paths = [os.path.join(self.img_dir, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png')
                 for channel in self.channels]
        return self._load_img(self.df.iloc[index].experiment, paths)
        
    def __getitem__(self, index):
        if self.mode == 'train':
            # returns a random site
            img = self._get_img(index, random.choice(self.sites))
            # get a random control image
            if self.df_ctrl is not None:
                ctrl_img = self._get_ctrl_img(index, random.choice(self.sites))
                img = torch.cat([img, ctrl_img], 0)
            return img, self.df.iloc[index][self.target]
        else:
            # returns raw images of all available sites
            imgs = []
            for site in self.sites:
                img = self._get_img(index, site)
                # get control image
                if self.df_ctrl is not None:
                    ctrl_img = self._get_ctrl_img(index, site)
                    img = torch.cat([img, ctrl_img], 0)
                imgs.append(img)
            return imgs

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.df)

class AverageCrop(RandomCrop):
    """Crop a random part of the input maintaining the mean pixel value of the image.
    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        images
    Image types:
        uint8, float32
    """
    def apply(self, img, **_):
        ## find a reasonable crop
        mu = img.mean()
        best, best_mu = None, -np.inf
        for i in range(7):
            crop = super().apply(img, **self.get_params())
            crop_mu = crop.mean()
            if crop_mu >= best_mu:
                best, best_mu = crop, crop_mu
            if best_mu >= mu:
                break  ## else try a new crop
        return best
    
def tta9crop(input, idx, resolution):
    ''' simple 9-crop 8-dihedral TTA '''
    raw_image_size = input.shape[-1]
    crops = [0, (raw_image_size - resolution)//2, raw_image_size-resolution]
    u = crops[idx//3]
    v = crops[idx%3]
    x = input[..., u:u+resolution, v:v+resolution] # crop it
    if idx in [1,4,7]: x = x.flip(-2)
    if idx in [3,4,5]: x = x.flip(-1)
    if idx in [0,4,8]: x = x.transpose(-1,-2)
    return x

def tta4crop(input, idx, resolution):
    ''' simple 4-crop 8-dihedral TTA '''
    raw_image_size = input.shape[-1]
    crops = [0, raw_image_size-resolution]
    u = crops[idx//2]
    v = crops[idx%2]
    x = input[..., u:u+resolution, v:v+resolution] # crop it
    if idx in [1]: x = x.flip(-2)
    if idx in [2]: x = x.flip(-1)
    if idx in [3]: x = x.transpose(-1,-2)
    return x


def plot_first_kernels(weight):
    with torch.no_grad():
        filters = weight.detach().cpu().float().numpy().transpose([0,2,3,1])  # channels last
        filters = normalize_channels(filters)
        filters /= filters.max()
    n = filters.shape[0]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axs = plt.subplots(rows, cols*2)
    for c in range(cols):
        for r in range(rows):
            idx = r + c * rows
            if idx < n:
                # every 2nd channel
                axs[r,c*2].imshow(filters[idx,...,::2])
                # next 3 channels
                axs[r,c*2+1].imshow(filters[idx,...,1::2])
            axs[r,c*2].set_axis_off()
            axs[r,c*2+1].set_axis_off()


def plot_norms(named_parameters, figsize=None):
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
    plt.show()


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
    plt.show()

def cosine_distance_heatmap(model, batch):
    with torch.no_grad():
        typ = next(iter(model.parameters()))[0].type()
        f0 = model.features(batch[0].type(typ))
        f1 = model.features(batch[1].type(typ))
        y  = batch[2]
        cosX = 1 - torch.mm(f0, f1.t()).cpu().numpy()
        del batch; del f0; del f1
        n = len(y)
        print('all-mean:', cosX.mean(), 'twins-mean:', cosX.trace()/n)

    idx = [f'{c}:{i:0>4}' for c,i in enumerate(y)]
    hm = pd.DataFrame(cosX, columns=idx, index=idx)
    fig = plt.figure(figsize=(n,n//2))
    sns.heatmap(hm, annot=True, fmt=".2f")
    plt.show()


### Loss functions:

class FocalLoss(nn.Module):
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


class MarginCosineProduct(nn.Module):
    """large margin cosine distance: https://github.com/MuggleWang/CosFace_pytorch
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class LMCL_loss(nn.Module):
    """ https://github.com/YirongMao/softmax_variants
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, s=7.00, m=0.2):
        super(LMCL_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits


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

    def run_evaluation(self, engine=None):
        self.evaluator.run(self.eval_loader)
        if self.interactive:
            print(self.evaluator.state.metrics)
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
        ax.plot()


## Post-processing

## thanks to https://www.kaggle.com/aharless
## based on https://www.kaggle.com/christopherberner/hungarian-algorithm-to-optimize-sirna-prediction
## and https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak
sirna_lists = None
def get_sirna_list(experiment_plate):
    global sirna_lists
    if sirna_lists is None:
        sirna_lists = pd.read_csv(glob.glob('../input/*/sirnas_for_each_experiment_plate.csv')[0], index_col=0)
    rawlist = list(set(sirna_lists.loc[experiment_plate].values.tolist()) - {9999})
    return(sorted(rawlist))

def get_experiment_plate(name): return('_'.join(name.split('_')[:2]))

def assign_sirna(df):
    predicted = np.zeros(len(df)).astype(np.uint16)
    plates = np.unique(list(map(get_experiment_plate, df.index.values)))
    for plate in plates:
        indices = list(map(lambda name: name.startswith(plate), df.index.values))
        sirnas = get_sirna_list(plate)
        plate_nprobabilities = -df.loc[indices,sirnas]
        _, sirna_subscripts = scipy.optimize.linear_sum_assignment(plate_nprobabilities)
        predicted[indices] = np.array([sirnas[s] for s in sirna_subscripts])
    return predicted

def remove_leaked_sirna(df, probs):
    all_sirnas = set(probs.columns.values)
    experiment_plate_for_each_case = list(map(get_experiment_plate, df.index.values))
    experiment_plates = np.unique(experiment_plate_for_each_case)
    for ep in experiment_plates:
        probs.loc[np.array(experiment_plate_for_each_case)==ep, 
                  sorted(list(all_sirnas - set(get_sirna_list(ep))))] = 0
    return(probs)

## thanks to https://www.kaggle.com/giuliasavorgnan
def normalize_both_ways(preds, validation_data=None):
    score = None
    sirna_preds = None
    ## re-normalise accross each row
    preds = preds.divide(preds.sum(axis=1),axis=0)
    leak_preds_norm1 = preds
    leak_preds_norm1["experiment_plate"] = [i.split("_")[0]+"_"+i.split("_")[1] 
                                            for i in leak_preds_norm1.index]
    group_list = []
    # normalise across siRNA's (each siRNA is is equally likely to appear in each experiment)
    for exp_pla, group in leak_preds_norm1.groupby("experiment_plate"):
        # iteration through groups preserves alphabetical order of experiment_plate
        group_list.append(group.drop(labels="experiment_plate", 
            axis=1).divide(group.drop(labels="experiment_plate",
            axis=1).sum(axis=0), axis=1).copy(deep=True).fillna(0))
    preds = pd.concat(group_list)
    if validation_data is not None:
        sirna_preds = assign_sirna(preds)
        score = np.mean(sirna_preds == validation_data)
    return(preds, score, sirna_preds)


if __name__ == "__main__":
    ## smoke test post processing
    assert len(get_sirna_list('HEPG2-08_1')) == 277
    sirna_lists.to_csv('sirnas_for_each_experiment_plate.csv')

    ## test it on the datatset
    path_data = '/kaggle/input/recursion-cellular-image-classification'
    df = pd.read_csv(path_data+'/train.csv').sample(frac=1).reset_index(drop=True)  # shuffle df
    df_test = pd.read_csv(path_data+'/test.csv')

    df['cell_type'] = df.experiment.apply(lambda s: s[:-3])
    df_test['cell_type'] = df.experiment.apply(lambda s: s[:-3])

    n_classes = df.sirna.nunique()
    print(n_classes)
    valid_experiments = get_fold(df, 1)
    
    df_train, df_valid = df[~df.experiment.isin(valid_experiments)], df[df.experiment.isin(valid_experiments)]
    print(df_train.shape, df_valid.shape, df_test.shape)

    valid_transform = Compose([
        AverageCrop(123, 380),
        ToTensor(),
    ])
    tmp = ExpNormRCICdataset(df_train, path_data+'/train', target='sirna', mode='train', transform=valid_transform)
    print(tmp[3][0].shape, tmp[3][1])
    
#     loss = FocalLoss()
#     print(loss(tmp[0][0].unsqueeze(0), torch.tensor(tmp[0][1]).unsqueeze(0)))

    tmp = ExpNormRCICdataset(df_test, path_data+'/test', target='sirna', mode='test', transform=ToTensor())
    print(tmp[42][1].shape)

    raw_image_size = tmp[0][0].shape[0]
    print(raw_image_size)
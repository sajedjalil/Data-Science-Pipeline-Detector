import sys,os
def installPkgs(pkgs,verbose=False):
    for pkgName in pkgs:
        
        if pkgName not in sys.modules:
            os.system(f"/opt/conda/bin/pip install {pkgName}")
        else:
            if verbose:
                print(f"{pkgName} already installed")
            
            

import pickle
def _save(x,fname):
    pickle.dump(x, open(fname, 'wb'))
def _load(fname):
    return pickle.load(open(fname, 'rb'))

            
            
import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv', residual=False):
    
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if i == 0 and residual:
                self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
                                                    padding, batch_norm, residual, first=True))
            else:
                self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
                                                    padding, batch_norm, residual))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode,
                                            padding, batch_norm, residual))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        res = self.last(x)
        return self.softmax(res)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, residual=False, first=False):
        super(UNetConvBlock, self).__init__()
        self.residual = residual
        self.out_size = out_size
        self.in_size = in_size
        self.batch_norm = batch_norm
        self.first = first
        self.residual_input_conv = nn.Conv2d(self.in_size, self.out_size, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm2d(self.out_size)

        if residual:
            padding = 1
        block = []

        if residual and not first:
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(in_size))

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))

        if not residual:
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            if self.in_size != self.out_size:
                x = self.residual_input_conv(x)
                x = self.residual_batchnorm(x)
            out = out + x

        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, residual=False):
        super(UNetUpBlock, self).__init__()
        self.residual = residual
        self.in_size = in_size
        self.out_size = out_size
        self.residual_input_conv = nn.Conv2d(self.in_size, self.out_size, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm2d(self.out_size)

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    @staticmethod
    def center_crop(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out_orig = torch.cat([up, crop1], 1)
        out = self.conv_block(out_orig)
        if self.residual:
            if self.in_size != self.out_size:
                out_orig = self.residual_input_conv(out_orig)
                out_orig = self.residual_batchnorm(out_orig)
            out = out + out_orig

        return out
    
    
# pytorch-ard 改
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from functools import reduce
import operator


class LinearARD(nn.Module):
    """
    Dense layer implementation with weights ARD-prior (arxiv:1701.05369)
    """

    def __init__(self, in_features, out_features, bias=True, thresh=3, ard_init=-10):
        super(LinearARD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.thresh = thresh
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.ard_init = ard_init
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def forward(self, input):
        if self.training:
            epsilon = self.weight.new(self.weight.shape).normal_()
            W = self.weight + epsilon * torch.exp(self.log_sigma2 / 2)
        else:
            W = self.weights_clipped
        return F.linear(input, W) + self.bias

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.02)
        if self.bias is not None:
            self.bias.data.zero_()
        self.log_sigma2.data.fill_(self.ard_init)

    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - \
            0.5 * torch.log1p(torch.exp(-self.log_alpha)) + C
        return -torch.sum(mdkl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -10, 10)
    

class ELBOLoss(nn.Module):
    def __init__(self, net, loss_fn,kl_weight=1.):
        super(ELBOLoss, self).__init__()
        self.loss_fn = loss_fn
        self.net = net
        self.kl_weight = kl_weight

    def forward(self, input, target, loss_weight=1.):
        assert not target.requires_grad
        # Estimate ELBO
        return loss_weight * self.loss_fn(input, target)  \
            + self.kl_weight * get_ard_reg(self.net)


def get_ard_reg(module):
    """
    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    #if isinstance(module, LinearARD) or isinstance(module, Conv2dARD):
    if isinstance(module, LinearARD) :
        return module.get_reg()
    elif hasattr(module, 'children'):
        return sum([get_ard_reg(submodule) for submodule in module.children()])
    return 0


def _get_dropped_params_cnt(module):
    if hasattr(module, 'get_dropped_params_cnt'):
        return module.get_dropped_params_cnt()
    elif hasattr(module, 'children'):
        return sum([_get_dropped_params_cnt(submodule) for submodule in module.children()])
    return 0


def _get_params_cnt(module):
    #if any([isinstance(module, l) for l in [LinearARD, Conv2dARD]]):
    if any([isinstance(module, l) for l in [LinearARD]]):
        return reduce(operator.mul, module.weight.shape, 1)
    elif hasattr(module, 'children'):
        return sum(
            [_get_params_cnt(submodule) for submodule in module.children()])
    return sum(p.numel() for p in module.parameters())


def get_dropped_params_ratio(model):
    return _get_dropped_params_cnt(model) * 1.0 / _get_params_cnt(model)


# ModelEMA
from copy import deepcopy
import math

class ModelEMA:

    def __init__(self, model, decay=0.9999, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.updates = 0  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.ema.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema.state_dict()

            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.ema, k, getattr(model, k))

import numpy as np
def calcScore(fvc_true, fvc_pred, sigma):
    
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(fvc_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2)
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return np.mean(metric)



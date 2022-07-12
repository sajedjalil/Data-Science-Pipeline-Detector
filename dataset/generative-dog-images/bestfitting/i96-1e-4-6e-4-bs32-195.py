# version: 2019-08-08 00:22:20 start
from timeit import default_timer as timer
_kaggle_start_ = timer()
######################## origin_dogs.py start ########################
import xml.etree.ElementTree as ET
import torchvision
import os
import mlcrate as mlc
import numpy as np
from PIL import Image
from tqdm import tqdm
opj=os.path.join
ope=os.path.exists
from PIL import Image, ImageDraw
import shutil
data_dir='../input/'
out_dir=opj('../output/data/generative-dog-images/origin_dogs')
if ope(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)
input_img_dir=opj(data_dir,'all-dogs/all-dogs')
annotation_dir=opj(data_dir,'annotation/Annotation')
annotation_outdir = f'{out_dir}_annotation'
if ope(annotation_outdir):
    shutil.rmtree(annotation_outdir)
os.makedirs(annotation_outdir, exist_ok=True)
annotation_list = os.listdir(annotation_dir)
for annotation in annotation_list:
    annotation_dirname = annotation.split('-')[1]
    shutil.copytree(f'{annotation_dir}/{annotation}', f'{annotation_outdir}/{annotation_dirname}')
def process_an_image(dog_fname):
    dog_fullname=opj(input_img_dir, dog_fname)
    img = torchvision.datasets.folder.default_loader(dog_fullname)  # default loader
    annotation_basename = os.path.splitext(dog_fname)[0]
    annotation_dirname = next(dirname for dirname in os.listdir(annotation_dir) if
                              dirname.startswith(annotation_basename.split('_')[0]))
    dog_type_dir=opj(out_dir,annotation_dirname.split('-')[1])
    os.makedirs(dog_type_dir,exist_ok=True)
    img.save(opj(dog_type_dir,dog_fname))
    return dog_fname
class MySuperpool(mlc.SuperPool):
    def map(self, func, array, chunksize=16, description=''):
        res = []
        def func_tracked(args):
            x, i = args
            return func(x), i
        array_tracked = zip(array, range(len(array)))
        desc = '[mlcrate] {} CPUs{}'.format(self.n_cpu, ' - {}'.format(description) if description else '')
        for out in self.tqdm.tqdm(self.pool.uimap(func_tracked, array_tracked, chunksize=chunksize),
                                  total=len(array), desc=desc, smoothing=0.05, disable=True):
            res.append(out)
        actual_res = [r[0] for r in sorted(res, key=lambda r: r[1])]
        return actual_res
dog_fnames=os.listdir(input_img_dir)
pool=MySuperpool()
imgs=pool.map(process_an_image,dog_fnames)
print(out_dir)
######################### origin_dogs.py end #########################
######################## train.py start ########################
import functools
import math
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
######################## inception_utils.py start ########################
from scipy import linalg # For numpy FID
import time
from torchvision.models.inception import inception_v3
class inception_utils_WrapInception(nn.Module):
  def __init__(self, net):
    super(inception_utils_WrapInception,self).__init__()
    self.net = net
    self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
    self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
  def forward(self, x):
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    if x.shape[2] != 299 or x.shape[3] != 299:
      x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    x = self.net.Conv2d_1a_3x3(x)
    x = self.net.Conv2d_2a_3x3(x)
    x = self.net.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = self.net.Conv2d_3b_1x1(x)
    x = self.net.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = self.net.Mixed_5b(x)
    x = self.net.Mixed_5c(x)
    x = self.net.Mixed_5d(x)
    x = self.net.Mixed_6a(x)
    x = self.net.Mixed_6b(x)
    x = self.net.Mixed_6c(x)
    x = self.net.Mixed_6d(x)
    x = self.net.Mixed_6e(x)
    x = self.net.Mixed_7a(x)
    x = self.net.Mixed_7b(x)
    x = self.net.Mixed_7c(x)
    pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    return pool, logits
def inception_utils_torch_cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
def inception_utils_sqrt_newton_schulz(A, numIters, dtype=None):
  with torch.no_grad():
    if dtype is None:
      dtype = A.type()
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
      T = 0.5*(3.0*I - Z.bmm(Y))
      Y = Y.bmm(T)
      Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
  return sA
def inception_utils_numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)
  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)
  assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'
  diff = mu1 - mu2
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = ('fid calculation produces singular product; '
           'adding %s to diagonal of cov estimates') % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
  if np.iscomplexobj(covmean):
    print('wat')
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError('Imaginary component {}'.format(m))
    covmean = covmean.real  
  tr_covmean = np.trace(covmean) 
  out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
  return out
def inception_utils_torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'
  diff = mu1 - mu2
  covmean = inception_utils_sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()  
  out = (diff.dot(diff) +  torch.trace(sigma1) + torch.trace(sigma2)
         - 2 * torch.trace(covmean))
  return out
def inception_utils_calculate_inception_score(pred, num_splits=10):
  scores = []
  for index in range(num_splits):
    pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
    kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
    kl_inception = np.mean(np.sum(kl_inception, 1))
    scores.append(np.exp(kl_inception))
  return np.mean(scores), np.std(scores)
def inception_utils_accumulate_inception_activations(sample, net, num_inception_images=50000):
  pool, logits, labels = [], [], []
  while (torch.cat(logits, 0).shape[0] if len(logits) else 0) < num_inception_images:
    with torch.no_grad():
      images, labels_val = sample()
      pool_val, logits_val = net(images.float())
      pool += [pool_val]
      logits += [F.softmax(logits_val, 1)]
      labels += [labels_val]
  return torch.cat(pool, 0), torch.cat(logits, 0), torch.cat(labels, 0)
def inception_utils_load_inception_net(parallel=False):
  inception_model = inception_v3(pretrained=True, transform_input=False)
  inception_model = inception_utils_WrapInception(inception_model.eval()).cuda()
  if parallel:
    print('Parallelizing Inception module...')
    inception_model = nn.DataParallel(inception_model)
  return inception_model
def inception_utils_prepare_inception_metrics(base_root,dataset, parallel, no_fid=False):
  dataset = dataset.strip('_hdf5')
  out_meta_dir=f'{base_root}/meta/'
  out_fname=out_meta_dir+dataset+'_inception_moments.npz'
  data_mu = np.load(out_fname)['mu']
  data_sigma = np.load(out_fname)['sigma']
  net = inception_utils_load_inception_net(parallel)
  def get_inception_metrics(sample, num_inception_images, num_splits=10, 
                            prints=True, use_torch=True):
    if prints:
      print('Gathering activations...')
    pool, logits, labels = inception_utils_accumulate_inception_activations(sample, net, num_inception_images)
    if prints:  
      print('Calculating Inception Score...')
    IS_mean, IS_std = inception_utils_calculate_inception_score(logits.cpu().numpy(), num_splits)
    if no_fid:
      FID = 9999.0
    else:
      if prints:
        print('Calculating means and covariances...')
      if use_torch:
        mu, sigma = torch.mean(pool, 0), inception_utils_torch_cov(pool, rowvar=False)
      else:
        mu, sigma = np.mean(pool.cpu().numpy(), axis=0), np.cov(pool.cpu().numpy(), rowvar=False)
      if prints:
        print('Covariances calculated, getting FID...')
      if use_torch:
        FID = inception_utils_torch_calculate_frechet_distance(mu, sigma, torch.tensor(data_mu).float().cuda(), torch.tensor(data_sigma).float().cuda())
        FID = float(FID.cpu().numpy())
      else:
        FID = inception_utils_numpy_calculate_frechet_distance(mu.cpu().numpy(), sigma.cpu().numpy(), data_mu, data_sigma)
    del mu, sigma, pool, logits, labels
    return IS_mean, IS_std, FID
  return get_inception_metrics
######################### inception_utils.py end #########################
######################## utils.py start ########################
import sys
import datetime
import json
import pickle
from argparse import ArgumentParser
######################## animal_hash.py start ########################
######################### animal_hash.py end #########################
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
######################## datasets.py start ########################
import os.path
import torchvision.datasets as dset
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
import os,torchvision
dset_IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
def dset_is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in dset_IMG_EXTENSIONS)
def dset_dset_find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
def dset_pil_loader(path):
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')
def dset_accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    return dset_pil_loader(path)
def dset_check_img(img):
  os.makedirs('/data/tmp/dogs',exist_ok=True)
  torchvision.utils.save_image((img+1)/2, f'/data/tmp/dogs/img_{str(np.random.randint(0,1000)).zfill(4)}.jpg')
def dset_default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    return dset_accimage_loader(path)
  else:
    return dset_pil_loader(path)
from torch.utils.data import Dataset
class dset_DogDataset(Dataset):
  def __init__(self, root, transform,image_size=64,load_in_mem=True,index_filename='imagenet_imgs.npz', **kwargs):
    self.transform1 = transforms.Resize(image_size, kwargs['resize_mode'])
    self.transform2 = transform
    cached_fname = root#
    self.cropped_imgs = pickle.load(open(cached_fname, 'rb'))
    self.imgs = []
    self.labels=[]
    classes=[]
    for dog_name, dog_type, img_cropped in tqdm(self.cropped_imgs):
      if self.transform1 is not None:
        img = self.transform1(img_cropped)
      if dog_type not in classes:
        classes.append(dog_type)
      self.imgs.append(img)
      self.labels.append(dog_type)
    self.classes=classes
    self.class_to_idx = {classes[i]: i for i in range(len(classes))}
  def __getitem__(self, index):
    img = self.imgs[index]
    if self.transform2 is not None:
      img = self.transform2(img)
    label=self.class_to_idx[self.labels[index]]
    return img,label
  def __len__(self):
    return len(self.imgs)
def dset_get_image_bboxes(image_path):
    bbox_fname = image_path.replace('origin_dogs', 'origin_dogs_annotation')
    bbox_fname = os.path.splitext(bbox_fname)[0]
    tree = ET.parse(bbox_fname)
    root = tree.getroot()
    objects = root.findall('object')
    bboxes = []
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox = (xmin, ymin, xmax, ymax)
        bboxes.append(bbox)
    return bboxes
def dset_make_dataset(dir, class_to_idx, use_bbox):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if dset_is_image_file(fname):
                    image_path = os.path.join(root, fname)
                    if use_bbox:
                        bboxes = dset_get_image_bboxes(image_path)
                        item = (image_path, class_to_idx[target], np.array(bboxes, dtype=int))
                    else:
                        item = (image_path, class_to_idx[target])
                    images.append(item)
    return images
def dset_get_origin_max_crop(img, bbox, image_size):
    xmin, ymin, xmax, ymax = bbox
    size = img.size
    bbox_size = (xmax - xmin, ymax - ymin)
    extend = (np.max(bbox_size) - np.min(bbox_size)) // 2
    if bbox_size[1] > bbox_size[0]:
        left = bbox[0]
        right = size[0] - bbox[2]
        max_extend = np.min((left, right))
        extend = np.minimum(extend, max_extend)
        bbox[0] -= extend
        bbox[2] += extend
    else:
        top = bbox[1]
        bottom = size[1] - bbox[3]
        max_extend = np.min((top, bottom))
        extend = np.minimum(extend, max_extend)
        bbox[1] -= extend
        bbox[3] += extend
    cropped = img.crop(bbox)
    cropped = cropped.resize((image_size, image_size), Image.ANTIALIAS)
    return cropped
class dset_ImageFolder(data.Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               loader=dset_default_loader, load_in_mem=False, 
               index_filename='imagenet_imgs.npz', **kwargs):
    classes, class_to_idx = dset_dset_find_classes(root)
    self.crop_mode = kwargs['crop_mode']
    self.image_size = kwargs['image_size']
    self.use_bbox = self.crop_mode > 0
    if os.path.exists(index_filename):
      print('Loading pre-saved Index file %s...' % index_filename)
      imgs = np.load(index_filename)['imgs']
    else:
      print('Generating  Index file %s...' % index_filename)
      imgs = dset_make_dataset(root, class_to_idx, self.use_bbox)
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                           "Supported image extensions are: " + ",".join(dset_IMG_EXTENSIONS)))
    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.debug = kwargs['debug']
    if self.debug:
        self.imgs = self.imgs[:1000]
    print('Loading all images into memory...')
    self.data, self.labels, self.cropped_data = [], [], []
    for index in tqdm(range(len(self.imgs)),desc='load images'):
      path, target = imgs[index][0], imgs[index][1]
      img = self.loader(path)
      if self.use_bbox:
        bboxes = imgs[index][2]
        cropped = [dset_get_origin_max_crop(img, bbox, self.image_size) for bbox in bboxes]
        if self.crop_mode >= 2:
            img = transforms.Resize(self.image_size, Image.ANTIALIAS)(img)
        else:
            img = None
        self.cropped_data.append(cropped)
      else:
        img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
      self.data.append(img)
      self.labels.append(target)
  def __getitem__(self, index):
    img = self.data[index]
    target = self.labels[index]
    if self.crop_mode > 0:
      cropped_list = self.cropped_data[index]
      ix = np.random.randint(len(cropped_list))
      cropped = cropped_list[ix]
      if self.crop_mode == 1:
          img = cropped
      elif self.crop_mode == 2:
          ix = np.random.randint(2)
          if ix == 1:
              img = cropped
      elif self.crop_mode == 3:
          ix = np.random.randint(3)
          if ix == 1:
              img = cropped
          elif ix == 2:
              img = dset_RandomCropLongEdge()(img)
      elif self.crop_mode == 4:
          if len(cropped_list) > 1:
              img = cropped
          else:
              ix = np.random.randint(3)
              if ix == 1:
                  img = cropped
              elif ix == 2:
                  img = dset_RandomCropLongEdge()(img)
      elif self.crop_mode == 5:
          v = np.random.rand()
          if v < 0.4:
              img = cropped
          elif v < 0.8:
              img = dset_RandomCropLongEdge()(img)
      elif self.crop_mode == 6:
          v = np.random.rand()
          if v < 0.4:
              img = cropped
          elif v < 0.6:
              img = dset_RandomCropLongEdge()(img)
      elif self.crop_mode == 7:
          v = np.random.rand()
          if v < 0.2:
              img = cropped
          elif v < 0.6:
              img = dset_RandomCropLongEdge()(img)
      elif self.crop_mode == 8:
          v = np.random.rand()
          if v < 0.5:
              img = cropped
          else:
              img = dset_RandomCropLongEdge()(img)
      img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
    if self.debug:
        output_dir = '/data4/data/gan_dogs/result/tmp/crop_mode%d'%self.crop_mode
        os.makedirs(output_dir, exist_ok=True)
        if len(os.listdir(output_dir)) < 150:
            img.save(f'{output_dir}/{str(np.random.rand())}.jpg')
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img, int(target)
  def __len__(self):
    return len(self.imgs)
  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str
class dset_RandomCropLongEdge(object):
  def __call__(self, img):
    size = (min(img.size), min(img.size))
    i = (0 if size[0] == img.size[0]
          else np.random.randint(low=0,high=img.size[0] - size[0]))
    j = (0 if size[1] == img.size[1]
          else np.random.randint(low=0,high=img.size[1] - size[1]))
    return transforms.functional.crop(img, j, i, size[0], size[1])
  def __repr__(self):
    return self.__class__.__name__
import h5py as h5
class dset_ILSVRC_HDF5(data.Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               load_in_mem=False, train=True,download=False, validate_seed=0,
               val_split=0, **kwargs): # last four are dummies
    self.root = root
    self.num_imgs = len(h5.File(root, 'r')['labels'])
    self.target_transform = target_transform   
    self.transform = transform
    self.load_in_mem = load_in_mem
    if self.load_in_mem:
      print('Loading %s into memory...' % root)
      with h5.File(root,'r') as f:
        self.data = f['imgs'][:]
        self.labels = f['labels'][:]
  def __getitem__(self, index):
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]
    else:
      with h5.File(self.root,'r') as f:
        img = f['imgs'][index]
        target = f['labels'][index]
    img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img, int(target)
  def __len__(self):
      return self.num_imgs
class dset_CIFAR10(dset.CIFAR10):
  def __init__(self, root, train=True,
           transform=None, target_transform=None,
           download=True, validate_seed=0,
           val_split=0, load_in_mem=True, **kwargs):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train  # training set or test set
    self.val_split = val_split
    if download:
      self.download()
    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.' +
                           ' You can use download=True to download it')
    self.data = []
    self.labels= []
    for fentry in self.train_list:
      f = fentry[0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data.append(entry['data'])
      if 'labels' in entry:
        self.labels += entry['labels']
      else:
        self.labels += entry['fine_labels']
      fo.close()
    self.data = np.concatenate(self.data)
    if self.val_split > 0:
      label_indices = [[] for _ in range(max(self.labels)+1)]
      for i,l in enumerate(self.labels):
        label_indices[l] += [i]  
      label_indices = np.asarray(label_indices)
      np.random.seed(validate_seed)
      self.val_indices = []           
      for l_i in label_indices:
        self.val_indices += list(l_i[np.random.choice(len(l_i), int(len(self.data) * val_split) // (max(self.labels) + 1) ,replace=False)])
    if self.train=='validate':    
      self.data = self.data[self.val_indices]
      self.labels = list(np.asarray(self.labels)[self.val_indices])
      self.data = self.data.reshape((int(50e3 * self.val_split), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    elif self.train:
      print(np.shape(self.data))
      if self.val_split > 0:
        self.data = np.delete(self.data,self.val_indices,axis=0)
        self.labels = list(np.delete(np.asarray(self.labels),self.val_indices,axis=0))
      self.data = self.data.reshape((int(50e3 * (1.-self.val_split)), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    else:
      f = self.test_list[0][0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data = entry['data']
      if 'labels' in entry:
        self.labels = entry['labels']
      else:
        self.labels = entry['fine_labels']
      fo.close()
      self.data = self.data.reshape((10000, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
  def __getitem__(self, index):
    img, target = self.data[index], self.labels[index]
    img = Image.fromarray(img)
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img, target
  def __len__(self):
      return len(self.data)
class dset_CIFAR100(dset_CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
######################### datasets.py end #########################
def utils_prepare_parser():
  usage = 'Parser for all scripts.'
  parser = ArgumentParser(description=usage)
  parser.add_argument('-f', default=None, type=str)
  parser.add_argument(
    '--dataset', type=str, default='DogOrigin96',
    help='Which Dataset to train on, out of I128, I256, C10, C100;'
         'Append "_hdf5" to use the hdf5 version for ISLVRC '
         '(default: %(default)s)')
  parser.add_argument(
    '--augment', type=int, default=1,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers; consider using less for HDF5 '
         '(default: %(default)s)')
  parser.add_argument(
    '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
    help='Pin data into memory through dataloader? (default: %(default)s)') 
  parser.add_argument(
    '--shuffle', action='store_true', default=True,
    help='Shuffle the data (strongly recommended)? (default: %(default)s)')
  parser.add_argument(
    '--load_in_mem', action='store_true', default=False,
    help='Load all data into memory? (default: %(default)s)')
  parser.add_argument(
    '--use_multiepoch_sampler', action='store_true', default=False,
    help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')
  parser.add_argument(
    '--model', type=str, default='BigGAN',
    help='Name of the model module (default: %(default)s)')
  parser.add_argument(
    '--G_param', type=str, default='SN',
    help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
          ' or None (default: %(default)s)')
  parser.add_argument(
    '--D_param', type=str, default='SN',
    help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
         ' or None (default: %(default)s)')    
  parser.add_argument(
    '--G_ch', type=int, default=24,
    help='Channel multiplier for G (default: %(default)s)')
  parser.add_argument(
    '--D_ch', type=int, default=24,
    help='Channel multiplier for D (default: %(default)s)')
  parser.add_argument(
    '--G_depth', type=int, default=1,
    help='Number of resblocks per stage in G? (default: %(default)s)')
  parser.add_argument(
    '--D_depth', type=int, default=1,
    help='Number of resblocks per stage in D? (default: %(default)s)')
  parser.add_argument(
    '--D_thin', action='store_false', dest='D_wide', default=True,
    help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
  parser.add_argument(
    '--G_shared', action='store_true', default=False,
    help='Use shared embeddings in G? (default: %(default)s)')
  parser.add_argument(
    '--shared_dim', type=int, default=0,
    help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
         '(default: %(default)s)')
  parser.add_argument(
    '--dim_z', type=int, default=128,
    help='Noise dimensionality: %(default)s)')
  parser.add_argument(
    '--z_var', type=float, default=1.0,
    help='Noise variance: %(default)s)')    
  parser.add_argument(
    '--hier', action='store_true', default=False,
    help='Use hierarchical z in G? (default: %(default)s)')
  parser.add_argument(
    '--cross_replica', action='store_true', default=False,
    help='Cross_replica batchnorm in G?(default: %(default)s)')
  parser.add_argument(
    '--mybn', action='store_true', default=False,
    help='Use my batchnorm (which supports standing stats?) %(default)s)')
  parser.add_argument(
    '--G_nl', type=str, default='relu',
    help='Activation function for G (default: %(default)s)')
  parser.add_argument(
    '--D_nl', type=str, default='relu',
    help='Activation function for D (default: %(default)s)')
  parser.add_argument(
    '--G_attn', type=str, default='0',
    help='What resolutions to use attention on for G (underscore separated) '
         '(default: %(default)s)')
  parser.add_argument(
    '--D_attn', type=str, default='0',
    help='What resolutions to use attention on for D (underscore separated) '
         '(default: %(default)s)')
  parser.add_argument(
    '--norm_style', type=str, default='bn',
    help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
         'ln [layernorm], gn [groupnorm] (default: %(default)s)')
  parser.add_argument(
    '--bottom_width', type=int, default=6,
    help='Bottom width for G (default: %(default)s)')
  parser.add_argument(
    '--add_blur', action='store_true', default=True,
    help='Add blur to Generator? (default: %(default)s)')
  parser.add_argument(
    '--add_noise', action='store_true', default=False,
    help='Add noise to Generator? (default: %(default)s)')
  parser.add_argument(
    '--add_style', action='store_true', default=True,
    help='Add style like StyleGAN? (default: %(default)s)')
  parser.add_argument(
    '--skip_z', action='store_true', default=False,
    help='use skip z? (default: %(default)s)')
  parser.add_argument(
    '--gdpp_loss', action='store_true', default=False,
    help='use gdpp_loss? (default: %(default)s)')
  parser.add_argument(
    '--mode_seeking_loss', action='store_true', default=False,
    help='use mode_seeking_loss? (default: %(default)s)')
  parser.add_argument(
    '--style_mlp', type=int, default=6,
    help='Style MLP layers (default: %(default)s)')
  parser.add_argument(
    '--no_conditional', action='store_true', default=False,
    help='Only use style instead of use conditional bn? (default: %(default)s)')
  parser.add_argument(
    '--attn_style', type=str, default='nl',
    help='Attention style one of nl [non local], cbam [cbam]  (default: %(default)s)')
  parser.add_argument(
    '--sched_version', type=str, default='default',
    help='Optim version default[keep the lr as initial], '
         'cal_v0[CosineAnnealingLR], cawr_v0 [CosineAnnealingWarmRestarts] '
         'cal_v1[CosineAnnealingLR], cawr_v1 [CosineAnnealingWarmRestarts] '
         ' (default: %(default)s)')
  parser.add_argument(
    '--z_dist', type=str, default='normal',
    help='z sample from distribution, one of normal [normal distribution], '
         'censored_normal [Censored Normal]  '
         'bernoulli [Bernoulli]  '  
         '(default: %(default)s)')
  parser.add_argument(
    '--arch', type=str, default=None,
    help='if None, use image_size to select arch (default: %(default)s)')
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use; affects both initialization and '
         ' dataloading. (default: %(default)s)')
  parser.add_argument(
    '--G_init', type=str, default='ortho',
    help='Init style to use for G (default: %(default)s)')
  parser.add_argument(
    '--D_init', type=str, default='ortho',
    help='Init style to use for D(default: %(default)s)')
  parser.add_argument(
    '--skip_init', action='store_true', default=False,
    help='Skip initialization, ideal for testing when ortho init was used '
          '(default: %(default)s)')
  parser.add_argument(
    '--G_lr', type=float, default=1e-4,
    help='Learning rate to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_lr', type=float, default=6e-4,
    help='Learning rate to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--G_B1', type=float, default=0.0,
    help='Beta1 to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_B1', type=float, default=0.0,
    help='Beta1 to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--G_B2', type=float, default=0.999,
    help='Beta2 to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_B2', type=float, default=0.999,
    help='Beta2 to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--loss_version', type=str, default='hinge',
    help='loss version(default: %(hinge)s)')
  parser.add_argument(
    '--batch_size', type=int, default=32,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--G_batch_size', type=int, default=0,
    help='Batch size to use for G; if 0, same as D (default: %(default)s)')
  parser.add_argument(
    '--num_G_accumulations', type=int, default=1,
    help='Number of passes to accumulate G''s gradients over '
         '(default: %(default)s)')  
  parser.add_argument(
    '--num_D_steps', type=int, default=1,
    help='Number of D steps per G step (default: %(default)s)')
  parser.add_argument(
    '--num_D_accumulations', type=int, default=1,
    help='Number of passes to accumulate D''s gradients over '
         '(default: %(default)s)')
  parser.add_argument(
    '--split_D', action='store_true', default=False,
    help='Run D twice rather than concatenating inputs? (default: %(default)s)')
  parser.add_argument(
    '--num_epochs', type=int, default=195,
    help='Number of epochs to train for (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=True,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--G_fp16', action='store_true', default=False,
    help='Train with half-precision in G? (default: %(default)s)')
  parser.add_argument(
    '--D_fp16', action='store_true', default=False,
    help='Train with half-precision in D? (default: %(default)s)')
  parser.add_argument(
    '--D_mixed_precision', action='store_true', default=False,
    help='Train with half-precision activations but fp32 params in D? '
         '(default: %(default)s)')
  parser.add_argument(
    '--G_mixed_precision', action='store_true', default=False,
    help='Train with half-precision activations but fp32 params in G? '
         '(default: %(default)s)')
  parser.add_argument(
    '--accumulate_stats', action='store_true', default=False,
    help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
  parser.add_argument(
    '--num_standing_accumulations', type=int, default=16,
    help='Number of forward passes to use in accumulating standing stats? '
         '(default: %(default)s)')        
  parser.add_argument(
    '--G_eval_mode', action='store_true', default=False,
    help='Run G in eval mode (running/standing stats?) at sample/test time? '
         '(default: %(default)s)')
  parser.add_argument(
    '--save_every', type=int, default=10,
    help='Save every X iterations (default: %(default)s)')
  parser.add_argument(
    '--num_save_copies', type=int, default=2,
    help='How many copies to save (default: %(default)s)')
  parser.add_argument(
    '--num_best_copies', type=int, default=5,
    help='How many previous best checkpoints to save (default: %(default)s)')
  parser.add_argument(
    '--which_best', type=str, default='IS',
    help='Which metric to use to determine when to save new "best"'
         'checkpoints, one of IS or FID (default: %(default)s)')
  parser.add_argument(
    '--no_fid', action='store_true', default=False,
    help='Calculate IS only, not FID? (default: %(default)s)')
  parser.add_argument(
    '--test_every', type=int, default=25,
    help='Test every X iterations (default: %(default)s)')
  parser.add_argument(
    '--num_inception_images', type=int, default=50000,
    help='Number of samples to compute inception metrics with '
         '(default: %(default)s)')
  parser.add_argument(
    '--hashname', action='store_true', default=False,
    help='Use a hash of the experiment name instead of the full config '
         '(default: %(default)s)') 
  parser.add_argument(
    '--base_root', type=str, default='../output',
    help='Default location to store all weights, samples, data, and logs '
           ' (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--weights_root', type=str, default='weights',
    help='Default location to store weights (default: %(default)s)')
  parser.add_argument(
    '--logs_root', type=str, default='logs',
    help='Default location to store logs (default: %(default)s)')
  parser.add_argument(
    '--samples_root', type=str, default='samples',
    help='Default location to store samples (default: %(default)s)')  
  parser.add_argument(
    '--pbar', type=str, default='mine',
    help='Type of progressbar to use; one of "mine" or "tqdm" '
         '(default: %(default)s)')
  parser.add_argument(
    '--name_suffix', type=str, default='',
    help='Suffix for experiment name for loading weights for sampling '
         '(consider "best0") (default: %(default)s)')
  parser.add_argument(
    '--experiment_name', type=str, default='i96_ch24_hinge_ema_dstep1_bs32_noatt_glr0001_glr0006_aug_init_ortho_blur_style_origin_crop_mode8_on_kaggle',
    help='Optionally override the automatic experiment naming with this arg. '
         '(default: %(default)s)')
  parser.add_argument(
    '--config_from_name', action='store_true', default=False,
    help='Use a hash of the experiment name instead of the full config '
         '(default: %(default)s)')
  parser.add_argument(
    '--ema', action='store_true', default=True,
    help='Keep an ema of G''s weights? (default: %(default)s)')
  parser.add_argument(
    '--ema_decay', type=float, default=0.9999,
    help='EMA decay rate (default: %(default)s)')
  parser.add_argument(
    '--use_ema', action='store_true', default=True,
    help='Use the EMA parameters of G for evaluation? (default: %(default)s)')
  parser.add_argument(
    '--ema_start', type=int, default=2000,
    help='When to start updating the EMA weights (default: %(default)s)')
  parser.add_argument(
    '--adam_eps', type=float, default=1e-8,
    help='epsilon value to use for Adam (default: %(default)s)')
  parser.add_argument(
    '--BN_eps', type=float, default=1e-5,
    help='epsilon value to use for BatchNorm (default: %(default)s)')
  parser.add_argument(
    '--SN_eps', type=float, default=1e-8,
    help='epsilon value to use for Spectral Norm(default: %(default)s)')
  parser.add_argument(
    '--num_G_SVs', type=int, default=1,
    help='Number of SVs to track in G (default: %(default)s)')
  parser.add_argument(
    '--num_D_SVs', type=int, default=1,
    help='Number of SVs to track in D (default: %(default)s)')
  parser.add_argument(
    '--num_G_SV_itrs', type=int, default=1,
    help='Number of SV itrs in G (default: %(default)s)')
  parser.add_argument(
    '--num_D_SV_itrs', type=int, default=1,
    help='Number of SV itrs in D (default: %(default)s)')
  parser.add_argument(
    '--G_ortho', type=float, default=0.0, # 1e-4 is default for BigGAN
    help='Modified ortho reg coefficient in G(default: %(default)s)')
  parser.add_argument(
    '--D_ortho', type=float, default=0.0,
    help='Modified ortho reg coefficient in D (default: %(default)s)')
  parser.add_argument(
    '--toggle_grads', action='store_true', default=True,
    help='Toggle D and G''s "requires_grad" settings when not training them? '
         ' (default: %(default)s)')
  parser.add_argument(
    '--which_train_fn', type=str, default='GAN',
    help='How2trainyourbois (default: %(default)s)')  
  parser.add_argument(
    '--load_weights', type=str, default='',
    help='Suffix for which weights to load (e.g. best0, copy0) '
         '(default: %(default)s)')
  parser.add_argument(
    '--resume', action='store_true', default=False,
    help='Resume training? (default: %(default)s)')
  parser.add_argument(
    '--logstyle', type=str, default='%3.3e',
    help='What style to use when logging training metrics?'
         'One of: %#.#f/ %#.#e (float/exp, text),'
         'pickle (python pickle),'
         'npz (numpy zip),'
         'mat (MATLAB .mat file) (default: %(default)s)')
  parser.add_argument(
    '--log_G_spectra', action='store_true', default=False,
    help='Log the top 3 singular values in each SN layer in G? '
         '(default: %(default)s)')
  parser.add_argument(
    '--log_D_spectra', action='store_true', default=False,
    help='Log the top 3 singular values in each SN layer in D? '
         '(default: %(default)s)')
  parser.add_argument(
    '--sv_log_interval', type=int, default=10,
    help='Iteration interval for logging singular values '
         ' (default: %(default)s)')
  parser.add_argument(
    '--truncated_threshold', type=float, default=1)
  parser.add_argument(
    '--on_kaggle', action='store_true', default=True)
  parser.add_argument(
    '--debug', action='store_true', default=False)
  parser.add_argument(
    '--crop_mode', type=int, default=8, help='0:None '
                                            '1:max_crop '
                                            '2:max crop + origin '
                                            '3:max crop + origin + CenterCropLongEdge'
                                            '4:max crop + origin(-multi dogs -small dogs) + CenterCropLongEdge')
  parser.add_argument(
    '--resize_mode', type=int, default=2,
      help='NEAREST = NONE = 0 '
           'BOX = 4 '
           'BILINEAR = LINEAR = 2 '
           'HAMMING = 5 '
           'BICUBIC = CUBIC = 3 '
           'LANCZOS = ANTIALIAS = 1')
  parser.add_argument(
    '--clip_norm', type=float, default=None)
  parser.add_argument(
    '--amsgrad', action='store_true', default=False)
  return parser
def utils_add_sample_parser(parser):
  parser.add_argument(
    '--sample_npz', action='store_true', default=False,
    help='Sample "sample_num_npz" images and save to npz? '
         '(default: %(default)s)')
  parser.add_argument(
    '--sample_num_npz', type=int, default=50000,
    help='Number of images to sample when sampling NPZs '
         '(default: %(default)s)')
  parser.add_argument(
    '--sample_sheets', action='store_true', default=False,
    help='Produce class-conditional sample sheets and stick them in '
         'the samples root? (default: %(default)s)')
  parser.add_argument(
    '--sample_interps', action='store_true', default=False,
    help='Produce interpolation sheets and stick them in '
         'the samples root? (default: %(default)s)')         
  parser.add_argument(
    '--sample_sheet_folder_num', type=int, default=-1,
    help='Number to use for the folder for these sample sheets '
         '(default: %(default)s)')
  parser.add_argument(
    '--sample_random', action='store_true', default=False,
    help='Produce a single random sheet? (default: %(default)s)')
  parser.add_argument(
    '--store_y', action='store_true', default=False)
  parser.add_argument(
    '--sample_trunc_curves', type=str, default='',
    help='Get inception metrics with a range of variances?'
         'To use this, specify a startpoint, step, and endpoint, e.g. '
         '--sample_trunc_curves 0.2_0.1_1.0 for a startpoint of 0.2, '
         'endpoint of 1.0, and stepsize of 1.0.  Note that this is '
         'not exactly identical to using tf.truncated_normal, but should '
         'have approximately the same effect. (default: %(default)s)')
  parser.add_argument(
    '--sample_inception_metrics', action='store_true', default=False,
    help='Calculate Inception metrics with sample.py? (default: %(default)s)')  
  return parser
utils_dset_dict = {'I32': dset_ImageFolder, 'I64': dset_ImageFolder,
             'I128': dset_ImageFolder, 'I256': dset_ImageFolder,
             'I32_hdf5': dset_ILSVRC_HDF5, 'I64_hdf5': dset_ILSVRC_HDF5,
             'I128_hdf5': dset_ILSVRC_HDF5, 'I256_hdf5': dset_ILSVRC_HDF5,
             'C10': dset_CIFAR10, 'C100': dset_CIFAR100,
             'Dog128': dset_DogDataset,'Dog96': dset_DogDataset,'Dog64': dset_DogDataset,'Dog32': dset_DogDataset,
             'DogOrigin64': dset_ImageFolder,'DogOrigin': dset_ImageFolder,
             'DogOrigin96': dset_ImageFolder,
             'DogOrigin128': dset_ImageFolder,
             'DogOriginSquare96': dset_DogDataset,
             'DogOriginSquare80': dset_DogDataset,
             'DogOriginMixCR96': dset_DogDataset,
             'DogOriginMixCR80': dset_DogDataset,
             'DogOrigin64_C1200': dset_ImageFolder,
             'DogOrigin64_O1200': dset_ImageFolder,
             'DogOrigin64_C480': dset_ImageFolder,
             'DogOrigin64_O480': dset_ImageFolder,
             'DogOrigin96_C60': dset_ImageFolder,
             'DogOrigin96_C30': dset_ImageFolder,
             }
utils_imsize_dict = {'I32': 32, 'I32_hdf5': 32,
               'I64': 64,'I64_hdf5': 64,
               'I128': 128, 'I128_hdf5': 128,
               'I256': 256, 'I256_hdf5': 256,
               'C10': 32, 'C100': 32,
               'Dog128': 128,'Dog96': 96,'Dog64': 64,'Dog32': 32,
               'DogOrigin64':64,'DogOrigin':64,'DogOrigin96': 96,'DogOrigin128': 128,
               'DogOriginSquare96':96,
               'DogOriginSquare80':80,
               'DogOriginMixCR96':96,
               'DogOriginMixCR80':80,
               'DogOrigin64_C1200':64,
               'DogOrigin64_O1200':64,
               'DogOrigin64_C480':64,
               'DogOrigin64_O480':64,
               'DogOrigin96_C60':96,
               'DogOrigin96_C30':96,
               }
utils_root_dict = {'I32': 'ImageNet', 'I32_hdf5': 'ILSVRC32.hdf5',
             'I64': 'ImageNet', 'I64_hdf5': 'ILSVRC64.hdf5',
             'I128': 'ImageNet', 'I128_hdf5': 'ILSVRC128.hdf5',
             'I256': 'ImageNet', 'I256_hdf5': 'ILSVRC256.hdf5',
             'C10': 'cifar', 'C100': 'cifar',
             'Dog128': 'cache/cropped_dogs_128.pkl',
             'Dog96': 'cache/cropped_dogs_96.pkl',
             'Dog64': 'cache/cropped_dogs_64.pkl',
             'Dog32': 'cache/cropped_dogs_32.pkl',
             'DogOrigin64': 'generative-dog-images/origin_dogs',
             'DogOrigin96': 'generative-dog-images/origin_dogs',
             'DogOrigin128': 'generative-dog-images/origin_dogs',
             'DogOriginSquare96': 'cache/origin_square_96.pkl',
             'DogOriginSquare80': 'cache/origin_square_80.pkl',
             'DogOriginMixCR96': 'cache/origin_96.pkl',
             'DogOriginMixCR80': 'cache/origin_80.pkl',
             'DogOrigin': 'generative-dog-images/origin_dogs',
             'DogOrigin64_C1200':'/data4/data/gan_dogs/result/samples/simplenet_crop/kmeans_1200',
             'DogOrigin64_O1200':'/data4/data/gan_dogs/result/samples/simplenet_origin/kmeans_1200',
             'DogOrigin64_C480':'/data4/data/gan_dogs/result/samples/simplenet_crop/kmeans_480',
             'DogOrigin64_O480':'/data4/data/gan_dogs/result/samples/simplenet_origin/kmeans_480',
             'DogOrigin96_C60':'/data4/data/gan_dogs/result/samples/simplenet_crop/kmeans_60',
             'DogOrigin96_C30':'/data4/data/gan_dogs/result/samples/simplenet_crop/kmeans_30',
             }
utils_nclass_dict = {'I32': 1000, 'I32_hdf5': 1000,
               'I64': 1000, 'I64_hdf5': 1000,
               'I128': 1000, 'I128_hdf5': 1000,
               'I256': 1000, 'I256_hdf5': 1000,
               'C10': 10, 'C100': 100,
               'Dog128':120,'Dog96':120,'Dog64':120,'Dog32': 120,
               'DogOrigin64':120,'DogOrigin':120,'DogOrigin96':120,'DogOrigin128':120,
               'DogOriginSquare96':120,
               'DogOriginSquare80':120,
               'DogOriginMixCR96':120,
               'DogOriginMixCR80':120,
               'DogOrigin64_C1200':1200,
               'DogOrigin64_O1200':1200,
               'DogOrigin64_C480':480,
               'DogOrigin64_O480':480,
               'DogOrigin96_C60':60,
               'DogOrigin96_C30':30,
               }
utils_classes_per_sheet_dict = {'I32': 50, 'I32_hdf5': 50,
                          'I64': 50, 'I64_hdf5': 50,
                          'I128': 20, 'I128_hdf5': 20,
                          'I256': 20, 'I256_hdf5': 20,
                          'C10': 10, 'C100': 100,
                          'Dog128':120,'Dog96':120,'Dog64':120,'Dog32':120,
                          'DogOrigin64':120,'DogOrigin':120,'DogOrigin96':120,'DogOrigin128':120,
                          'DogOriginSquare96':120,
                          'DogOriginSquare80':120,
                          'DogOriginMixCR96':120,
                          'DogOriginMixCR80':120,
                          'DogOrigin64_C1200':1200,
                          'DogOrigin64_O1200':1200,
                          'DogOrigin64_C480':480,
                          'DogOrigin64_O480':480,
                          'DogOrigin96_C60':60,
                          'DogOrigin96_C30':30,
                          }
utils_activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'leaky_relu': nn.LeakyReLU(inplace=False),
                   'leaky_relu_02': nn.LeakyReLU(0.2, inplace=False),
                   'ir': nn.ReLU(inplace=True),}
class utils_CenterCropLongEdge(object):
  def __call__(self, img):
    return transforms.functional.center_crop(img, min(img.size))
  def __repr__(self):
    return self.__class__.__name__
class utils_RandomCropLongEdge(object):
  def __call__(self, img):
    size = (min(img.size), min(img.size))
    i = (0 if size[0] == img.size[0] 
          else np.random.randint(low=0,high=img.size[0] - size[0]))
    j = (0 if size[1] == img.size[1]
          else np.random.randint(low=0,high=img.size[1] - size[1]))
    return transforms.functional.crop(img, j, i, size[0], size[1])
  def __repr__(self):
    return self.__class__.__name__
class utils_MultiEpochSampler(torch.utils.data.Sampler):
  def __init__(self, data_source, num_epochs, start_itr=0, batch_size=128):
    self.data_source = data_source
    self.num_samples = len(self.data_source)
    self.num_epochs = num_epochs
    self.start_itr = start_itr
    self.batch_size = batch_size
    if not isinstance(self.num_samples, int) or self.num_samples <= 0:
      raise ValueError("num_samples should be a positive integeral "
                       "value, but got num_samples={}".format(self.num_samples))
  def __iter__(self):
    n = len(self.data_source)
    num_epochs = int(np.ceil((n * self.num_epochs 
                              - (self.start_itr * self.batch_size)) / float(n)))
    out = [torch.randperm(n) for epoch in range(self.num_epochs)][-num_epochs:]
    out[0] = out[0][(self.start_itr * self.batch_size % n):]
    output = torch.cat(out).tolist()
    print('Length dataset output is %d' % len(output))
    return iter(output)
  def __len__(self):
    return len(self.data_source) * self.num_epochs - self.start_itr * self.batch_size
def utils_get_data_loaders(dataset, data_root=None, augment=False, batch_size=64, 
                     num_workers=8, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False,
                     **kwargs):
  if utils_root_dict[dataset].startswith('/'):
    data_root = utils_root_dict[dataset]
  else:
    data_root += '/%s' % utils_root_dict[dataset]
  print('Using dataset root location %s' % data_root)
  which_dataset = utils_dset_dict[dataset]
  norm_mean = [0.5,0.5,0.5]
  norm_std = [0.5,0.5,0.5]
  image_size = utils_imsize_dict[dataset]
  dataset_kwargs = {'index_filename': '%s_imgs.npz' % dataset}
  if dataset.startswith('Dog'):
    dataset_kwargs['image_size']= image_size
    dataset_kwargs['crop_mode'] = kwargs.get('crop_mode', 0)
    dataset_kwargs['resize_mode'] = kwargs.get('resize_mode', 2)
    dataset_kwargs['debug'] = kwargs.get('debug', False)
  if 'hdf5' in dataset:
    train_transform = None
  else:
    if augment == 1:
      print('Data will be augmented...%d' % augment)
      if dataset in ['C10', 'C100']:
        train_transform = [transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip()]
      elif dataset in ['Dog128','Dog96','Dog64','Dog32', 'DogOrigin64', 'DogOrigin']:
        train_transform =[utils_RandomCropLongEdge(),
                         transforms.Resize(image_size, kwargs['resize_mode']),
                         transforms.RandomHorizontalFlip()]
      elif dataset in ['DogOriginSquare80','DogOriginSquare96']:
        train_transform =[transforms.RandomHorizontalFlip()]
      elif dataset in ['DogOriginMixCR80','DogOriginMixCR96']:
        resize_trans1=transforms.Compose([utils_RandomCropLongEdge(), transforms.Resize(image_size, kwargs['resize_mode'])])
        resize_trans2=transforms.Resize([image_size,image_size], kwargs['resize_mode'])
        resize_trans=transforms.RandomChoice([resize_trans1,resize_trans2])
        train_transform =[resize_trans,transforms.RandomHorizontalFlip()]
      else:
        train_transform = [utils_RandomCropLongEdge(),
                         transforms.Resize(image_size, kwargs['resize_mode']),
                         transforms.RandomHorizontalFlip()]
    elif augment == 2:
      print('Data will be augmented...%d' % augment)
      train_transform = [
                     transforms.RandomRotation(5),
                     transforms.RandomHorizontalFlip()]
    elif augment == 3:
      print('Data will be augmented...%d' % augment)
      train_transform = [
                     transforms.RandomRotation(8),
                     transforms.RandomHorizontalFlip()]
    elif augment == 4:
      print('Data will be augmented...%d' % augment)
      train_transform = [
                     transforms.RandomRotation(10),
                     transforms.RandomHorizontalFlip()]
    elif augment == 0:
      print('Data will not be augmented...')
      if dataset in ['C10', 'C100']:
        train_transform = []
      elif dataset in ['Dog128','Dog96','Dog64','Dog32','DogOrigin64', 'DogOrigin']:
        train_transform = [transforms.Resize(image_size, kwargs['resize_mode']),
                           transforms.CenterCrop(image_size)]
      elif dataset in ['DogOriginSquare80','DogOriginSquare96']:
        train_transform =[]
      elif dataset in ['DogOriginMixCR80','DogOriginMixCR96']:
        train_transform = [transforms.Resize([image_size, image_size], kwargs['resize_mode'])]
      else:
        train_transform = [utils_CenterCropLongEdge(), transforms.Resize(image_size, kwargs['resize_mode'])]
    else:
        raise ValueError("augment must be 0~4")
    train_transform = transforms.Compose(train_transform + [
                     transforms.ToTensor(),
                     transforms.Normalize(norm_mean, norm_std)])
  train_set = which_dataset(root=data_root, transform=train_transform,
                            load_in_mem=load_in_mem, **dataset_kwargs)
  loaders = []   
  if use_multiepoch_sampler:
    print('Using multiepoch sampler from start_itr %d...' % start_itr)
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    sampler = utils_MultiEpochSampler(train_set, num_epochs, start_itr, batch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=sampler, **loader_kwargs)
  else:
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                     'drop_last': drop_last} # Default, drop last incomplete batch
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=shuffle, **loader_kwargs)
  loaders.append(train_loader)
  return loaders
def utils_seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
def utils_update_config_roots(config):
  if config['base_root']:
    print('Pegging all root folders to base root %s' % config['base_root'])
    for key in ['data', 'weights', 'logs', 'samples']:
      config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
  return config
def utils_prepare_root(config):
  for key in ['weights_root', 'logs_root', 'samples_root']:
    if not os.path.exists(config[key]):
      print('Making directory %s for %s...' % (config[key], key))
      os.mkdir(config[key])
class utils_ema(object):
  def __init__(self, source, target, decay=0.9999, start_itr=0):
    self.source = source
    self.target = target
    self.decay = decay
    self.start_itr = start_itr
    self.source_dict = self.source.state_dict()
    self.target_dict = self.target.state_dict()
    print('Initializing EMA parameters to be source parameters...')
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.source_dict[key].data)
  def update(self, itr=None):
    if itr and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.target_dict[key].data * decay 
                                     + self.source_dict[key].data * (1 - decay))
def utils_ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      if len(param.shape) < 2 or any([param is item for item in blacklist]):
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t()) 
              * (1. - torch.eye(w.shape[0], device=w.device)), w))
      param.grad.data += strength * grad.view(param.shape)
def utils_default_ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      if len(param.shape) < 2 or param in blacklist:
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t()) 
               - torch.eye(w.shape[0], device=w.device), w))
      param.grad.data += strength * grad.view(param.shape)
def utils_toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off
def utils_join_strings(base_string, strings):
  return base_string.join([item for item in strings if item])
def utils_save_weights(G, D, state_dict, weights_root, experiment_name, 
                 name_suffix=None, G_ema=None):
  root = '/'.join([weights_root, experiment_name])
  if not os.path.exists(root):
    os.mkdir(root)
  if name_suffix:
    print('Saving weights to %s/%s...' % (root, name_suffix))
  else:
    print('Saving weights to %s...' % root)
  torch.save(G.state_dict(), 
              '%s/%s.pth' % (root, utils_join_strings('_', ['G', name_suffix])))
  torch.save(G.optim.state_dict(), 
              '%s/%s.pth' % (root, utils_join_strings('_', ['G_optim', name_suffix])))
  torch.save(D.state_dict(), 
              '%s/%s.pth' % (root, utils_join_strings('_', ['D', name_suffix])))
  torch.save(D.optim.state_dict(),
              '%s/%s.pth' % (root, utils_join_strings('_', ['D_optim', name_suffix])))
  torch.save(state_dict,
              '%s/%s.pth' % (root, utils_join_strings('_', ['state_dict', name_suffix])))
  if G_ema is not None:
    torch.save(G_ema.state_dict(), 
                '%s/%s.pth' % (root, utils_join_strings('_', ['G_ema', name_suffix])))
def utils_load_weights(G, D, state_dict, weights_root, experiment_name, 
                 name_suffix=None, G_ema=None, strict=True, load_optim=True):
  root = '/'.join([weights_root, experiment_name])
  if name_suffix:
    print('Loading %s weights from %s...' % (name_suffix, root))
  else:
    print('Loading weights from %s...' % root)
  if G is not None:
    G.load_state_dict(
      torch.load('%s/%s.pth' % (root, utils_join_strings('_', ['G', name_suffix]))),
      strict=strict)
    if load_optim:
      G.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, utils_join_strings('_', ['G_optim', name_suffix]))))
  if D is not None:
    D.load_state_dict(
      torch.load('%s/%s.pth' % (root, utils_join_strings('_', ['D', name_suffix]))),
      strict=strict)
    if load_optim:
      D.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, utils_join_strings('_', ['D_optim', name_suffix]))))
  for item in state_dict:
    state_dict[item] = torch.load('%s/%s.pth' % (root, utils_join_strings('_', ['state_dict', name_suffix])))[item]
  if G_ema is not None:
    G_ema.load_state_dict(
      torch.load('%s/%s.pth' % (root, utils_join_strings('_', ['G_ema', name_suffix]))),
      strict=strict)
class utils_MetricsLogger(object):
  def __init__(self, fname, reinitialize=False):
    self.fname = fname
    self.reinitialize = reinitialize
    if os.path.exists(self.fname):
      if self.reinitialize:
        print('{} exists, deleting...'.format(self.fname))
        os.remove(self.fname)
  def log(self, record=None, **kwargs):
    if record is None:
      record = {}
    record.update(kwargs)
    record['_stamp'] = time.time()
    with open(self.fname, 'a') as f:
      f.write(json.dumps(record, ensure_ascii=True) + '\n')
class utils_MyLogger(object):
  def __init__(self, fname, reinitialize=False, logstyle='%3.3f'):
    self.root = fname
    if not os.path.exists(self.root):
      os.mkdir(self.root)
    self.reinitialize = reinitialize
    self.metrics = []
    self.logstyle = logstyle # One of '%3.3f' or like '%3.3e'
  def reinit(self, item):
    if os.path.exists('%s/%s.log' % (self.root, item)):
      if self.reinitialize:
        if 'sv' in item :
          if not any('sv' in item for item in self.metrics):
            print('Deleting singular value logs...')
        else:
          print('{} exists, deleting...'.format('%s_%s.log' % (self.root, item)))
        os.remove('%s/%s.log' % (self.root, item))
  def log(self, itr, **kwargs):
    for arg in kwargs:
      if arg not in self.metrics:
        if self.reinitialize:
          self.reinit(arg)
        self.metrics += [arg]
      if self.logstyle == 'pickle':
        print('Pickle not currently supported...')
      elif self.logstyle == 'mat':
        print('.mat logstyle not currently supported...')
      else:
        with open('%s/%s.log' % (self.root, arg), 'a') as f:
          f.write('%d: %s\n' % (itr, self.logstyle % kwargs[arg]))
def utils_write_metadata(logs_root, experiment_name, config, state_dict):
  with open(('%s/%s/metalog.txt' % 
             (logs_root, experiment_name)), 'w') as writefile:
    writefile.write('datetime: %s\n' % str(datetime.datetime.now()))
    writefile.write('config: %s\n' % str(config))
    writefile.write('state: %s\n' %str(state_dict))
def utils_progress(items, desc='', total=None, min_delay=0.1, displaytype='s1k'):
  total = total or len(items)
  t_start = time.time()
  t_last = 0
  for n, item in enumerate(items):
    t_now = time.time()
    if t_now - t_last > min_delay:
      print("\r%s%d/%d (%6.2f%%)" % (
              desc, n+1, total, n / float(total) * 100), end=" ")
      if n > 0:
        if displaytype == 's1k': # minutes/seconds for 1000 iters
          next_1000 = n + (1000 - n%1000)
          t_done = t_now - t_start
          t_1k = t_done / n * next_1000
          outlist = list(divmod(t_done, 60)) + list(divmod(t_1k - t_done, 60))
          print("(TE/ET1k: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
        else:# displaytype == 'eta':
          t_done = t_now - t_start
          t_total = t_done / n * total
          outlist = list(divmod(t_done, 60)) + list(divmod(t_total - t_done, 60))
          print("(TE/ETA: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
      sys.stdout.flush()
      t_last = t_now
    yield item
  t_total = time.time() - t_start
  print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))
def utils_sample(G, z_, y_, config):
  with torch.no_grad():
    z_.sample_()
    y_.sample_()
    if config['parallel']:
      G_z =  nn.parallel.data_parallel(G, (z_, G.shared(y_)))
    else:
      G_z = G(z_, G.shared(y_))
    return G_z, y_
def utils_sample_sheet(G, classes_per_sheet, num_classes, samples_per_class, parallel,
                 samples_root, experiment_name, folder_number, z_=None):
  if not os.path.isdir('%s/%s' % (samples_root, experiment_name)):
    os.mkdir('%s/%s' % (samples_root, experiment_name))
  if not os.path.isdir('%s/%s/%d' % (samples_root, experiment_name, folder_number)):
    os.mkdir('%s/%s/%d' % (samples_root, experiment_name, folder_number))
  for i in range(num_classes // classes_per_sheet):
    ims = []
    y = torch.arange(i * classes_per_sheet, (i + 1) * classes_per_sheet, device='cuda')
    for j in range(samples_per_class):
      if (z_ is not None) and hasattr(z_, 'sample_') and classes_per_sheet <= z_.size(0):
        z_.sample_()
      else:
        z_ = torch.randn(classes_per_sheet, G.dim_z, device='cuda')        
      with torch.no_grad():
        if parallel:
          o = nn.parallel.data_parallel(G, (z_[:classes_per_sheet], G.shared(y)))
        else:
          o = G(z_[:classes_per_sheet], G.shared(y))
      ims += [o.data.cpu()]
    out_ims = torch.stack(ims, 1).view(-1, ims[0].shape[1], ims[0].shape[2], 
                                       ims[0].shape[3]).data.float().cpu()
    image_filename = '%s/%s/%d/samples%d.jpg' % (samples_root, experiment_name, 
                                                 folder_number, i)
    torchvision.utils.save_image(out_ims, image_filename,
                                 nrow=samples_per_class, normalize=True)
def utils_interp(x0, x1, num_midpoints):
  lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cuda').to(x0.dtype)
  return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))
def utils_interp_sheet(G, num_per_sheet, num_midpoints, num_classes, parallel,
                 samples_root, experiment_name, folder_number, sheet_number=0,
                 fix_z=False, fix_y=False, device='cuda'):
  if fix_z: # If fix Z, only sample 1 z per row
    zs = torch.randn(num_per_sheet, 1, G.dim_z, device=device)
    zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, G.dim_z)
  else:
    zs = utils_interp(torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                num_midpoints).view(-1, G.dim_z)
  if fix_y: # If fix y, only sample 1 z per row
    ys = utils_sample_1hot(num_per_sheet, num_classes)
    ys = G.shared(ys).view(num_per_sheet, 1, -1)
    ys = ys.repeat(1, num_midpoints + 2, 1).view(num_per_sheet * (num_midpoints + 2), -1)
  else:
    ys = utils_interp(G.shared(utils_sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                G.shared(utils_sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                num_midpoints).view(num_per_sheet * (num_midpoints + 2), -1)
  if G.fp16:
    zs = zs.half()
  with torch.no_grad():
    if parallel:
      out_ims = nn.parallel.data_parallel(G, (zs, ys)).data.cpu()
    else:
      out_ims = G(zs, ys).data.cpu()
  interp_style = '' + ('Z' if not fix_z else '') + ('Y' if not fix_y else '')
  image_filename = '%s/%s/%d/interp%s%d.jpg' % (samples_root, experiment_name,
                                                folder_number, interp_style,
                                                sheet_number)
  torchvision.utils.save_image(out_ims, image_filename,
                               nrow=num_midpoints + 2, normalize=True)
def utils_print_grad_norms(net):
    gradsums = [[float(torch.norm(param.grad).item()),
                 float(torch.norm(param).item()), param.shape]
                for param in net.parameters()]
    order = np.argsort([item[0] for item in gradsums])
    print(['%3.3e,%3.3e, %s' % (gradsums[item_index][0],
                                gradsums[item_index][1],
                                str(gradsums[item_index][2])) 
                              for item_index in order])
def utils_get_SVs(net, prefix):
  d = net.state_dict()
  return {('%s_%s' % (prefix, key)).replace('.', '_') :
            float(d[key].item())
            for key in d if 'sv' in key}
def utils_name_from_config(config):
  name = '_'.join([
  item for item in [
  'Big%s' % config['which_train_fn'],
  config['dataset'],
  config['model'] if config['model'] != 'BigGAN' else None,
  'seed%d' % config['seed'],
  'Gch%d' % config['G_ch'],
  'Dch%d' % config['D_ch'],
  '%s' % config['loss_version'],
  'Gd%d' % config['G_depth'] if config['G_depth'] > 1 else None,
  'Dd%d' % config['D_depth'] if config['D_depth'] > 1 else None,
  'bs%d' % config['batch_size'],
  'Gfp16' if config['G_fp16'] else None,
  'Dfp16' if config['D_fp16'] else None,
  'nDs%d' % config['num_D_steps'] if config['num_D_steps'] > 1 else None,
  'nDa%d' % config['num_D_accumulations'] if config['num_D_accumulations'] > 1 else None,
  'nGa%d' % config['num_G_accumulations'] if config['num_G_accumulations'] > 1 else None,
  'Glr%2.1e' % config['G_lr'],
  'Dlr%2.1e' % config['D_lr'],
  'GB%3.3f' % config['G_B1'] if config['G_B1'] !=0.0 else None,
  'GBB%3.3f' % config['G_B2'] if config['G_B2'] !=0.999 else None,
  'DB%3.3f' % config['D_B1'] if config['D_B1'] !=0.0 else None,
  'DBB%3.3f' % config['D_B2'] if config['D_B2'] !=0.999 else None,
  'Gnl%s' % config['G_nl'],
  'Dnl%s' % config['D_nl'],
  'Ginit%s' % config['G_init'],
  'Dinit%s' % config['D_init'],
  'G%s' % config['G_param'] if config['G_param'] != 'SN' else None,
  'D%s' % config['D_param'] if config['D_param'] != 'SN' else None,
  'Gattn%s' % config['G_attn'] if config['G_attn'] != '0' else None,
  'Dattn%s' % config['D_attn'] if config['D_attn'] != '0' else None,
  'Gortho%2.1e' % config['G_ortho'] if config['G_ortho'] > 0.0 else None,
  'Dortho%2.1e' % config['D_ortho'] if config['D_ortho'] > 0.0 else None,
  config['norm_style'] if config['norm_style'] != 'bn' else None,
  'cr' if config['cross_replica'] else None,
  'Gshared' if config['G_shared'] else None,
  'hier' if config['hier'] else None,
  'ema' if config['ema'] else None,
  config['name_suffix'] if config['name_suffix'] else None,
  ]
  if item is not None])
  if config['hashname']:
    return utils_hashname(name)
  else:
    return name
def utils_hashname(name):
  h = hash(name)
  a = h % len(animal_hash_a)
  h = h // len(animal_hash_a)
  b = h % len(animal_hash_b)
  h = h // len(animal_hash_c)
  c = h % len(animal_hash_c)
  return animal_hash_a[a] + animal_hash_b[b] + animal_hash_c[c]
def utils_query_gpu(indices):
  os.system('nvidia-smi -i 0 --query-gpu=memory.free --format=csv')
def utils_count_parameters(module):
  print('Number of parameters: {}'.format(
    sum([p.data.nelement() for p in module.parameters()])))
def utils_sample_1hot(batch_size, num_classes, device='cuda'):
  return torch.randint(low=0, high=num_classes, size=(batch_size,),
          device=device, dtype=torch.int64, requires_grad=False)
from scipy.stats import truncnorm
def utils_truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    values = torch.from_numpy(values)
    return values
class utils_Distribution(torch.Tensor):
  def init_distribution(self, dist_type, **kwargs):    
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']
    elif self.dist_type=='censored_normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type=='bernoulli':
      pass
    elif self.dist_type=='truncated_normal':
      self.threshold = kwargs['threshold']
  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)
    elif self.dist_type=='censored_normal':
      self.normal_(self.mean, self.var)
      self.relu_()
    elif self.dist_type=='bernoulli':
      self.bernoulli_()
    elif self.dist_type=='truncated_normal':
      v = utils_truncated_normal(self.shape, self.threshold)
      self.set_(v.float().cuda())
  def to(self, *args, **kwargs):
    new_obj = utils_Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)    
    return new_obj
def utils_prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda',
                fp16=False,z_var=1.0,z_dist='normal', threshold=1):
  z_ = utils_Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
  if z_dist=='normal':
    z_.init_distribution(z_dist, mean=0, var=z_var)
  elif z_dist=='censored_normal':
    z_.init_distribution(z_dist, mean=0, var=z_var)
  elif z_dist=='bernoulli':
    z_.init_distribution(z_dist)
  elif z_dist=='truncated_normal':
    z_.init_distribution(z_dist, threshold=threshold)
  z_ = z_.to(device,torch.float16 if fp16 else torch.float32)   
  if fp16:
    z_ = z_.half()
  y_ = utils_Distribution(torch.zeros(G_batch_size, requires_grad=False))
  y_.init_distribution('categorical',num_categories=nclasses)
  y_ = y_.to(device, torch.int64)
  return z_, y_
def utils_initiate_standing_stats(net):
  for module in net.modules():
    if hasattr(module, 'accumulate_standing'):
      module.reset_stats()
      module.accumulate_standing = True
def utils_accumulate_standing_stats(net, z, y, nclasses, num_accumulations=16):
  utils_initiate_standing_stats(net)
  net.train()
  for i in range(num_accumulations):
    with torch.no_grad():
      z.normal_()
      y.random_(0, nclasses)
      x = net(z, net.shared(y)) # No need to parallelize here unless using syncbn
  net.eval() 
from torch.optim.optimizer import Optimizer
class utils_Adam16(Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0):
    defaults = dict(lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay)
    params = list(params)
    super(utils_Adam16, self).__init__(params, defaults)
  def load_state_dict(self, state_dict):
    super(utils_Adam16, self).load_state_dict(state_dict)
    for group in self.param_groups:
      for p in group['params']:
        self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
        self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
        self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data.float()
        state = self.state[p]
        if len(state) == 0:
          state['step'] = 0
          state['exp_avg'] = grad.new().resize_as_(grad).zero_()
          state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
          state['fp32_p'] = p.data.float()
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        state['step'] += 1
        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], state['fp32_p'])
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        denom = exp_avg_sq.sqrt().add_(group['eps'])
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
        p.data = state['fp32_p'].half()
    return loss
######################### utils.py end #########################
######################## losses.py start ########################
def losses_GDPPLoss(phiFake, phiReal, backward=True):
    def compute_diversity(phi):
        phi = F.normalize(phi, p=2, dim=1)
        SB = torch.mm(phi, phi.t())
        eigVals, eigVecs = torch.symeig(SB, eigenvectors=True)
        return eigVals, eigVecs
    def normalize_min_max(eigVals):
        minV, maxV = torch.min(eigVals), torch.max(eigVals)
        return (eigVals - minV) / (maxV - minV)
    phiFake=phiFake.view(phiFake.size(0),-1)
    phiReal=phiReal.view(phiReal.size(0),-1)
    fakeEigVals, fakeEigVecs = compute_diversity(phiFake)
    realEigVals, realEigVecs = compute_diversity(phiReal)
    magnitudeLoss = 0.0001 * F.mse_loss(target=realEigVals, input=fakeEigVals)
    structureLoss = -torch.sum(torch.mul(fakeEigVecs, realEigVecs), 0)
    normalizedRealEigVals = normalize_min_max(realEigVals)
    weightedStructureLoss = torch.sum(
        torch.mul(normalizedRealEigVals, structureLoss))
    gdppLoss = magnitudeLoss + weightedStructureLoss
    if backward:
        gdppLoss.backward(retain_graph=True)
    return gdppLoss
def losses_loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2
def losses_loss_dcgan_gen(dis_fake, dis_real):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss
def losses_loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
def losses_loss_hinge_dis2(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(0.5 - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
def losses_loss_hinge_gen(dis_fake, dis_real):
  loss = -torch.mean(dis_fake)
  return loss
def losses_loss_rals_dis(dis_fake, dis_real):
  real_label = 0.5
  batch_size=dis_fake.size(0)
  labels = torch.full((batch_size, 1), real_label).cuda()
  loss_real = torch.mean((dis_real - torch.mean(dis_fake) - labels) ** 2)
  loss_fake=torch.mean((dis_fake - torch.mean(dis_real) + labels) ** 2)
  return loss_real, loss_fake
def losses_loss_rals_gen(dis_fake, dis_real):
  real_label = 0.5
  batch_size = dis_fake.size(0)
  labels = torch.full((batch_size, 1), real_label).cuda()
  errG = (torch.mean((dis_real - torch.mean(dis_fake) + labels) ** 2) +
          torch.mean((dis_fake - torch.mean(dis_real) - labels) ** 2)) / 2
  return errG
def losses_loss_hinge_rals_dis(dis_fake, dis_real):
  loss_real1,loss_fake1 = losses_loss_hinge_dis(dis_fake, dis_real)
  loss_real2,loss_fake2 = losses_loss_rals_dis(dis_fake, dis_real)
  loss_real=loss_real1+loss_real2
  loss_fake=loss_fake1+loss_fake2
  return loss_real/2, loss_fake/2
def losses_loss_hinge_rals_gen(dis_fake, dis_real):
  loss1 = losses_loss_hinge_gen(dis_fake, dis_real)
  loss2 = losses_loss_rals_gen(dis_fake, dis_real)
  return (loss1+loss2)/2
losses_generator_loss = losses_loss_hinge_gen
losses_discriminator_loss = losses_loss_hinge_dis
######################### losses.py end #########################
######################## train_fns.py start ########################
def train_fns_dummy_training_function():
  def train(x, y):
    return {}
  return train
def train_fns_GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
  if config['loss_version']=='hinge2':
    generator_loss = losses_loss_hinge_gen
    discriminator_loss = losses_loss_hinge_dis2
  elif config['loss_version']=='rals':
    generator_loss = losses_loss_rals_gen
    discriminator_loss = losses_loss_rals_dis
  elif config['loss_version']=='hinge_rals':
    generator_loss = losses_loss_hinge_rals_gen
    discriminator_loss = losses_loss_hinge_rals_dis
  else:
    generator_loss = losses_loss_hinge_gen
    discriminator_loss = losses_loss_hinge_dis
  def train_mode_seeing(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    if config['toggle_grads']:
      utils_toggle_grad(D, True)
      utils_toggle_grad(G, False)
    for step_index in range(config['num_D_steps']):
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_fake_features, D_real, D_real_features = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                                              x[counter], y[counter], train_G=False,
                                                              split_D=config['split_D'])
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
      if config['D_ortho'] > 0.0:
        utils_ortho(D, config['D_ortho'])
      if config['clip_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(D.parameters(), config['clip_norm'])
      D.optim.step()
    if config['toggle_grads']:
      utils_toggle_grad(D, False)
      utils_toggle_grad(G, True)
    G.optim.zero_grad()
    for accumulation_index in range(config['num_G_accumulations']):
      z_.sample_()
      y_.sample_()
      z1=z_.data.clone().detach()
      D_fake1, _, fake_image1= GD(z1, y_, train_G=True, split_D=config['split_D'],return_G_z=True)
      G_loss1 = generator_loss(D_fake1,D_real.detach()) / float(config['num_G_accumulations'])
      z_.sample_()
      z2 =z_.data.clone().detach()
      D_fake2, _ ,fake_image2= GD(z2, y_, train_G=True, split_D=config['split_D'],return_G_z=True)
      G_loss2 = generator_loss(D_fake2,D_real.detach()) / float(config['num_G_accumulations'])
      G_loss_gan=G_loss1+G_loss2
      lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean( torch.abs(z2 - z1))
      eps = 1 * 1e-5
      loss_lz = 1 / (lz + eps)
      G_loss=G_loss_gan+loss_lz
      G_loss.backward()
    if config['G_ortho'] > 0.0:
      utils_ortho(G, config['G_ortho'],
                  blacklist=[param for param in G.shared.parameters()])
    if config['clip_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(G.parameters(), config['clip_norm'])
    G.optim.step()
    if config['ema']:
      ema.update(state_dict['itr'])
    out = {'G_loss': float(G_loss.item()),
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    return out
  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    if config['toggle_grads']:
      utils_toggle_grad(D, True)
      utils_toggle_grad(G, False)
    for step_index in range(config['num_D_steps']):
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        ret = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                                              x[counter], y[counter], train_G=False,
                                                              split_D=config['split_D'])
        if len(ret)>2:
          D_fake, D_fake_features, D_real, D_real_features=ret
        else:
          D_fake,  D_real = ret
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
      if config['D_ortho'] > 0.0:
        utils_ortho(D, config['D_ortho'])
      if config['clip_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(D.parameters(), config['clip_norm'])
      D.optim.step()
    if config['toggle_grads']:
      utils_toggle_grad(D, False)
      utils_toggle_grad(G, True)
    G.optim.zero_grad()
    for accumulation_index in range(config['num_G_accumulations']):
      z_.sample_()
      y_.sample_()
      ret= GD(z_, y_, train_G=True, split_D=config['split_D'])
      if len(ret)==2:
        D_fake, D_fake_features=ret
      else:
        D_fake = ret
      G_loss = generator_loss(D_fake, D_real.detach()) / float(config['num_G_accumulations'])
      if config['gdpp_loss']:
        gdpp_loss = losses_GDPPLoss(D_fake_features, D_real_features.detach(), backward=False)
        gdpp_loss = gdpp_loss / float(config['num_G_accumulations'])
        G_loss += gdpp_loss
      G_loss.backward()
    if config['G_ortho'] > 0.0:
      utils_ortho(G, config['G_ortho'],
                  blacklist=[param for param in G.shared.parameters()])
    if config['clip_norm'] is not None:
      torch.nn.utils.clip_grad_norm_(G.parameters(), config['clip_norm'])
    G.optim.step()
    if config['ema']:
      ema.update(state_dict['itr'])
    out = {'G_loss': float(G_loss.item()),
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item())}
    return out
  if config['mode_seeking_loss']:
    return train_mode_seeing
  else:
    return train
def train_fns_generate_submission(sample, config, experiment_name):
    print('generate submission...')
    image_num = 10000
    output_dir = f"{config['samples_root']}/{experiment_name}/submission"
    os.makedirs(output_dir, exist_ok=True)
    image_list = []
    cnt = 0
    with torch.no_grad():
        while cnt < image_num:
            images, labels_val = sample()
            image_list += [images.data.cpu()]
            cnt += len(images)
    image_list = torch.cat(image_list, 0)[:image_num]
    for i,image in enumerate(image_list):
        image_fname = f'{output_dir}/{i}.png'
        image = transforms.ToPILImage()((image+1)/2)
        image = image.resize((64, 64), Image.ANTIALIAS)
        image.save(image_fname)
    import shutil
    shutil.make_archive('images', 'zip', output_dir)
    log_dir = f"{config['logs_root']}/{experiment_name}"
    if os.path.exists(log_dir):
        log_list = os.listdir(log_dir)
        for i in log_list:
            if i.count('loss') or i.count('metalog'):
                shutil.copy(f'{log_dir}/{i}', f'./')
    print('generate submission done')
def train_fns_save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                    state_dict, config, experiment_name):
  utils_save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
  if config['num_save_copies'] > 0:
    utils_save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' %  state_dict['save_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  if config['accumulate_stats']:
    utils_accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  with torch.no_grad():
    if config['parallel']:
      fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
    else:
      fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                  experiment_name,
                                                  state_dict['itr'])
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
  num_classes = int(np.minimum(120, config['n_classes']))
  utils_sample_sheet(which_G,
                     classes_per_sheet=utils_classes_per_sheet_dict[config['dataset']],
                     num_classes=num_classes,
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_)
  for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    utils_interp_sheet(which_G,
                       num_per_sheet=16,
                       num_midpoints=8,
                       num_classes=num_classes,
                       parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       sheet_number=0,
                       fix_z=fix_z, fix_y=fix_y, device='cuda')
def train_fns_test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils_accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  IS_mean, IS_std, FID = get_inception_metrics(sample, 
                                               config['num_inception_images'],
                                               num_splits=10)
  print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
  if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
    or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
    print('%s improved over previous best, saving checkpoint...' % config['which_best'])
    utils_save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
               IS_std=float(IS_std), FID=float(FID))
######################### train_fns.py end #########################
######################## __init__.py start ########################
######################## batchnorm.py start ########################
import collections
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
######################## comm.py start ########################
import queue
import threading
__all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']
class FutureResult(object):
    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()
    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res
_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])
class SlavePipe(_SlavePipeBase):
    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret
class SyncMaster(object):
    def __init__(self, master_callback):
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False
    def __getstate__(self):
        return {'master_callback': self._master_callback}
    def __setstate__(self, state):
        self.__init__(state['master_callback'])
    def register_slave(self, identifier):
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)
    def run_master(self, master_msg):
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]
    @property
    def nr_slaves(self):
        return len(self._registry)
######################### comm.py end #########################
__all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']
def _sum_ft(tensor):
    return tensor.sum(dim=0).sum(dim=-1)
def _unsqueeze_ft(tensor):
    return tensor.unsqueeze(0).unsqueeze(-1)
_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])
class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None
    def forward(self, input, gain=None, bias=None):
        if not (self._is_parallel and self.training):
            out = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)
            if gain is not None:
              out = out + gain
            if bias is not None:
              out = out + bias
            return out
        input_shape = input.size()
        input = input.view(input.size(0), input.size(1), -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))
        if gain is not None:
          output = (input - _unsqueeze_ft(mean)) * (_unsqueeze_ft(inv_std) * gain.squeeze(-1)) + bias.squeeze(-1)
        elif self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)        
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)
    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)
    def _data_parallel_master(self, intermediates):
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))
        return outputs
    def _compute_mean_std(self, sum_, ssum, size):
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        return mean, torch.rsqrt(bias_var + self.eps)
class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)
class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)
class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)
######################### batchnorm.py end #########################
######################## replicate.py start ########################
from torch.nn.parallel.data_parallel import DataParallel
__all__ = [
    'CallbackContext',
    'execute_replication_callbacks',
    'DataParallelWithCallback',
    'patch_replication_callback'
]
class CallbackContext(object):
    pass
def execute_replication_callbacks(modules):
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)
class DataParallelWithCallback(DataParallel):
    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules
def patch_replication_callback(data_parallel):
    assert isinstance(data_parallel, DataParallel)
    old_replicate = data_parallel.replicate
    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules
    data_parallel.replicate = new_replicate
######################### replicate.py end #########################
######################### __init__.py end #########################
######################## BigGAN.py start ########################
######################## layers.py start ########################
from math import sqrt
def layers_proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())
def layers_gram_schmidt(x, ys):
  for y in ys:
    x = x - layers_proj(x, y)
  return x
def layers_power_iteration(W, u_, update=True, eps=1e-12):
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    with torch.no_grad():
      v = torch.matmul(u, W)
      v = F.normalize(layers_gram_schmidt(v, vs), eps=eps)
      vs += [v]
      u = torch.matmul(v, W.t())
      u = F.normalize(layers_gram_schmidt(u, us), eps=eps)
      us += [u]
      if update:
        u_[i][:] = u
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
  return svs, us, vs
class layers_identity(nn.Module):
  def forward(self, input):
    return input
class layers_SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    self.num_itrs = num_itrs
    self.num_svs = num_svs
    self.transpose = transpose
    self.eps = eps
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    for _ in range(self.num_itrs):
      svs, us, vs = layers_power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]
class layers_SNConv2d(nn.Conv2d, layers_SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    layers_SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)
class layers_SNLinear(nn.Linear, layers_SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    layers_SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)
class layers_SNEmbedding(nn.Embedding, layers_SN):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=None, 
               max_norm=None, norm_type=2, scale_grad_by_freq=False,
               sparse=False, _weight=None,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq, 
                          sparse, _weight)
    layers_SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
  def forward(self, x):
    return F.embedding(x, self.W_())
class layers_Attention(nn.Module):
  def __init__(self, ch, which_conv=layers_SNConv2d, name='attention'):
    super(layers_Attention, self).__init__()
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None,style=None):
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])    
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x
class layers_CBAM(nn.Module):
    def __init__(self, channels, which_conv=layers_SNConv2d,reduction=8,attention_kernel_size=3):
        super(layers_CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = which_conv(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = which_conv(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = which_conv(2, 1,
                                           kernel_size = attention_kernel_size,
                                           stride=1,
                                           padding = attention_kernel_size//2)
        self.sigmoid_spatial = nn.Sigmoid()
    def forward(self, x, y=None,style=None):
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        x = module_input * x
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x
def layers_fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
  scale = torch.rsqrt(var + eps)
  if gain is not None:
    scale = scale * gain
  shift = mean * scale
  if bias is not None:
    shift = shift - bias
  return x * scale - shift
def layers_manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
  float_x = x.float()
  m = torch.mean(float_x, [0, 2, 3], keepdim=True)
  m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
  var = (m2 - m **2)
  var = var.type(x.type())
  m = m.type(x.type())
  if return_mean_var:
    return layers_fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
  else:
    return layers_fused_bn(x, m, var, gain, bias, eps)
class layers_myBN(nn.Module):
  def __init__(self, num_channels, eps=1e-5, momentum=0.1):
    super(layers_myBN, self).__init__()
    self.momentum = momentum
    self.eps = eps
    self.momentum = momentum
    self.register_buffer('stored_mean', torch.zeros(num_channels))
    self.register_buffer('stored_var',  torch.ones(num_channels))
    self.register_buffer('accumulation_counter', torch.zeros(1))
    self.accumulate_standing = False
  def reset_stats(self):
    self.stored_mean[:] = 0
    self.stored_var[:] = 0
    self.accumulation_counter[:] = 0
  def forward(self, x, gain, bias):
    if self.training:
      out, mean, var = layers_manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
      if self.accumulate_standing:
        self.stored_mean[:] = self.stored_mean + mean.data
        self.stored_var[:] = self.stored_var + var.data
        self.accumulation_counter += 1.0
      else:
        self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
        self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
      return out
    else:         
      mean = self.stored_mean.view(1, -1, 1, 1)
      var = self.stored_var.view(1, -1, 1, 1)
      if self.accumulate_standing:
        mean = mean / self.accumulation_counter
        var = var / self.accumulation_counter
      return layers_fused_bn(x, mean, var, gain, bias, self.eps)
def layers_groupnorm(x, norm_style):
  if 'ch' in norm_style:
    ch = int(norm_style.split('_')[-1])
    groups = max(int(x.shape[1]) // ch, 1)
  elif 'grp' in norm_style:
    groups = int(norm_style.split('_')[-1])
  else:
    groups = 16
  return F.group_norm(x, groups)
class layers_ccbn(nn.Module):
  def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn',
               style_linear=None,dim_z=0,no_conditional=False,skip_z=False):
    super(layers_ccbn, self).__init__()
    self.output_size, self.input_size = output_size, input_size
    self.gain = which_linear(input_size, output_size)
    self.bias = which_linear(input_size, output_size)
    self.eps = eps
    self.momentum = momentum
    self.cross_replica = cross_replica
    self.mybn = mybn
    self.norm_style = norm_style
    self.no_conditional = no_conditional
    if self.cross_replica:
      self.bn = SynchronizedBatchNorm2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    elif self.mybn:
      self.bn = layers_myBN(output_size, self.eps, self.momentum)
    elif self.norm_style in ['bn', 'in']:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))
    if style_linear is not None:
      if skip_z:
        self.style = style_linear(dim_z*2, output_size * 2)
      else:
        self.style = style_linear(dim_z, output_size * 2)
      self.style.bias.data[:output_size] = 1
      self.style.bias.data[output_size:] = 0
  def forward(self, x, y,style=None):
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1)
    if self.mybn or self.cross_replica:
      if style is None:
        return self.bn(x, gain=gain, bias=bias)
      else:
        out = self.bn(x, gain=gain, bias=bias)
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = gamma * out + beta
        return out
    else:
      if self.norm_style == 'bn':
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'in':
        out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'gn':
        out = layers_groupnorm(x, self.normstyle)
      elif self.norm_style == 'nonorm':
        out = x
      if style is not None:
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        if not self.no_conditional:
          out=out * gain + bias
        out = gamma * out + beta
        return out
      else:
        return out * gain + bias
  def extra_repr(self):
    s = 'out: {output_size}, in: {input_size},'
    s +=' cross_replica={cross_replica}'
    return s.format(**self.__dict__)
class layers_bn(nn.Module):
  def __init__(self, output_size,  eps=1e-5, momentum=0.1,
                cross_replica=False, mybn=False):
    super(layers_bn, self).__init__()
    self.output_size= output_size
    self.gain = P(torch.ones(output_size), requires_grad=True)
    self.bias = P(torch.zeros(output_size), requires_grad=True)
    self.eps = eps
    self.momentum = momentum
    self.cross_replica = cross_replica
    self.mybn = mybn
    if self.cross_replica:
      self.bn = SynchronizedBatchNorm2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)    
    elif mybn:
      self.bn = layers_myBN(output_size, self.eps, self.momentum)
    else:     
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))
  def forward(self, x, y=None):
    if self.cross_replica or self.mybn:
      gain = self.gain.view(1,-1,1,1)
      bias = self.bias.view(1,-1,1,1)
      return self.bn(x, gain=gain, bias=bias)
    else:
      return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                          self.bias, self.training, self.momentum, self.eps)
from torch.autograd import Function
class layers_PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
class layers_BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )
        return grad_input
    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )
        return grad_input, None, None
class layers_BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])
        return output
    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = layers_BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)
        return grad_input, None, None
layers_blur = layers_BlurFunction.apply
class layers_Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()
        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])
        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))
    def forward(self, input):
        return layers_blur(input, self.weight, self.weight_flip)
class layers_EqualLR:
    def __init__(self, name):
        self.name = name
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)
    @staticmethod
    def apply(module, name):
        fn = layers_EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn
    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)
def layers_equal_lr(module, name='weight'):
    layers_EqualLR.apply(module, name)
    return module
class layers_EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = layers_equal_lr(conv)
    def forward(self, input):
        return self.conv(input)
class layers_EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = layers_equal_lr(linear)
    def forward(self, input):
        return self.linear(input)
class layers_NoiseInjection(nn.Module):
  def __init__(self, channel):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))
  def forward(self, image, noise):
    return image + self.weight * noise
class layers_AdaptiveInstanceNorm(nn.Module):
  def __init__(self, in_channel, style_dim):
    super().__init__()
    self.norm = nn.InstanceNorm2d(in_channel)
    self.style = layers_EqualLinear(style_dim, in_channel * 2)
    self.style.linear.bias.data[:in_channel] = 1
    self.style.linear.bias.data[in_channel:] = 0
  def forward(self, input, style):
    style = self.style(style).unsqueeze(2).unsqueeze(3)
    gamma, beta = style.chunk(2, 1)
    out = self.norm(input)
    out = gamma * out + beta
    return out
class layers_GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=layers_bn, activation=None, 
               upsample=None,add_blur=False,add_noise=False,add_style=False):
    super(layers_GBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    self.add_blur=add_blur
    self.add_noise=add_noise
    self.add_style=add_style
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    if self.add_blur:
      self.blur = layers_Blur(out_channels)
    if self.add_noise:
      self.noise1 = layers_equal_lr(layers_NoiseInjection(out_channels))
      self.noise2 = layers_equal_lr(layers_NoiseInjection(out_channels))
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
    self.bn1 = self.which_bn(in_channels)
    self.bn2 = self.which_bn(out_channels)
    self.upsample = upsample
  def forward(self, x, y,style=None):
    h = self.activation(self.bn1(x, y,style))
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    if self.add_blur:
      h = self.blur(h)
    if self.add_noise:
      batch=x.size(0)
      size=x.size(2)
      noise=torch.randn(batch, 1, size, size).cuda()
      h = self.noise1(h, noise)
    h = self.activation(self.bn2(h, y,style))
    h = self.conv2(h)
    if self.add_noise:
      h = self.noise2(h, noise)
    if self.learnable_sc:       
      x = self.conv_sc(x)
    return h + x
class layers_StyleLayer(nn.Module):
  def __init__(self,dim_z,which_linear,activation):
    super(layers_StyleLayer, self).__init__()
    self.which_linear=which_linear(dim_z,dim_z)
    self.activation=activation
  def forward(self, x):
    x=self.which_linear(x)
    x=self.activation(x)
    return x
class layers_DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=layers_SNConv2d, wide=True,
               preactivation=False, activation=None, downsample=None,):
    super(layers_DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
    self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x
  def forward(self, x):
    if self.preactivation:
      h = F.relu(x)
    else:
      h = x    
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)     
    return h + self.shortcut(x)
######################### layers.py end #########################
def BigGAN_G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch['512'] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch['256'] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch['128'] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch['128x']  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch['96a'] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [6, 12, 24, 48, 96],
               'attention' : {6*2**i: (6*2**i in [int(item) for item in attention.split('_')])
                              for i in range(0,5)}}
  arch['96']  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [12, 24, 48, 96],
               'attention' : {12*2**i: (6*2**i in [int(item) for item in attention.split('_')]) for i in range(0,4)}}
  arch['80'] = {'in_channels': [ch * item for item in [16, 16, 8, 4]],
              'out_channels': [ch * item for item in [16, 8, 4, 2]],
              'upsample': [True] * 4,
              'resolution': [10, 20, 40, 80],
              'attention': {10 * 2 ** i: (2 ** i in [int(item) for item in attention.split('_')]) for i in range(0, 4)}}
  arch['64']  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch['32']  = {'in_channels' :  [ch * item for item in [16, 8, 4]],
               'out_channels' : [ch * item for item in [8, 4, 2]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}
  return arch
class BigGAN_Generator(nn.Module):
  def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=1000,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn',
               add_blur=False,add_noise=False,add_style=False,
               style_mlp=6,
               attn_style='nl',
               no_conditional=False,
               sched_version='default',
               num_epochs=500,
               arch=None,
               skip_z=False,
               **kwargs):
    super(BigGAN_Generator, self).__init__()
    self.ch = G_ch
    self.dim_z = dim_z
    self.bottom_width = bottom_width
    self.resolution = resolution
    self.kernel_size = G_kernel_size
    self.attention = G_attn
    self.n_classes = n_classes
    self.G_shared = G_shared
    self.shared_dim = shared_dim if shared_dim > 0 else dim_z
    self.hier = hier
    self.cross_replica = cross_replica
    self.mybn = mybn
    self.activation = G_activation
    self.init = G_init
    self.G_param = G_param
    self.norm_style = norm_style
    self.add_blur = add_blur
    self.add_noise = add_noise
    self.add_style = add_style
    self.skip_z = skip_z
    self.BN_eps = BN_eps
    self.SN_eps = SN_eps
    self.fp16 = G_fp16
    if arch is None:
      arch=f'{resolution}'
    self.arch = BigGAN_G_arch(self.ch, self.attention)[arch]
    if self.hier:
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      self.dim_z = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0
    if self.G_param == 'SN':
      self.which_conv = functools.partial(layers_SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers_SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
    if attn_style=='cbam':
      self.which_attn=layers_CBAM
    else:
      self.which_attn = layers_Attention
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    self.which_bn = functools.partial(layers_ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps,
                          style_linear=self.which_linear,
                          dim_z=self.dim_z,
                          no_conditional=no_conditional,
                          skip_z=self.skip_z)
    self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                    else layers_identity())
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **2))
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers_GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None),
                             add_blur=add_blur,
                             add_noise=add_noise,
                                     )
                       ]]
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [self.which_attn(self.arch['out_channels'][index], self.which_conv)]
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    self.output_layer = nn.Sequential(layers_bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 3))
    if self.add_style:
      style_layers = []
      for i in range(style_mlp):
        style_layers.append(layers_StyleLayer(self.dim_z,self.which_linear,self.activation))
      self.style = nn.Sequential(*style_layers)
    if not skip_init:
      self.init_weights()
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
    if G_mixed_precision:
      print('Using fp16 adam in G...')
      self.optim = utils_Adam16(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps, amsgrad=kwargs['amsgrad'])
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps, amsgrad=kwargs['amsgrad'])
    if sched_version=='default':
      self.lr_sched=None
    elif  sched_version=='cal_v0':
      self.lr_sched =optim.lr_scheduler.CosineAnnealingLR(self.optim,
                                T_max=num_epochs, eta_min=self.lr/2, last_epoch=-1)
    elif  sched_version=='cal_v1':
      self.lr_sched =optim.lr_scheduler.CosineAnnealingLR(self.optim,
                                T_max=num_epochs, eta_min=self.lr/4, last_epoch=-1)
    elif  sched_version=='cawr_v0':
      self.lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,
                                T_0=10, T_mult=2, eta_min=self.lr/2)
    elif  sched_version=='cawr_v1':
      self.lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,
                                                                     T_0=25, T_mult=2, eta_min=self.lr/4)
    else:
      self.lr_sched = None
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)
  def forward(self, z, y):
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
    h = self.linear(z)
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    if self.add_style:
      style=self.style(z)
      if self.skip_z:
        style=torch.cat([style,z],-1)
    else:
      style=None
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h, ys[index],style)
    return torch.tanh(self.output_layer(h))
def BigGAN_D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
  arch = {}
  arch['256'] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch['128'] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch['96a'] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [48, 24, 12, 6, 3, 3],
               'attention' : {3*2**i: 3*2**i in [int(item) for item in attention.split('_')]
                              for i in range(0,5)}}
  arch['96']  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [48, 24, 12, 6, 6],
               'attention' : {6*2**i: 6*2**i in [int(item) for item in attention.split('_')]
                              for i in range(0,4)}}
  arch['80']  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [40, 20, 10, 5, 5],
               'attention' : {5*2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(0,5)}}
  arch['64']  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch['32']  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, False, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch
class BigGAN_Discriminator(nn.Module):
  def __init__(self, D_ch=64, D_wide=True, resolution=128,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN',attn_style='nl',
               sched_version='default',
               num_epochs=500,
               arch=None,
               **kwargs):
    super(BigGAN_Discriminator, self).__init__()
    self.ch = D_ch
    self.D_wide = D_wide
    self.resolution = resolution
    self.kernel_size = D_kernel_size
    self.attention = D_attn
    self.n_classes = n_classes
    self.activation = D_activation
    self.init = D_init
    self.D_param = D_param
    self.SN_eps = SN_eps
    self.fp16 = D_fp16
    if arch is None:
      arch=f'{resolution}'
    self.arch = BigGAN_D_arch(self.ch, self.attention)[arch]
    if self.D_param == 'SN':
      self.which_conv = functools.partial(layers_SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers_SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_embedding = functools.partial(layers_SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=self.SN_eps)
    if attn_style == 'cbam':
      self.which_attn = layers_CBAM
    else:
      self.which_attn = layers_Attention
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers_DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [self.which_attn(self.arch['out_channels'][index],
                                             self.which_conv)]
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
    if not skip_init:
      self.init_weights()
    self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
    if D_mixed_precision:
      print('Using fp16 adam in D...')
      self.optim = utils_Adam16(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps, amsgrad=kwargs['amsgrad'])
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps, amsgrad=kwargs['amsgrad'])
    if sched_version=='default':
      self.lr_sched=None
    elif  sched_version=='cal_v0':
      self.lr_sched =optim.lr_scheduler.CosineAnnealingLR(self.optim,
                                T_max=num_epochs, eta_min=self.lr/2, last_epoch=-1)
    elif  sched_version=='cal_v1':
      self.lr_sched =optim.lr_scheduler.CosineAnnealingLR(self.optim,
                                T_max=num_epochs, eta_min=self.lr/4, last_epoch=-1)
    elif  sched_version=='cawr_v0':
      self.lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,
                                T_0=10, T_mult=2, eta_min=self.lr/2)
    elif  sched_version=='cawr_v1':
      self.lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,
                                                                     T_0=25, T_mult=2, eta_min=self.lr/4)
    else:
      self.lr_sched = None
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)
  def forward(self, x, y=None):
    h = x
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    h = torch.sum(self.activation(h), [2, 3])
    features = h
    out = self.linear(h)
    out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
    return out,features
class BigGAN_G_D(nn.Module):
  def __init__(self, G, D):
    super(BigGAN_G_D, self).__init__()
    self.G = G
    self.D = D
  def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
              split_D=False):              
    with torch.set_grad_enabled(train_G):
      G_z = self.G(z, self.G.shared(gy))
      if self.G.fp16 and not self.D.fp16:
        G_z = G_z.float()
      if self.D.fp16 and not self.G.fp16:
        G_z = G_z.half()
    if split_D:
      D_fake,D_fake_features = self.D(G_z, gy)
      if x is not None:
        D_real,D_real_features = self.D(x, dy)
        return D_fake,D_fake_features, D_real,D_real_features
      else:
        if return_G_z:
          return D_fake,D_fake_features, G_z
        else:
          return D_fake,D_fake_features
    else:
      D_input = torch.cat([G_z, x], 0) if x is not None else G_z
      D_class = torch.cat([gy, dy], 0) if dy is not None else gy
      D_out,D_features = self.D(D_input, D_class)
      if x is not None:
        D_fake, D_real= torch.split(D_out, [G_z.shape[0], x.shape[0]]) # D_fake, D_real
        D_fake_features, D_real_features= torch.split(D_features, [G_z.shape[0], x.shape[0]]) # D_fake_features, D_real_features
        return D_fake, D_fake_features, D_real, D_real_features
      else:
        if return_G_z:
          return D_out,D_features, G_z
        else:
          return D_out,D_features
######################### BigGAN.py end #########################
def run(config):
  config['resolution'] = utils_imsize_dict[config['dataset']]
  config['n_classes'] = utils_nclass_dict[config['dataset']]
  config['G_activation'] = utils_activation_dict[config['G_nl']]
  config['D_activation'] = utils_activation_dict[config['D_nl']]
  if config['resume']:
    print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils_update_config_roots(config)
  device = 'cuda'
  if config['base_root']:
    os.makedirs(config['base_root'],exist_ok=True)
  utils_seed_rng(config['seed'])
  utils_prepare_root(config)
  torch.backends.cudnn.benchmark = True
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils_name_from_config(config))
  print('Experiment name is %s' % experiment_name)
  G = BigGAN_Generator(**config).to(device)
  D = BigGAN_Discriminator(**config).to(device)
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = BigGAN_Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
    ema = utils_ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
  GD = BigGAN_G_D(G, D)
  print(G)
  print(D)
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
  if config['resume']:
    print('Loading weights...')
    utils_load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None,
                       )
    if G.lr_sched is not None:G.lr_sched.step(state_dict['epoch'])
    if D.lr_sched is not None:D.lr_sched.step(state_dict['epoch'])
  if config['parallel']:
    GD = nn.DataParallel(GD)
    if config['cross_replica']:
      patch_replication_callback(GD)
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils_MetricsLogger(test_metrics_fname, 
                                 reinitialize=(not config['resume']))
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils_MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  utils_write_metadata(config['logs_root'], experiment_name, config, state_dict)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils_get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})
  if not config['on_kaggle']:
    get_inception_metrics = inception_utils_prepare_inception_metrics(config['base_root'],config['dataset'], config['parallel'], config['no_fid'])
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils_prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'],z_dist=config['z_dist'],
                             threshold=config['truncated_threshold'])
  fixed_z, fixed_y = utils_prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'],z_dist=config['z_dist'],
                                       threshold=config['truncated_threshold'])
  fixed_z.sample_()
  fixed_y.sample_()
  if config['which_train_fn'] == 'GAN':
    train = train_fns_GAN_training_function(G, D, GD, z_, y_, 
                                            ema, state_dict, config)
  else:
    train = train_fns_dummy_training_function()
  sample = functools.partial(utils_sample,
                              G=(G_ema if config['ema'] and config['use_ema']
                                 else G),
                              z_=z_, y_=y_, config=config)
  print('Beginning training at epoch %d...' % state_dict['epoch'])
  by_epoch=False if config['save_every']>100 else True
  start_time = time.time()
  for epoch in range(state_dict['epoch'], config['num_epochs']):
    if config['on_kaggle']:
      pbar = loaders[0]
    elif config['pbar'] == 'mine':
      pbar = utils_progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      pbar = tqdm(loaders[0])
    epoch_start_time = time.time()
    for i, (x, y) in enumerate(pbar):
      state_dict['itr'] += 1
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      metrics = train(x, y)
      train_log.log(itr=int(state_dict['itr']), **metrics)
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), 
                      **{**utils_get_SVs(G, 'G'), **utils_get_SVs(D, 'D')})
      if config['on_kaggle']:
        if i == len(loaders[0])-1:
          metrics_str = ', '.join(['%s : %+4.3f' % (key, metrics[key]) for key in metrics])
          epoch_time = (time.time()-epoch_start_time) / 60
          total_time = (time.time()-start_time) / 60
          print(f"[{epoch+1}/{config['num_epochs']}][{epoch_time:.1f}min/{total_time:.1f}min] {metrics_str}")
      elif config['pbar'] == 'mine':
        if D.lr_sched is None:
          print(', '.join(['epoch:%d' % (epoch+1),'itr: %d' % state_dict['itr']]
                       + ['%s : %+4.3f' % (key, metrics[key])
                       for key in metrics]), end=' ')
        else:
          print(', '.join(['epoch:%d' % (epoch+1),'lr:%.5f' % D.lr_sched.get_lr()[0] ,'itr: %d' % state_dict['itr']]
                       + ['%s : %+4.3f' % (key, metrics[key])
                       for key in metrics]), end=' ')
      if not by_epoch:
        if not (state_dict['itr'] % config['save_every']) and not config['on_kaggle']:
          if config['G_eval_mode']:
            print('Switchin G to eval mode...')
            G.eval()
            if config['ema']:
              G_ema.eval()
          train_fns_save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                    state_dict, config, experiment_name)
        if not (state_dict['itr'] % config['test_every']) and not config['on_kaggle']:
          if config['G_eval_mode']:
            print('Switchin G to eval mode...')
            G.eval()
          train_fns_test(G, D, G_ema, z_, y_, state_dict, config, sample,
                         get_inception_metrics, experiment_name, test_log)
    if by_epoch:
      if not ((epoch+1) % config['save_every']) and not config['on_kaggle']:
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        train_fns_save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                  state_dict, config, experiment_name)
      if not ((epoch+1) % config['test_every']) and not config['on_kaggle']:
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
        train_fns_test(G, D, G_ema, z_, y_, state_dict, config, sample,
                       get_inception_metrics, experiment_name, test_log)
      if G_ema is not None and (epoch+1) % config['test_every'] == 0 and not config['on_kaggle']:
        torch.save(G_ema.state_dict(),  '%s/%s/G_ema_epoch_%03d.pth' %
                   (config['weights_root'], config['experiment_name'], epoch+1))
    state_dict['epoch'] += 1
    if G.lr_sched is not None:
      G.lr_sched.step()
    if D.lr_sched is not None:
      D.lr_sched.step()
  if config['on_kaggle']:
    train_fns_generate_submission(sample, config, experiment_name)
def main():
  parser = utils_prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)
if __name__ == '__main__':
  main()
######################### train.py end #########################
_kaggle_end_ = timer()
_kaggle_time_ = (_kaggle_end_ - _kaggle_start_) / 60.
print('kaggle time: %.2f min' % _kaggle_time_)
# version: 2019-08-08 00:22:20 end
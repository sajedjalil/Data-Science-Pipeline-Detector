import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from ignite.metrics import EpochMetric
from ignite.engine import create_supervised_evaluator


HEIGHT = 137
WIDTH = 236
SIZE = 128

NUM_GRAPHEME_ROOT = 168
NUM_VOWEL = 11
NUM_CONSONANT = 7


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class MyModel(nn.Module):

    def __init__(self, architecture, img_type, pretrained):
        super().__init__()

        self.img_type = img_type

        if architecture == 'resnet34':
            self.main = models.resnet34(pretrained=pretrained)
            self.main.fc = nn.Linear(512, NUM_GRAPHEME_ROOT + NUM_CONSONANT + NUM_VOWEL)
            self.main.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))

        elif architecture == 'resnet50':
            self.main = models.resnet50(pretrained=pretrained)
            self.main.fc = nn.Linear(2048, NUM_GRAPHEME_ROOT + NUM_CONSONANT + NUM_VOWEL)
            self.main.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))

        elif architecture == 'densenet121':
            self.main = models.densenet121(pretrained=pretrained)
            self.main.classifier = nn.Linear(1024, NUM_GRAPHEME_ROOT + NUM_CONSONANT + NUM_VOWEL)
            self.main.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))

        else:
            raise ValueError('Unknown Architecture')

        self.main.avgpool = GeM()

        self.resizer = nn.Upsample(size=(128, 128), mode='bicubic')

    def forward(self, x):

        img_raw, img_square = x

        if self.img_type == 'square_crop':
            out = self.main.forward(img_square)
        elif self.img_type == 'square_resize':
            out = self.main.forward(self.resizer(img_raw))
        elif self.img_type == 'raw':
            out = self.main.forward(img_raw)
        else:
            raise ValueError('Unknown image type')

        return out


def build_model(architecture, img_type, pretrained):
    model = MyModel(architecture, img_type, pretrained)
    return model


class EnsembleModels(nn.Module):

    def __init__(self, dir_model, list_models, device):
        super().__init__()

        self.models = list()

        for model_name in list_models:

            print(f'==> load model {model_name}')

            with open(str(params.dir_model / f'{model_name}.json'), 'r') as f:
                model_conf = json.load(f)

            model = build_model(model_conf['architecture'],
                                model_conf['img_type'],
                                pretrained=False)
            model.load_state_dict(torch.load(dir_model / '{}.pth'.format(model_conf['model_name']),
                                             map_location=torch.device(device)))
            model.to(device)

            self.models.append(model)

    def forward(self, x):
        list_out = [model.forward(x) for model in self.models]
        return torch.mean(torch.stack(list_out, 0), 0)


class BengaliDataset(Dataset):

    def __init__(self, dir_img: Path=None, list_images=None, train_csv=None, images_array=None,
                 is_aug=False, get_augmenter_func=None):

        self.dir_img = dir_img
        self.images_array = images_array
        self.is_aug = is_aug

        if self.is_aug:
            self.augmenter = get_augmenter_func()

        if list_images is None:
            self.list_file_names = sorted(os.listdir(str(self.dir_img)))
        else:
            self.list_file_names = list_images

        if train_csv is None:
            self.train_csv = None
        else:
            train_csv = train_csv.copy()
            train_csv.index = train_csv['image_id']
            self.train_csv = train_csv.to_dict(orient='index')

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):

        if self.images_array is None:
            if isinstance(idx, str):
                file_name = idx
            else:
                file_name = self.list_file_names[idx]
            img = cv2.imread(str(self.dir_img / file_name))[:, :, [0]]

        else:
            img = self.images_array[idx]

        img_raw = np.copy(img)

        if np.ndim(img) == 3:
            img_square = crop_resize(img[:, :, 0])
        else:
            img_square = crop_resize(img)
            img = np.expand_dims(img, axis=2)

        if np.ndim(img_square) == 2:
            img_square = np.expand_dims(img_square, axis=2)

        if self.is_aug:
            data = {'image': img_raw}
            augmented = self.augmenter(**data)
            img_raw = augmented['image']

            data = {'image': img_square}
            augmented = self.augmenter(**data)
            img_square = augmented['image']

        img_raw = self.to_tensor(img_raw)
        img_square = self.to_tensor(img_square)

        if self.train_csv is not None:
            row = self.train_csv[Path(file_name).stem]
            targets = torch.LongTensor([
                row['grapheme_root'],
                row['vowel_diacritic'],
                row['consonant_diacritic']
            ])

            one_hot_root = np.zeros(NUM_GRAPHEME_ROOT)
            one_hot_root[row['grapheme_root']] = 1.0

            one_hot_vowel = np.zeros(NUM_VOWEL)
            one_hot_vowel[row['vowel_diacritic']] = 1.0

            one_hot_consonant = np.zeros(NUM_CONSONANT)
            one_hot_consonant[row['consonant_diacritic']] = 1.0

            one_hot_targets = torch.from_numpy(
                np.concatenate([one_hot_root, one_hot_vowel, one_hot_consonant])
            )

        else:
            targets = torch.zeros(3)
            one_hot_targets = torch.zeros(NUM_GRAPHEME_ROOT + NUM_VOWEL + NUM_CONSONANT)

        return (img_raw.float(), img_square.float()), (targets.long(), one_hot_targets.float())

    def __len__(self):
        return len(self.list_file_names)

    def get_list_images(self):
        return self.list_file_names


def bbox(img):
    """
    https://www.kaggle.com/iafoss/image-preprocessing-128x128
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16):
    """
    https://www.kaggle.com/iafoss/image-preprocessing-128x128
    """
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin, ymax-ymin
    l = max(lx, ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img, (size, size))


def expand_parquet(dir_input: Path, target_file: str):

    list_imgs = list()
    list_names = list()

    df = pd.read_parquet(str(dir_input / target_file))
    data_array = df.iloc[:, 1:].values

    for i, name in enumerate(tqdm(df['image_id'], miniters=100, total=len(df))):

        # the input is inverted
        target = 255 - data_array[i].reshape(HEIGHT, WIDTH).astype(np.uint8)

        # normalize each image by its max val
        img = (target * (255.0 / target.max())).astype(np.uint8)

        list_imgs.append(img)
        list_names.append(name)

    return np.asarray(list_imgs), list_names


def bengali_argmax(y_pred, y):

    probs_grapheme_root = y_pred[:, :NUM_GRAPHEME_ROOT]
    probs_vowel_diacritic = y_pred[:, NUM_GRAPHEME_ROOT:NUM_GRAPHEME_ROOT+NUM_VOWEL]
    probs_consonant_diacritic = y_pred[:, NUM_GRAPHEME_ROOT+NUM_VOWEL:]

    argmax_grapheme_root = probs_grapheme_root.argmax(dim=1)
    argmax_vowel_diacritic = probs_vowel_diacritic.argmax(dim=1)
    argmax_consonant_diacritic = probs_consonant_diacritic.argmax(dim=1)

    return torch.stack([argmax_grapheme_root, argmax_vowel_diacritic, argmax_consonant_diacritic], 1)


def infer(images_array: np.ndarray, list_names: list,
          dir_dataset: Path, dir_model: Path, list_models: str, seed: int):

    test_csv = pd.read_csv(dir_dataset / 'test.csv')
    test_ids = list_names

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EnsembleModels(dir_model, list_models, device)

    test_dataset = BengaliDataset(None, list_images=[f'{s}.png' for s in test_ids], images_array=images_array)

    batch_size = 8
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    metrics = {
        'output': EpochMetric(bengali_argmax, output_transform=lambda x: (x[0], x[1][0])),
    }

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    evaluator.run(test_loader)
    output = evaluator.state.metrics['output']

    output_df = pd.DataFrame(
        output.cpu().numpy(), index=test_ids,
        columns=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
    )
    print(output_df)

    test_csv = test_csv.query('image_id in @list_names').reset_index(drop=True)

    submission = pd.DataFrame(test_csv['row_id'], columns=['row_id'])
    submission_target = np.zeros(len(test_csv), dtype=np.int32)

    for i, row in test_csv.iterrows():
        submission_target[i] = output_df.loc[row['image_id'], row['component']]

    submission['target'] = submission_target
    return submission


def main():

    tic = time.time()

    print(f'torch version {torch.__version__}')
    print(f'numpy version {np.__version__}')
    print(f'pandas version {pd.__version__}')

    target_files = [
        'test_image_data_0.parquet',
        'test_image_data_1.parquet',
        'test_image_data_2.parquet',
        'test_image_data_3.parquet'
    ]

    list_submissions = list()

    list_models = params.model_names.split(',')

    for target_file in target_files:

        print('==> expand parquet')
        images_array, list_names = expand_parquet(params.dir_dataset, target_file)

        list_submissions.append(
            infer(
                images_array,
                list_names,
                dir_dataset=params.dir_dataset,
                dir_model=params.dir_model,
                list_models=list_models,
                seed=1048
            )
        )

    submission = pd.concat(list_submissions, axis=0)
    submission.to_csv('submission.csv', index=False)

    elapsed_time = time.time() - tic
    print(f'elapsed time: {elapsed_time / 60.0:.1f} [min]')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dir_dataset',
        default=Path('/kaggle/input/bengaliai-cv19'),
        type=Path
    )

    parser.add_argument(
        '--dir_model',
        default=Path('/kaggle/input/bengali-nn-5models'),
        type=Path
    )

    parser.add_argument(
        '--model_names',
        default='model_01,model_02,model_03,model_04,model_05',
        type=str
    )

    params = parser.parse_args()

    main()

import logging
import multiprocessing
import os
import pathlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from typing import List, Tuple
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

INPUT_DIR = os.path.join('..', 'input')

DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
IMAGE_EXT = '.jpg'
SAMPLE_SUBMISSION_CSV = os.path.join(DATASET_DIR, 'sample_submission.csv')
KERNEL_DATASET = os.path.join(INPUT_DIR, 'landmarkdencenetweights')
LANDMARK_IDS = os.path.join(KERNEL_DATASET, "landmark_ids.txt")
LANDMARK_IDS_SANITY_CHECK = "landmark_ids_sanity_check.txt"
SUBMISSION_FILE_PATH = "submission.csv"


if __debug__:
    logging.getLogger().setLevel(logging.DEBUG)


class DenseNet121(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super(DenseNet121, self).__init__()
        self.num_classes = num_classes
        self.default_dense_net = models.densenet121(pretrained=pretrained, progress=True)
        logging.debug(f"NUM CLASSES: {num_classes}")
        self.hat = nn.Linear(1000, num_classes)

    def forward(self, x):
        embedding = self.default_dense_net(x)
        final_probs_logits = self.hat(embedding)
        return final_probs_logits


@dataclass
class Config:
    working_dir = "/home/den/prj/kaggle/landmark/wd"
    experiment_name = "top_10_finetune"
    model_factory: Callable = DenseNet121
    model_weights: Optional[str] = os.path.join(KERNEL_DATASET, "model_epoch_2.pt")
    train_images_per_landmark: int = 1
    batch_size: int = 32
    test_batch_size: int = 32
    epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 1e-4
    log_interval: int = 100
    seed: int = 43
    use_cuda: bool = torch.cuda.is_available()
    save_model: bool = True
    num_workers: int = multiprocessing.cpu_count()
    transform: Callable = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    scheduler_kwargs: dict = field(default_factory=lambda: {'step_size': 10000, 'gamma': 0.9})

    def __post_init__(self):
        self.experiment_dir = os.path.join(self.working_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)


cfg = Config()
device = torch.device("cuda" if cfg.use_cuda else "cpu")

logging.debug(f"DEVICE: {device}")
logging.debug(f"CPUS: {cfg.num_workers}")
logging.debug(cfg)


def get_all_landmark_ids(from_file: str = LANDMARK_IDS) -> List[str]:
    with open(from_file, 'r') as f:
        lmk_ids = f.readline().split(" ")
    return lmk_ids


all_landmark_ids = get_all_landmark_ids(from_file=LANDMARK_IDS)


class Subset(Enum):
    train = '0123456789ab'
    val = 'cdef'

    @classmethod
    def from_id(cls, image_id: str):
        if image_id[-1] in cls.train.value:
            return cls.train
        elif image_id[-1] in cls.val.value:
            return cls.val
        else:
            raise ValueError(f"No such subset for {image_id} id")


def filter_labels_by_lmk_ids(labels: List[Tuple[str, str]], landmark_ids: List[str] = get_all_landmark_ids()):
    landmark_ids = set(landmark_ids)
    return [lbl for lbl in labels if lbl[1] in landmark_ids]


def build_landmark_id_to_landmark_number_map(landmark_ids):
    landmark_number_to_id_map = sorted([lmk_id for lmk_id in landmark_ids], key=lambda lmk_id: int(lmk_id))
    landmark_id_to_number_map = {lmk_id: lmk_number for lmk_number, lmk_id in enumerate(landmark_ids)}
    return landmark_id_to_number_map, landmark_number_to_id_map


def parse_csv_labels(csv_lines: List[str], delimiter: str = ',', eol: str = '\n') -> (str, List[str]):
    def parse_csv_line(line: str):
        return [v.strip() for v in (line[:-len(eol)] if line.endswith(eol) else line).split(delimiter)]
    lines_values = [parse_csv_line(line) for line in csv_lines]
    header, records = lines_values[0], lines_values[1:]
    logging.debug("reading csv with headers %s and total %s lines" % (header, len(records)))
    logging.debug("line example %s" % (records[0], ))
    return header, records


def save_submission_file(predictions: List[Tuple[str, float, int]]):
    with open(SUBMISSION_FILE_PATH, 'w') as f:
        lines = ["id,landmarks\n"]
        for image_id, score, landmark_id in predictions:
            lines.append(f"{image_id},{landmark_id} {score}\n")
        f.writelines(lines)


def read_train_labels(filepath: str):
    with open(filepath, 'r') as f:
        train_csv = f.readlines()
    header, records = parse_csv_labels(train_csv)
    return records


def get_image_path(dataset_dir: str, image_id: str):
    return os.path.join(dataset_dir, image_id[0], image_id[1], image_id[2], image_id + IMAGE_EXT)


def read_image(dataset_dir: str, image_id: str) -> Image:
    return Image.open(get_image_path(dataset_dir, image_id))


def to_pil_image(image_tensor: np.ndarray) -> Image:
    return transforms.ToPILImage()(np.transpose((image_tensor * 255).astype(np.uint8), axes=(1, 2, 0)))


class LandmarkDataset(Dataset):

    def __init__(self, dataset_dir: str, all_landmark_ids: List[str], image_ids: List[str], labels: Optional[List[str]],
                 transform=None):
        if labels:
            assert len(image_ids) == len(labels)
        self.all_landmark_ids: List[str] = all_landmark_ids
        self.landmark_numbers: Dict[str, int] = {lmk_id: lmk_number for lmk_number, lmk_id in enumerate(all_landmark_ids)}
        self.dataset_dir = dataset_dir
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image = read_image(self.dataset_dir, self.image_ids[idx])
            if self.transform:
                image = self.transform(image)
            if self.labels:
                return image, self.landmark_numbers[self.labels[idx]]
            else:
                return image, self.image_ids[idx]
        except Exception as e:
            print(e)

    def __len__(self):
        return len(self.image_ids)


class LandmarkSampler(Sampler):

    def __init__(self, data_source: LandmarkDataset, images_per_landmark: int, samples_to_select: Optional[int] = None):
        self.data_source = data_source
        self.images_per_landmark = images_per_landmark
        self.samples_to_select = samples_to_select

    def __iter__(self):
        sample_ids_by_landmark_id = defaultdict(list)
        for sample_id, label in enumerate(self.data_source.labels):
            sample_ids_by_landmark_id[label].append(sample_id)
        for sample_ids in sample_ids_by_landmark_id.values():
            random.shuffle(sample_ids)
        landmark_ids = list(sample_ids_by_landmark_id.keys())
        samples_order = []
        while len(samples_order) < len(self):
            lmk_id = random.choice(landmark_ids)
            collected_in_batch = 0
            while collected_in_batch < self.images_per_landmark and sample_ids_by_landmark_id[lmk_id]:
                samples_order.append(sample_ids_by_landmark_id[lmk_id].pop())
                collected_in_batch += 1
                if len(samples_order) >= len(self):
                    break
            if not sample_ids_by_landmark_id[lmk_id]:
                del sample_ids_by_landmark_id[lmk_id]
        return iter(samples_order)

    def __len__(self):
        return self.samples_to_select or len(self.data_source)


def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train_epoch()
    loss_acc, proxy_loss_acc = 0.0, 0.0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        mask = torch.zeros(output.size(-1), dtype=torch.bool)
        sorted_class_indexes = []
        for class_index in target.unique():
            sorted_class_indexes.append(class_index.detach().cpu().numpy().item())
            mask[class_index] = True
        sorted_class_indexes = sorted(sorted_class_indexes)
        class_id_to_proxy_id = {class_id: proxy_id for proxy_id, class_id in enumerate(sorted_class_indexes)}
        proxy_class_indexes = [class_id_to_proxy_id[class_id] for class_id in target.detach().cpu().numpy()]

        proxy_target = torch.from_numpy(np.array(proxy_class_indexes, dtype=np.int64))
        proxy_target = proxy_target.to(device)
        target = target.to(device)
        proxy_output_logits = output[..., mask]
        proxy_output = F.log_softmax(proxy_output_logits)
        proxy_loss = F.nll_loss(proxy_output, proxy_target)
        proxy_loss.backward()
        optimizer.step()
        loss = F.nll_loss(F.log_softmax(output), target)
        proxy_loss_acc += proxy_loss.detach().cpu().numpy().item()
        loss_acc += loss.detach().cpu().numpy().item()
        if (batch_idx + 1) % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tProxy Loss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), proxy_loss_acc / cfg.log_interval, loss_acc / cfg.log_interval))
            proxy_loss_acc, loss_acc = 0.0, 0.0


def global_ap(results: List[Tuple[float, int, int]]):
    results = sorted(results, key=lambda x: x[0], reverse=True)
    rel = [pred == true for conf, pred, true in results]
    precision = np.cumsum(rel, dtype=np.float64) / np.array(range(1, len(rel) + 1), dtype=np.float64)
    ap = np.mean(precision * np.array(rel))
    return ap


def top_k_accuracy(top_10_results: List[Tuple[List[float], List[int], int]], k: int = 10):
    rel = [true in pred[:k] for conf, pred, true in top_10_results]
    return np.mean(rel)


def validation(model, device, test_loader):
    model.eval()
    correct = 0
    loss_acc, proxy_loss_acc = 0.0, 0.0
    results = []
    top_10_results = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader), desc="Validation"):
            data = data.to(device)
            output = model(data)

            mask = torch.zeros(output.size(-1), dtype=torch.bool)
            sorted_class_indexes = []
            for class_index in target.unique():
                sorted_class_indexes.append(class_index.detach().cpu().numpy().item())
                mask[class_index] = True
            sorted_class_indexes = sorted(sorted_class_indexes)
            class_id_to_proxy_id = {class_id: proxy_id for proxy_id, class_id in enumerate(sorted_class_indexes)}
            proxy_class_indexes = [class_id_to_proxy_id[class_id] for class_id in target.detach().cpu().numpy()]

            proxy_target = torch.from_numpy(np.array(proxy_class_indexes, dtype=np.int64))
            proxy_target = proxy_target.to(device)
            target = target.to(device)
            proxy_output_logits = output[..., mask]
            proxy_output = F.log_softmax(proxy_output_logits)

            proxy_loss_acc += F.nll_loss(proxy_output, proxy_target)
            loss_acc += F.nll_loss(F.log_softmax(output), target, reduction='sum').item()
            pred_class_position = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
            target_class_position = target.view_as(pred_class_position)
            correct += pred_class_position.eq(target_class_position).sum().item()

            pred_confidences = torch.gather(F.softmax(output), dim=1,
                                            index=pred_class_position.unsqueeze(-1)).squeeze(-1)
            results += [(c.item(), p.item(), t.item()) for c, p, t in zip(pred_confidences.detach().cpu().numpy(),
                                                                          pred_class_position.detach().cpu().numpy(),
                                                                          target_class_position.detach().cpu().numpy())]
            top_10_scores, top_10_argmax = torch.topk(F.softmax(output), dim=1, k=10)
            top_10_results += [(c.tolist(), p.tolist(), t.item()) for c, p, t in
                               zip(top_10_scores.detach().cpu().numpy(),
                                   top_10_argmax.detach().cpu().numpy(),
                                   target_class_position.detach().cpu().numpy())]

    loss_acc /= len(test_loader.dataset)
    proxy_loss_acc /= len(test_loader.dataset)
    ap = global_ap(results)
    top_10_accuracy = top_k_accuracy(top_10_results, k=10)
    top_5_accuracy = top_k_accuracy(top_10_results, k=5)
    msg = '\nTest set: \n' \
          'Average loss: {:.4f}, \n' \
          'Average proxy loss: {:.4f}, \n' \
          'Accuracy: {}/{} ({:.0f}%) \n' \
          'Accuracy top 5: ({:.0f}%) \n' \
          'Accuracy top 10: ({:.0f}%) \n' \
          'AP: {:.6f}%\n'.format(loss_acc,
                                 proxy_loss_acc,
                                 correct,
                                 len(test_loader.dataset),
                                 100. * correct / len(test_loader.dataset),
                                 100. * top_5_accuracy,
                                 100. * top_10_accuracy,
                                 ap)
    print(msg)


def get_data_loaders(train_dataset: LandmarkDataset, val_dataset: LandmarkDataset) -> (DataLoader, DataLoader):
    kwargs = {'batch_size': cfg.batch_size}
    if cfg.use_cuda:
        kwargs.update({'num_workers': cfg.num_workers,
                       'pin_memory': True},)

    train_sampler = LandmarkSampler(train_dataset, images_per_landmark=cfg.train_images_per_landmark)
    val_sampler = LandmarkSampler(val_dataset, images_per_landmark=cfg.train_images_per_landmark)

    def filter_collate(batch):
        batch = filter(lambda img: img is not None, batch)
        return torch.utils.data.dataloader.default_collate(list(batch))

    train_loader = DataLoader(train_dataset, drop_last=False, collate_fn=filter_collate,
                              sampler=train_sampler, **kwargs)
    val_loader = DataLoader(val_dataset, drop_last=False, collate_fn=filter_collate,
                            sampler=val_sampler, **kwargs)
    return train_loader, val_loader


def run_train(model: nn.Module):
    labels = read_train_labels(TRAIN_CSV)
    labels = filter_labels_by_lmk_ids(labels, all_landmark_ids)

    train_labels = [l for l in labels if Subset.from_id(l[0]) == Subset.train]
    val_labels = [l for l in labels if Subset.from_id(l[0]) == Subset.val]
    logging.debug(f"train samples: {len(train_labels)}")
    logging.debug(f"val samples: {len(val_labels)}")

    train_image_ids, train_labels = zip(*train_labels)
    val_image_ids, val_labels = zip(*val_labels)

    train_dataset = LandmarkDataset(TRAIN_DIR, all_landmark_ids=all_landmark_ids, image_ids=train_image_ids,
                                    labels=train_labels, transform=cfg.transform)
    val_dataset = LandmarkDataset(TRAIN_DIR, all_landmark_ids=all_landmark_ids, image_ids=val_image_ids,
                                  labels=val_labels, transform=cfg.transform)

    torch.manual_seed(cfg.seed)

    train_loader, test_loader = get_data_loaders(train_dataset, val_dataset)

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr,
                          momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler = StepLR(optimizer, **cfg.scheduler_kwargs)

    for epoch in range(1, cfg.epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch)
        validation(model, device, test_loader)
        scheduler.step()
        if cfg.save_model:
            torch.save(model.state_dict(), os.path.join(cfg.experiment_dir, f"model_epoch_{epoch}.pt"))
    if cfg.save_model:
        torch.save(model.state_dict(), "final.pt")


def make_predictions(model: nn.Module) -> List[Tuple[str, float, int]]:
    model.eval()
    image_paths = [x for x in pathlib.Path(TEST_DIR).rglob('*.jpg')]
    test_image_ids = [image_path.name.split('.')[0] for image_path in image_paths]

    test_dataset = LandmarkDataset(TEST_DIR, all_landmark_ids=all_landmark_ids, image_ids=test_image_ids, labels=None,
                                   transform=cfg.transform)
    test_loader = DataLoader(test_dataset, drop_last=False, batch_size=cfg.test_batch_size, num_workers=cfg.num_workers)

    predictions = []
    with torch.no_grad():
        for data, image_ids in tqdm(test_loader, total=len(test_loader), desc="Making Predictions"):
            data = data.to(device)
            output = model(data)

            pred_class_position = output.argmax(dim=1, keepdim=False)

            pred_confidences = torch.gather(F.softmax(output), dim=1,
                                            index=pred_class_position.unsqueeze(-1)).squeeze(-1)
            predictions += [(image_id, score.item(), all_landmark_ids[pred.item()])
                            for score, pred, image_id in
                            zip(pred_confidences.detach().cpu().numpy(),
                                pred_class_position.detach().cpu().numpy(),
                                image_ids)]

    return sorted(predictions, key=lambda x: x[1], reverse=True)


def main():
    model = cfg.model_factory(num_classes=len(all_landmark_ids)).to(device)
    if cfg.model_weights:
        model.load_state_dict(torch.load(cfg.model_weights, map_location=device))
    if False:
        run_train(model)
    predictions = make_predictions(model)
    save_submission_file(predictions)


if __name__ == '__main__':
    main()

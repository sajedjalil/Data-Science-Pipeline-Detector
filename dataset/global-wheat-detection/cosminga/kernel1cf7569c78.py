import csv
import os

import imageio
import torch
import torchvision.ops


class WheatDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_path,
                 encodings_path=None,
                 raw_bbox_path=None,
                 gen_encodings=False,
                 mean=None,
                 std=None):
        super(WheatDataset, self).__init__()
        self._images = sorted(os.listdir(img_path))
        self._len = len(self._images)
        self._img_path = img_path
        self._encodings_path = encodings_path
        if raw_bbox_path is not None:
            self._raw_bboxes = read_bboxes(raw_bbox_path)
        else:
            self._raw_bboxes = None
        self._mean = mean
        self._std = std
        self._gen_encodings = gen_encodings

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        img_file = self._images[key]
        img = torch.tensor(
            imageio.imread(os.path.join(f"{self._img_path}",
                                        f"{img_file}"))).permute(2, 0, 1)
        img = img.float() / 255
        data = (img, )
        if self._gen_encodings:
            bboxes = self._raw_bboxes.get(img_file.split('.')[0], [])
            encodings = encode_bboxes_and_scores(bboxes, self._mean, self._std)
            data = (*data, *encodings)
        else:
            pixel_class = torch.load(
                os.path.join(f"{self._encodings_path}",
                             f"{img_file}_pixel_class.pt"))
            box_class = torch.load(
                os.path.join(f"{self._encodings_path}",
                             f"{img_file}_box_class.pt"))
            box_sign_class = torch.load(
                os.path.join(f"{self._encodings_path}",
                             f"{img_file}_box_sign_class.pt"))
            offset_class = torch.load(
                os.path.join(f"{self._encodings_path}",
                             f"{img_file}_offset_class.pt"))
            offset_sign_class = torch.load(
                os.path.join(f"{self._encodings_path}",
                             f"{img_file}_offset_sign_class.pt"))
            data = (*data, pixel_class, box_class, box_sign_class,
                    offset_class, offset_sign_class)

        return data


def read_bboxes(path):
    with open(path, newline='') as bbox_file:
        bbox_reader = csv.reader(bbox_file)
        bboxes = {}
        next(bbox_reader)
        for id, _, _, bbox_str, _ in bbox_reader:
            bbox = [float(x) for x in bbox_str[1:-1].split(',')]
            bbox_list = bboxes.get(id, [])
            bboxes[id] = [*bbox_list, bbox]

    bboxes = {id: torch.tensor(bbox) for id, bbox in bboxes.items()}
    return bboxes


def get_wheat_loaders(img_path,
                      batch_size=1,
                      split=1,
                      num_workers=4,
                      encodings_path=None,
                      raw_bbox_path=None,
                      shuffle=False,
                      gen_encodings=False,
                      mean=None,
                      std=None):
    full_dataset = WheatDataset(img_path, encodings_path, raw_bbox_path,
                                gen_encodings, mean, std)

    train_size = int(len(full_dataset) * split)
    validate_size = len(full_dataset) - train_size

    indices = list(range(len(full_dataset)))
    if shuffle:
        random.shuffle(indices)

    train = torch.utils.data.Subset(full_dataset, indices[0:train_size])
    validate = torch.utils.data.Subset(full_dataset,
                                       indices[train_size:len(full_dataset)])

    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers)
    validate_loader = torch.utils.data.DataLoader(validate,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)

    return train_loader, validate_loader


def select_bboxes_nms(bboxes, scores, iou_theshold=0.95):
    bboxes_t = bboxes.clone().flatten(1, 2)
    bboxes_t[:, :, 2:4] = bboxes_t[:, :, 2:4] + bboxes_t[:, :, 0:2]
    scores = scores.clone().flatten(1, 3)
    keep = torchvision.ops.nms(bboxes_t[0], scores[0], iou_theshold)

    return keep


def box_intersection(bboxes, bbox):
    bboxes_xy = bboxes.clone()
    bboxes_xy[:, :, 2:4] = bboxes_xy[:, :, 2:4] + bboxes_xy[:, :, 0:2]
    bbox_xy = bbox.clone()
    bbox_xy[2:4] = bbox_xy[2:4] + bbox_xy[0:2]
    x_y_min = torch.max(bboxes_xy[:, :, 0:2], bbox_xy[0:2])
    x_y_max = torch.min(bboxes_xy[:, :, 2:4], bbox_xy[2:4])

    intersection = x_y_max > x_y_min
    intersection = torch.logical_and(intersection[:, :, 0], intersection[:, :,
                                                                         1])

    intersection = intersection * torch.prod(x_y_max - x_y_min, 2)

    return intersection


def box_iou(bboxes, bbox):
    intersection = box_intersection(bboxes, bbox)
    union = bboxes[:, :, 2] * bboxes[:, :, 3] + bbox[2] * bbox[3]
    union = union - intersection
    iou = intersection / union
    return iou


def select_bboxes2(bboxes, scores, iou_threshold, shape_threshold,
                   score_threshold):
    bboxes = bboxes.clone()
    scores = scores.clone()
    kept = scores >= score_threshold

    l2dist = lambda x, y: torch.sqrt(torch.sum((x - y)**2, 2))
    sdist = lambda x, y: 0.5 * torch.sum(torch.min(x, y) / torch.max(x, y), 2)
    for b in range(kept.shape[0]):
        for h in range(kept.shape[2]):
            for w in range(kept.shape[3]):
                if kept[b, 0, h, w]:
                    iou = box_iou(bboxes[b, :, :], bboxes[b, h, w])
                    shape_dist = sdist(bboxes[b, :, :, 0:2], bboxes[b, h, w,
                                                                    0:2])

                    #if torch.any(iou > 0):
                    #    print(iou)
                    same = iou > iou_threshold
                    #same = torch.logical_and(same,
                    #                         (shape_dist > shape_threshold))
                    same = torch.logical_and(same, kept[b, 0, :, :])

                    if same.any():
                        same = torch.logical_and(
                            same, (scores[b, 0, :, :] < scores[b, 0, h, w]))
                        if same.any():
                            kept[b, 0, :, :][same] = False

    return kept


def select_bboxes(bboxes, scores, center_threshold, shape_threshold,
                  score_threshold):
    bboxes = bboxes.clone()
    scores = scores.clone()
    kept = scores >= score_threshold

    l2dist = lambda x, y: torch.sqrt(torch.sum((x - y)**2, 2))
    for b in range(kept.shape[0]):
        for h in range(kept.shape[2]):
            for w in range(kept.shape[3]):
                if kept[b, 0, h, w]:
                    iou = box_iou(bboxes[b, :, :], bboxes[b, h, w])
                    if torch.any(iou > 0):
                        center_dist = l2dist(bboxes[b, :, :, 0:2],
                                             bboxes[b, h, w, 0:2])
                        shape_dist = l2dist(bboxes[b, :, :, 2:4],
                                            bboxes[b, h, w, 2:4])

                        same = center_dist < center_threshold
                        same = torch.logical_and(
                            same, (shape_dist < shape_threshold))
                        same = torch.logical_and(same, kept[b, 0, :, :])

                        if same.any():
                            same = torch.logical_and(
                                same,
                                (scores[b, 0, :, :] < scores[b, 0, h, w]))
                            if same.any():
                                kept[b, 0, :, :][same] = False

    return kept


def decode_bboxes(box_class,
                  box_sign_class,
                  offset_class,
                  offset_sign_class,
                  mean,
                  std,
                  num_offset_classes=9,
                  num_size_classes=10,
                  max_std=5.0):
    #_, _, height, width = box_class.shape
    height, width = (1024, 1024)
    box_class = box_class.clone().permute(0, 2, 3, 1).float()
    box_sign_class = box_sign_class.clone().permute(0, 2, 3, 1)

    offset_class = offset_class.clone().permute(0, 2, 3, 1).float()
    offset_sign_class = offset_sign_class.clone().permute(0, 2, 3, 1)

    mean = torch.tensor([mean[1], mean[0]])
    std = torch.tensor([std[1], std[0]])
    position = torch.cartesian_prod(torch.arange(height), torch.arange(width))
    position = position.reshape(height, width, 2)
    position = position[0:1024:2, 0:1024:2]

    box_shape = (box_class * max_std / num_size_classes) * std
    box_shape[box_sign_class == 0] = box_shape[box_sign_class == 0] * (-1)
    box_shape = box_shape + mean

    box_offset = (offset_class / num_offset_classes) * (box_shape * 0.5)
    box_offset[offset_sign_class ==
               0] = box_offset[offset_sign_class == 0] * (-1)
    box_center = position - box_offset
    box_xy_min = box_center - box_shape * 0.5

    bboxes = torch.cat((box_xy_min, box_shape), 3)

    #y_min, x_min, height, width
    return bboxes


def encode_bboxes_and_scores(bboxes,
                             mean,
                             std,
                             height=1024,
                             width=1024,
                             num_offset_classes=9,
                             num_size_classes=10,
                             max_std=5.0):
    mean = torch.tensor([mean[1], mean[0]])
    std = torch.tensor([std[1], std[0]])
    box_class = torch.zeros((height, width, 2), dtype=torch.int8)
    box_sign_class = torch.zeros((height, width, 2), dtype=torch.int8)
    offset_class = torch.zeros((height, width, 2), dtype=torch.int8)
    offset_sign_class = torch.zeros((height, width, 2), dtype=torch.int8)
    pixel_class = torch.zeros((height, width), dtype=torch.int8)

    ocupancy = torch.zeros(height, width) != 0
    dist = torch.zeros(height, width)

    for bbox in bboxes:
        x_min, y_min, box_width, box_height = bbox
        x_center = x_min + torch.floor(box_width * 0.5)
        y_center = y_min + torch.floor(box_height * 0.5)
        indices = torch.cartesian_prod(
            torch.arange(int(y_min),
                         int(y_min) + int(box_height)),
            torch.arange(int(x_min),
                         int(x_min) + int(box_width)))
        indices = indices.reshape((int(box_height), int(box_width), 2))

        offset = indices - torch.tensor([y_center, x_center])
        new_offset_class = offset / (torch.tensor(
            [box_height * 0.5, box_width * 0.5]))
        new_offset_class = new_offset_class * num_offset_classes
        new_offset_class = new_offset_class.int()
        new_offset_sign_class = new_offset_class >= 0
        new_offset_class = torch.abs(new_offset_class)
        new_offset_class = new_offset_class.int()
        new_offset_class[new_offset_class ==
                         num_offset_classes] = num_offset_classes - 1

        new_dist = torch.sqrt(offset[:, :, 0]**2 + offset[:, :, 1]**2)

        box_size = torch.tensor([box_height, box_width])
        box_size = (torch.tensor([box_height, box_width]) - mean) / std
        box_size = torch.clamp(box_size, -max_std, max_std)
        new_box_sign_class = box_size >= 0
        new_box_class = torch.abs(box_size) / (max_std / num_size_classes)
        new_box_class = new_box_class.int()
        new_box_class[new_box_class == num_size_classes] = num_size_classes - 1

        for y in range(int(box_height)):
            for x in range(int(box_width)):
                h, w = indices[y, x]
                if h < height and w < width:
                    if not ocupancy[h, w]:
                        box_class[h, w] = new_box_class
                        box_sign_class[h, w] = new_box_sign_class
                        offset_class[h, w] = new_offset_class[y, x]
                        offset_sign_class[h, w] = new_offset_sign_class[y, x]
                        pixel_class[h, w] = 1
                        dist[h, w] = new_dist[y, x]
                        ocupancy[h, w] = True
                    elif new_dist[y, x] < dist[h, w]:
                        box_class[h, w] = new_box_class
                        box_sign_class[h, w] = new_box_sign_class
                        offset_class[h, w] = new_offset_class[y, x]
                        offset_sign_class[h, w] = new_offset_sign_class[y, x]
                        dist[h, w] = new_dist[y, x]

    box_class = box_class.permute(2, 0, 1)
    box_sign_class = box_sign_class.permute(2, 0, 1)
    offset_class = offset_class.permute(2, 0, 1)
    offset_sign_class = offset_sign_class.permute(2, 0, 1)
    pixel_class = pixel_class.unsqueeze(0)

    return (pixel_class, box_class, box_sign_class, offset_class,
            offset_sign_class)


def bbox_stats(raw_bboxes):
    # box = x_min, y_min, width, height
    bboxes = torch.cat([bbox[:, 2:4] for _, bbox in raw_bboxes.items()], 0)
    mean = torch.mean(bboxes, 0)
    std = torch.std(bboxes, 0)

    return mean, std

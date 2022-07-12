import sys
import time

import ignite
import torchvision
import torch

import kernel1cf7569c78 as dataset
import kernel73ed4cc67f as wheat_detector


def evaluate(train_engine, test_engine, train_loader, test_loader):
    print(f"Evaluating\n-----------------------------------------")
    test_engine.run(test_loader)
    test_metrics = test_engine.state.metrics
    test_engine.run(train_loader)
    train_metrics = test_engine.state.metrics

    for name, value in train_metrics.items():
        print(f"{name} {value}/{test_metrics[name]}")


class WheatCriterion:
    def __init__(self, pixel_criterion, size_criterion, offset_criterion,
                 sign_criterion):
        self.pixel_criterion = pixel_criterion
        self.size_criterion = size_criterion
        self.offset_criterion = offset_criterion
        self.sign_criterion = sign_criterion

    def __call__(self, predicted, target):
        pixel_loss = self.pixel_criterion(predicted[0], target[0])
        box_w_loss = self.size_criterion(predicted[1], target[1])
        box_h_loss = self.size_criterion(predicted[2], target[2])
        box_w_sign_loss = self.sign_criterion(predicted[3], target[3])
        box_h_sign_loss = self.sign_criterion(predicted[4], target[4])

        offset_w_loss = self.size_criterion(predicted[5], target[5])
        offset_h_loss = self.size_criterion(predicted[6], target[6])
        offset_w_sign_loss = self.sign_criterion(predicted[7], target[7])
        offset_h_sign_loss = self.sign_criterion(predicted[8], target[8])

        loss = (pixel_loss + box_w_loss + box_h_loss + box_w_sign_loss +
                box_h_sign_loss + offset_w_loss + offset_h_loss +
                offset_w_sign_loss + offset_h_sign_loss)

        if torch.isnan(loss):
            raise Exception("NaN LOSS")

        return loss


def prepare_batch(batch, device, non_blocking):
    img, pixel_class, box_class, box_sign, offset_class, offset_sign = batch

    img = img.float().to(device)

    pixel_class = pixel_class[:, 0, :, :].long().to(device)

    box_height_class = box_class[:, 0, :, :].long().to(device)
    box_width_class = box_class[:, 1, :, :].long().to(device)
    box_height_sign_class = box_sign[:, 0, :, :].long().to(device)
    box_width_sign_class = box_sign[:, 1, :, :].long().to(device)

    box_h_offset_class = offset_class[:, 0, :, :].long().to(device)
    box_w_offset_class = offset_class[:, 1, :, :].long().to(device)
    box_h_offset_sign_class = offset_sign[:, 0, :, :].long().to(device)
    box_w_offset_sign_class = offset_sign[:, 1, :, :].long().to(device)

    return img, (pixel_class, box_width_class, box_height_class,
                 box_width_sign_class, box_height_sign_class,
                 box_w_offset_class, box_h_offset_class,
                 box_w_offset_sign_class, box_h_offset_sign_class)


def main():
    if True and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    raw_bboxes = dataset.read_bboxes("../input/global-wheat-detection/train.csv")
    mean, std = dataset.bbox_stats(raw_bboxes)

    train_loader, validate_loader = dataset.get_wheat_loaders(
        "../input/global-wheat-detection/train",
        batch_size=4,
        split=0.9,
        num_workers=4,
        encodings_path="../input/wheatclasses/train_classes_half_size",
        raw_bbox_path="../input/global-wheat-detection/train.csv",
        gen_encodings=False,
        mean=mean,
        std=std)

    pixel_criterion = torch.nn.CrossEntropyLoss()
    size_criterion = torch.nn.CrossEntropyLoss()
    offset_criterion = torch.nn.CrossEntropyLoss()
    sign_criterion = torch.nn.CrossEntropyLoss()

    wheat_criterion = WheatCriterion(pixel_criterion, size_criterion,
                                     offset_criterion, sign_criterion)

    wheat_net = wheat_detector.WheatDetector()
    wheat_net.load_state_dict(torch.load("../input/boxclasses295/wheat_detector_model_265_295.pth", map_location=device))

    optimizer = torch.optim.Adam(wheat_net.parameters(), lr=0.0001)

    trainer = ignite.engine.create_supervised_trainer(
        wheat_net,
        optimizer,
        wheat_criterion,
        device,
        prepare_batch=prepare_batch)

    metrics = {"loss": ignite.metrics.Loss(wheat_criterion, device=device)}
    evaluator = ignite.engine.create_supervised_evaluator(
        wheat_net, metrics, device, prepare_batch=prepare_batch)
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED(every=4),
        ignite.handlers.ModelCheckpoint('/kaggle/working/',
                                        'wheat_detector',
                                        n_saved=4,
                                        require_empty=False,
                                        create_dir=True), {'model': wheat_net})
    trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED(every=4),
                              evaluate, evaluator, train_loader,
                              validate_loader)

    trainer.run(train_loader, max_epochs=8)
    #evaluate(trainer, evaluator, train_loader, validate_loader)


if __name__ == "__main__":
    main()

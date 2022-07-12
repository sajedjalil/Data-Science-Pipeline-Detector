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
    def __init__(self, score_criterion, bbox_criterion):
        self.score_criterion = score_criterion
        self.bbox_criterion = bbox_criterion

    def __call__(self, predicted, target):
        score_loss = self.score_criterion(predicted[0], target[0])
        bbox_loss = self.bbox_criterion(predicted[1], target[1])
        loss = score_loss + bbox_loss

        return loss


def prepare_batch(batch, device, non_blocking):
    img, scores, bboxes = batch
    img = img.float()
    img = img.to(device)
    scores = scores.to(device)
    bboxes = bboxes.to(device)

    return img, (scores, bboxes)


def main():
    if True and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    raw_bboxes = dataset.read_bboxes("/kaggle/input/global-wheat-detection/train.csv")
    mean, std = dataset.bbox_stats(raw_bboxes)

    train_loader, validate_loader = dataset.get_wheat_loaders(
        "/kaggle/input/global-wheat-detection/train",
        batch_size=1,
        split=0.9,
        num_workers=4,
        seg_path=None,
        bbox_path=None,
        raw_bbox_path="/kaggle/input/global-wheat-detection/train.csv",
        gen_targets=True,
        mean=mean,
        std=std
    )

    score_criterion = torch.nn.BCELoss(reduction='mean')
    bbox_criterion = torch.nn.SmoothL1Loss(reduction='mean')
    wheat_criterion = WheatCriterion(score_criterion, bbox_criterion)

    wheat_net = wheat_detector.WheatDetector2()
    wheat_net.load_state_dict(torch.load("../input/wheatdetector/best2_4.pth"))
    optimizer = torch.optim.Adam(wheat_net.parameters(), lr=0.0000001)

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
        ignite.handlers.ModelCheckpoint('/kaggle/working',
                                        'wheat_detector_2',
                                        n_saved=4,
                                        require_empty=False,
                                        create_dir=True), {'model': wheat_net})
    trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED(every=4),
                              evaluate, evaluator, train_loader,
                              validate_loader)

    trainer.run(train_loader, max_epochs=1 * 4)


if __name__ == "__main__":
    main()

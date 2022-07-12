import sys
import time

import copy

import ignite
import torchvision
import torch

import kernel3a2ac1615a as dataset
import kernel447322982f as wheat_detector


def evaluate(train_engine, test_engine, train_loader, test_loader):
    print(f"Evaluating\n-----------------------------------------")
    test_engine.run(test_loader)
    test_metrics = copy.deepcopy(test_engine.state.metrics)
    test_engine.run(train_loader)
    train_metrics = copy.deepcopy(test_engine.state.metrics)

    for name, value in train_metrics.items():
        print(f"{name} {value}/{test_metrics[name]}")

class WheatCriterion:
    def __init__(self, score_criterion, bbox_criterion):
        self.score_criterion = score_criterion
        self.bbox_criterion = bbox_criterion

    def __call__(self, predicted, target):
        score_loss = self.score_criterion(predicted[0], target[0])
        bbox_loss = self.bbox_criterion(predicted[1], target[1])
        #print(f"score_loss = {score_loss.item()} | bbox_loss = {bbox_loss.item()}")
        loss = score_loss + bbox_loss

        return loss


def prepare_batch(batch, device, non_blocking):
    img, scores, bboxes = batch
    img = img.float()
    img = img.to(device)
    
    scores = scores.to(device)
    bboxes = bboxes.to(device)
    mask = scores > 0.9
    scores = mask * scores
    bboxes = mask * bboxes

    return img, (scores, bboxes)


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
        batch_size=1,
        split=0.9,
        num_workers=4,
        seg_path="../input/encodedwheat/train_set_qsize/score",
        bbox_path="../input/encodedwheat/train_set_qsize/bbox",
        shuffle=False
        #raw_bbox_path="./data/train.csv",
        #gen_targets=False,
        #mean=mean,
        #std=std 
    )

    score_criterion = torch.nn.BCELoss(reduction='mean')
    bbox_criterion = torch.nn.SmoothL1Loss(reduction='mean')
    wheat_criterion = WheatCriterion(score_criterion, bbox_criterion)

    wheat_net = wheat_detector.WheatDetector()
    wheat_net.load_state_dict(torch.load("../input/bestmodel4/wheat_detector_model_64659.pt",
        map_location=device))
    wheat_net = wheat_net.to(device)
    
    optimizer = torch.optim.Adam(wheat_net.parameters(), lr=0.00001)

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
        ignite.engine.Events.EPOCH_COMPLETED(every=1),
        ignite.handlers.ModelCheckpoint('/kaggle/working',
                                        'wheat_detector',
                                        n_saved=5,
                                        require_empty=False,
                                        create_dir=True), {'model': wheat_net})
    trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED(every=5),
                              evaluate, evaluator, train_loader,
                              validate_loader)

    evaluate(trainer, evaluator, train_loader, validate_loader)
    trainer.run(train_loader, max_epochs=5*3)

    
if __name__ == "__main__":
    main()

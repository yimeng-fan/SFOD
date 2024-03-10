from os.path import join
import sys
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn.functional as F

from datasets.gen1_od_dataset import GEN1DetectionDataset
from object_detection_module import DetectionLitModule

from numpy import random

import copy


def horizontal_flip_boxes(boxes, width):
    """
    Performs a horizontal flip of the bounding box

    :param boxes: Bounding box with shape [number_boxes, 4] ([xmin, ymin, xmax, ymax])
    :param width: Width of the image
    :return: Bounding box after flipping
    """
    boxes_flipped = boxes.clone()
    boxes_flipped[:, 0] = width - boxes[:, 2]
    boxes_flipped[:, 2] = width - boxes[:, 0]
    return boxes_flipped


def augmentation_collate_fn(batch):
    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)
    batch_size, num_steps, channels, height, width = samples.size()
    targets = [item[1] for item in batch]

    augmented_samples = []
    augmented_targets = []

    PATIENCE = 2
    patience = PATIENCE
    i = 0
    while i < batch_size:
        transform_t = torch.eye(3)  # 3x3unit matrix
        flip = False
        if random.random() > 0.5:
            flip = True
            transform_t[0, 0] *= -1

        transform_t_single = transform_t[:2, :].unsqueeze(0).repeat(num_steps, 1, 1).to(torch.float32)
        affine_t = F.affine_grid(transform_t_single.view(-1, 2, 3), [num_steps, channels, height, width],
                                 align_corners=False)

        sample_augmented = F.grid_sample(samples[i], affine_t, padding_mode='border', align_corners=False)
        augmented_samples.append(sample_augmented)

        real_boxes = targets[i]['boxes'].clone()
        if flip:
            targets[i]['boxes'] = horizontal_flip_boxes(targets[i]['boxes'], width)

        augmented_targets.append(copy.deepcopy(targets[i]))
        targets[i]['boxes'] = real_boxes

        if targets[i]['labels'].sum() > 0:
            patience -= 1
            if patience != 0:
                i -= 1
            else:
                patience = PATIENCE

        i += 1

    assert len(augmented_samples) == len(augmented_targets), "length is wrong"
    augmented_samples = torch.stack(augmented_samples, 0)  # Stack them together
    return [augmented_samples, augmented_targets]


def collate_fn(batch):
    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)

    targets = [item[1] for item in batch]
    return [samples, targets]


def main():
    parser = argparse.ArgumentParser(description='Classify event dataset')
    # Dataset
    parser.add_argument('-dataset', default='gen1', type=str, help='dataset used {GEN1}')
    parser.add_argument('-path', default='PropheseeGEN1/detection_dataset_duration_60s_ratio_1.0', type=str,
                        help='path to dataset location')
    parser.add_argument('-num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('-devices', default='auto', type=str, help='number of devices')

    # Data
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
    parser.add_argument('-tbin', default=2, type=int, help='number of micro time bins')
    parser.add_argument('-image_shape', default=(240, 304), type=tuple, help='spatial resolution of events')

    # Training
    parser.add_argument('-epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate used')
    parser.add_argument('-wd', default=1e-4, type=float, help='weight decay used')
    parser.add_argument('-limit_train_batches', default=1., type=float, help='train batches limit')
    parser.add_argument('-limit_val_batches', default=1., type=float, help='val batches limit')
    parser.add_argument('-limit_test_batches', default=1., type=float, help='test batches limit')
    parser.add_argument('-num_workers', default=4, type=int, help='number of workers for dataloaders')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-test', action='store_true', help='whether to test the model')
    parser.add_argument('-precision', default='16-mixed', type=str, help='whether to use AMP {16, 32, 64}')
    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    parser.add_argument('-early_stopping', action='store_true', help='early stopping')

    # Backbone
    parser.add_argument('-backbone', default='densenet-121_24', type=str,
                        help='model used {densenet-v}', dest='model')
    parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
    parser.add_argument('-pretrained_backbone', default=None, type=str, help='path to pretrained backbone model')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-extras', type=int, default=[640, 320, 320], nargs=3,
                        help='number of channels for extra layers after the backbone')
    parser.add_argument('-fusion', action='store_true', help='if to fusion the features')

    # Neck
    parser.add_argument('-fusion_layers', default=4, type=int, help='number of fusion layers')
    parser.add_argument('-mode', type=str, default='norm', help='The mode of detection_pyramid')

    # Priors
    parser.add_argument('-min_ratio', default=0.05, type=float, help='min ratio for priors\' box generation')
    parser.add_argument('-max_ratio', default=0.80, type=float, help='max ratio for priors\' box generation')
    parser.add_argument('-aspect_ratios', default=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], type=int,
                        help='aspect ratios for priors\' box generation')

    # Loss parameters
    parser.add_argument('-box_coder_weights', default=[10.0, 10.0, 5.0, 5.0], type=float, nargs=4,
                        help='weights for the BoxCoder class')
    parser.add_argument('-iou_threshold', default=0.50, type=float,
                        help='intersection over union threshold for the SSDMatcher class')
    parser.add_argument('-score_thresh', default=0.01, type=float,
                        help='score threshold used for postprocessing the detections')
    parser.add_argument('-nms_thresh', default=0.45, type=float,
                        help='NMS threshold used for postprocessing the detections')
    parser.add_argument('-topk_candidates', default=200, type=int, help='number of best detections to keep before NMS')
    parser.add_argument('-detections_per_img', default=100, type=int,
                        help='number of best detections to keep after NMS')

    args = parser.parse_args()
    print(args)

    torch.set_float32_matmul_precision('medium')

    if args.dataset == "gen1":
        dataset = GEN1DetectionDataset
        torch.multiprocessing.set_sharing_strategy('file_system')
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

    module = DetectionLitModule(args)

    # LOAD PRETRAINED MODEL
    if args.pretrained is not None:
        ckpt_path = join(f"ckpt-od-{args.dataset}-{args.model}-val", args.pretrained)
        module = DetectionLitModule.load_from_checkpoint(ckpt_path, strict=False)

    callbacks = []
    if args.save_ckpt:
        ckpt_callback_val = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f"ckpt-od-{args.dataset}-{args.model}-val/",
            filename=f"{args.dataset}" + "-{epoch:02d}-{val_loss:.4f}",
            save_top_k=5,
            mode='min',
        )
        ckpt_callback_train = ModelCheckpoint(
            monitor='train_loss',
            dirpath=f"ckpt-od-{args.dataset}-{args.model}-train/",
            filename=f"{args.dataset}" + "-{epoch:02d}-{train_loss:.4f}",
            save_top_k=3,
            mode='min',
        )
        callbacks.append(ckpt_callback_val)
        callbacks.append(ckpt_callback_train)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
        )
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        devices=args.devices, accelerator="gpu",
        gradient_clip_val=1., max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        check_val_every_n_epoch=1,
        deterministic=False,
        precision=args.precision,
        callbacks=callbacks,
        strategy='ddp_find_unused_parameters_true',
    )

    if args.train:
        train_dataset = dataset(args, mode="train")
        val_dataset = dataset(args, mode="val")

        train_dataloader = DataLoader(train_dataset, batch_size=args.b, collate_fn=augmentation_collate_fn,
                                      num_workers=args.num_workers, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.b, collate_fn=collate_fn, num_workers=args.num_workers)

        trainer.fit(module, train_dataloader, val_dataloader)
    if args.test:
        test_dataset = dataset(args, mode="test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.b, collate_fn=collate_fn,
                                     num_workers=args.num_workers)

        trainer.test(module, test_dataloader)


if __name__ == '__main__':
    main()

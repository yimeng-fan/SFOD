from os.path import join
import sys
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datasets.classification_datasets import NCARSClassificationDataset
from models.utils import get_model
from classification_module import ClassificationLitModule


def main():
    parser = argparse.ArgumentParser(description='Classify event dataset')
    parser.add_argument('-devices', default='1', type=str, help='number of devices')
    parser.add_argument('-precision', default='16-mixed', type=str, help='whether to use AMP {16, 32, 64}')
    parser.add_argument('-num_workers', default=2, type=int, help='The number of workers')

    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
    parser.add_argument('-tbin', default=2, type=int, help='number of micro time bins')
    parser.add_argument('-image_shape', default=(240, 304), type=tuple, help='spatial resolution of events')

    parser.add_argument('-dataset', default='ncars', type=str, help='dataset used NCAR')
    parser.add_argument('-path', default='PropheseeNCARS', type=str, help='dataset used. NCAR')

    parser.add_argument('-model', default='densenet-121_24', type=str,
                        help='model used {densenet-v}')
    parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-lr', default=5e-3, type=float, help='learning rate used')
    parser.add_argument('-epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-test', action='store_true', help='whether to test the model')

    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    parser.add_argument('-early_stopping', action='store_true', help='early stopping')

    parser.add_argument('-loss_fun', default='mse', type=str, help='loss function used {mse, mae, cross}')
    parser.add_argument('-encoding', default='fre', type=str, help='encoding method used {fre, num}')

    args = parser.parse_args()
    print(args)

    torch.set_float32_matmul_precision('medium')

    if args.dataset == "ncars":
        dataset = NCARSClassificationDataset
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

    train_dataset = dataset(args, mode="train")
    test_dataset = dataset(args, mode="test")

    train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=args.num_workers, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.b, num_workers=args.num_workers)

    model = get_model(args)
    module = ClassificationLitModule(model, args.T, epochs=args.epochs, lr=args.lr, loss_fun=args.loss_fun,
                                     encoding=args.encoding)

    # LOAD PRETRAINED MODEL
    if args.pretrained is not None:
        ckpt_path = join(f"ckpt-{args.dataset}-{args.model}", args.pretrained)
        module = ClassificationLitModule.load_from_checkpoint(ckpt_path, strict=False, model=model)

    callbacks = []
    if args.save_ckpt:
        ckpt_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=f"ckpt-{args.dataset}-{args.model}/",
            filename=f"{args.dataset}" + "-{epoch:02d}-{val_acc:.4f}",
            save_top_k=3,
            mode='max',
        )
        callbacks.append(ckpt_callback)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
        )
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        devices=args.devices, accelerator="gpu",
        gradient_clip_val=1., max_epochs=args.epochs,
        limit_train_batches=1., limit_val_batches=1.,
        limit_test_batches=1., check_val_every_n_epoch=1,
        deterministic=False,
        precision=args.precision,
        callbacks=callbacks,
    )

    if args.train:
        trainer.fit(module, train_dataloader, test_dataloader)
    if args.test:
        trainer.test(module, test_dataloader)


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics

from spikingjelly.clock_driven import functional


class ClassificationLitModule(pl.LightningModule):

    def __init__(self, model, T, epochs=10, lr=5e-3, num_classes=2, loss_fun='mse', encoding='fre'):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.lr, self.epochs = lr, epochs
        self.num_classes = num_classes
        self.loss_fun, self.encoding = loss_fun, encoding
        self.T = T
        self.all_nnz, self.all_nnumel = 0, 0

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.train_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none", task='multiclass')
        self.val_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none", task='multiclass')
        self.test_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none", task='multiclass')
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='multiclass')
        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='multiclass')
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='multiclass')

        self.model = model
        self.model.add_hooks()

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, C, H, W]
        output = self.model(x).permute(1, 0, 2)  # [N, T, C]
        return output.sum(dim=1)  # sum in the T

    def step(self, batch, batch_idx, mode):
        events, target = batch

        outputs = self(events)

        # 频率
        if self.encoding == 'fre':
            outputs = outputs / self.T

        # MSE
        if self.loss_fun == 'mse':
            loss = nn.functional.mse_loss(outputs,
                                          nn.functional.one_hot(target, self.num_classes).float().cuda())
        # MAE
        elif self.loss_fun == 'mae':
            loss = nn.functional.l1_loss(outputs,
                                         nn.functional.one_hot(target, self.num_classes).float().cuda())
        # 交叉熵
        elif self.loss_fun == 'cross':
            loss = nn.functional.cross_entropy(outputs, target)

        # Metrics computation
        sm_outputs = outputs.softmax(dim=-1)

        # Measure sparsity if not training
        if mode != 'train':
            self.process_nz(self.model.get_nz_numel())
            self.model.reset_nz_numel()

        acc, acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        confmat = getattr(self, f'{mode}_confmat')

        acc(sm_outputs, target)
        acc_by_class(sm_outputs, target)
        confmat(sm_outputs, target)

        if mode != "test":
            self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=(mode == "train"))

        functional.reset_net(self.model)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="test")

    def on_mode_epoch_end(self, mode):
        print()

        mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()
        for i, acc_i in enumerate(acc_by_class):
            self.log(f'{mode}_acc_{["Background", "Cars"][i]}', acc_i, sync_dist=True)
        self.log(f'{mode}_acc', acc, sync_dist=True)

        print(f"{mode} accuracy: {100 * acc:.2f}%")
        print(f"Background {100 * acc_by_class[0]:.2f}% - Cars {100 * acc_by_class[1]:.2f}%")
        mode_acc.reset()
        mode_acc_by_class.reset()

        print(f"{mode} confusion matrix:")
        self_confmat = getattr(self, f"{mode}_confmat")
        confmat = self_confmat.compute()
        print(confmat)
        self_confmat.reset()

        # reset the nz and numel for train
        if mode == 'train':
            self.model.reset_nz_numel()
        else:
            print(
                f"{mode}: Total sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / (self.all_nnumel + 1e-3):.2f}%)")
            self.all_nnz, self.all_nnumel = 0, 0

    def on_train_epoch_end(self):
        self.on_mode_epoch_end(mode="train")

    def on_validation_epoch_end(self):
        self.on_mode_epoch_end(mode="val")

    def on_test_epoch_end(self):
        self.on_mode_epoch_end(mode="test")

    def process_nz(self, nz_numel):
        nz, numel = nz_numel
        total_nnz, total_nnumel = 0, 0

        for module, nnz in nz.items():
            if "act" in module:
                nnumel = numel[module]
                if nnumel != 0:
                    total_nnz += nnz
                    total_nnumel += nnumel
        if total_nnumel != 0:
            self.all_nnz += total_nnz
            self.all_nnumel += total_nnumel

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

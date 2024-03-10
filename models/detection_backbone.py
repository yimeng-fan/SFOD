import torch
from torch import nn

from models.utils import SpikingBlock, get_model
from models.SSD_utils import init_weights


class DetectionBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.nz, self.numel = {}, {}

        self.model = get_model(args)
        self.fusion = args.fusion
        self.fusion_layers = args.fusion_layers

        if args.pretrained_backbone is not None:
            ckpt = torch.load(args.pretrained_backbone)
            state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
            self.model.load_state_dict(state_dict, strict=False)

        self.out_channels = self.model.out_channels
        extras_fm = args.extras

        self.extras = None
        if self.fusion and self.fusion_layers == 4:
            self.extras = nn.ModuleList(
                [
                    nn.Sequential(
                        SpikingBlock(self.out_channels[-1], self.out_channels[-1] // 2, kernel_size=1),
                        SpikingBlock(self.out_channels[-1] // 2, extras_fm[0], kernel_size=3, padding=1, stride=2),
                    ),
                ]
            )
            self.out_channels.extend([extras_fm[0]])
        elif not self.fusion:
            self.extras = nn.ModuleList(
                [
                    nn.Sequential(
                        SpikingBlock(self.out_channels[-1], self.out_channels[-1] // 2, kernel_size=1),
                        SpikingBlock(self.out_channels[-1] // 2, extras_fm[0], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(extras_fm[0], extras_fm[0] // 4, kernel_size=1),
                        SpikingBlock(extras_fm[0] // 4, extras_fm[1], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(extras_fm[1], extras_fm[1] // 2, kernel_size=1),
                        SpikingBlock(extras_fm[1] // 2, extras_fm[2], kernel_size=3, padding=1, stride=2),
                    ),
                ]
            )
            self.out_channels.extend(extras_fm)

        if self.extras is not None:
            self.extras.apply(init_weights)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # x.shape = [T, N, C, H, W]
        feature_maps = self.model(x, classify=False)
        x = feature_maps[-1]
        T = x.shape[0]
        if self.fusion:
            detection_feed = [fm.permute(1, 0, 2, 3, 4) for fm in feature_maps]  # [N, T, C, H, W]
        else:
            detection_feed = [fm.sum(0) / T for fm in feature_maps]  # [N, C, H, W]

        if self.extras is not None:
            for block in self.extras:
                x = block(x)
                if self.fusion:
                    detection_feed.append(x.permute(1, 0, 2, 3, 4))  # [N, T, C, H, W]
                else:
                    detection_feed.append(x.sum(0) / T)  # [N, C, H, W]

        return detection_feed

    def add_hooks(self):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()

            return hook

        self.hooks = {}
        for name, module in self.model.features.named_modules():
            self.nz[name], self.numel[name] = 0, 0
            self.hooks[name] = module.register_forward_hook(get_nz(name))

        if self.extras is not None:
            for name, module in self.extras.named_modules():
                self.nz[name], self.numel[name] = 0, 0
                self.hooks[name] = module.register_forward_hook(get_nz(name))

    def reset_nz_numel(self):
        for name, module in self.model.features.named_modules():
            self.nz[name], self.numel[name] = 0, 0

        if self.extras is not None:
            for name, module in self.extras.named_modules():
                self.nz[name], self.numel[name] = 0, 0

    def get_nz_numel(self):
        return self.nz, self.numel

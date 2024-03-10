from torch import nn
import torch

from models.utils import SpikingNeckBlock
from models.SSD_utils import init_weights


class DetectionNeck(nn.Module):
    def __init__(self, in_channels, fusion_layers=4):
        super().__init__()

        self.nz, self.numel = {}, {}

        # Define the upsampling module
        self.ft_module = nn.ModuleList(
            [
                SpikingNeckBlock(in_channels[0], 128, up_flag=False),
                SpikingNeckBlock(in_channels[1], 128, kernel_size=4, stride=2, padding=1),
                SpikingNeckBlock(in_channels[2], 128, kernel_size=8, stride=4, padding=1),
            ]
        )
        if fusion_layers == 4:
            self.ft_module.append(SpikingNeckBlock(in_channels[3], 128, kernel_size=(11, 12), stride=7, padding=1))

        self.ft_module.apply(init_weights)
        self.out_channel = 128 * fusion_layers

    def forward(self, source_features):
        assert len(source_features) == len(self.ft_module)
        transformed_features = []

        # upsample
        for k, v in enumerate(self.ft_module):
            transformed_features.append(v(source_features[k].permute(1, 0, 2, 3, 4)))  # Before inputting, change to T N C H W

        concat_fea = torch.cat(transformed_features, 2)

        return concat_fea.permute(1, 0, 2, 3, 4)  # N T C H W

    def add_hooks(self):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()

            return hook

        self.hooks = {}
        for name, module in self.ft_module.named_modules():
            self.nz[name], self.numel[name] = 0, 0
            self.hooks[name] = module.register_forward_hook(get_nz(name))

    def reset_nz_numel(self):
        for name, module in self.ft_module.named_modules():
            self.nz[name], self.numel[name] = 0, 0

    def get_nz_numel(self):
        return self.nz, self.numel

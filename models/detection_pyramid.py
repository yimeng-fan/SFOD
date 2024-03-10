from torch import nn
import torch
from spikingjelly.clock_driven import layer, neuron

from models.utils import SpikingBlock, SEWBlock, SEWSkipBlock
from models.SSD_utils import init_weights
from models.spiking_densenet import _DenseBlock


class DetectionPyramid(nn.Module):
    def __init__(self, in_channel, T, mode='norm', num_layers=2, growth_rate=24):
        """
            mode：norm：basic version
                 dense：Spiking Dense Block-enhanced version
                 res：SEW Res Block-enhanced version
        """
        super().__init__()

        self.nz, self.numel = {}, {}
        self.out_channels = [512, 512, 256, 256, 256, 256]

        # Define the upsampling module
        if mode == 'norm':
            self.pyramid_ext = nn.ModuleList(
                [
                    nn.Sequential(
                        SpikingBlock(in_channel, in_channel // 2, kernel_size=1),
                        SpikingBlock(in_channel // 2, self.out_channels[0], kernel_size=3, padding=1, stride=1),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[0], self.out_channels[0] // 2, kernel_size=1),
                        SpikingBlock(self.out_channels[0] // 2, self.out_channels[1], kernel_size=3, padding=1,
                                     stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[1], self.out_channels[1] // 2, kernel_size=1),
                        SpikingBlock(self.out_channels[1] // 2, self.out_channels[2], kernel_size=3, padding=1,
                                     stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[2], self.out_channels[2] // 2, kernel_size=1),
                        SpikingBlock(self.out_channels[2] // 2, self.out_channels[3], kernel_size=3, padding=1,
                                     stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[3], self.out_channels[3] // 2, kernel_size=1),
                        SpikingBlock(self.out_channels[3] // 2, self.out_channels[4], kernel_size=3, padding=1,
                                     stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[4], self.out_channels[4] // 2, kernel_size=1),
                        SpikingBlock(self.out_channels[4] // 2, self.out_channels[5], kernel_size=3, padding=1,
                                     stride=2)
                    )
                ]
            )
        elif mode == 'dense':
            self.pyramid_ext = nn.ModuleList(
                [
                    nn.Sequential(
                        SpikingBlock(in_channel, in_channel // 2, kernel_size=1),
                        layer.SeqToANNContainer(_DenseBlock(
                            num_layers=num_layers,
                            num_input_features=in_channel // 2,
                            bn_size=4,
                            growth_rate=growth_rate,
                            drop_rate=0,
                            norm_layer=nn.BatchNorm2d,
                            bias=False,
                            T=T,
                            neuron=neuron.MultiStepParametricLIFNode,
                            backend="cupy",
                        )),
                        SpikingBlock(in_channel // 2 + num_layers * growth_rate, self.out_channels[0], kernel_size=3,
                                     padding=1, stride=1),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[0], self.out_channels[0] // 2, kernel_size=1),
                        layer.SeqToANNContainer(_DenseBlock(
                            num_layers=num_layers,
                            num_input_features=self.out_channels[0] // 2,
                            bn_size=4,
                            growth_rate=growth_rate,
                            drop_rate=0,
                            norm_layer=nn.BatchNorm2d,
                            bias=False,
                            T=T,
                            neuron=neuron.MultiStepParametricLIFNode,
                            backend="cupy",
                        )),
                        SpikingBlock(self.out_channels[0] // 2 + num_layers * growth_rate, self.out_channels[1],
                                     kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[1], self.out_channels[1] // 2, kernel_size=1),
                        layer.SeqToANNContainer(_DenseBlock(
                            num_layers=num_layers,
                            num_input_features=self.out_channels[1] // 2,
                            bn_size=4,
                            growth_rate=growth_rate,
                            drop_rate=0,
                            norm_layer=nn.BatchNorm2d,
                            bias=False,
                            T=T,
                            neuron=neuron.MultiStepParametricLIFNode,
                            backend="cupy",
                        )),
                        SpikingBlock(self.out_channels[1] // 2 + num_layers * growth_rate, self.out_channels[2],
                                     kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[2], self.out_channels[2] // 2, kernel_size=1),
                        layer.SeqToANNContainer(_DenseBlock(
                            num_layers=num_layers,
                            num_input_features=self.out_channels[2] // 2,
                            bn_size=4,
                            growth_rate=growth_rate,
                            drop_rate=0,
                            norm_layer=nn.BatchNorm2d,
                            bias=False,
                            T=T,
                            neuron=neuron.MultiStepParametricLIFNode,
                            backend="cupy",
                        )),
                        SpikingBlock(self.out_channels[2] // 2 + num_layers * growth_rate, self.out_channels[3],
                                     kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[3], self.out_channels[3] // 2, kernel_size=1),
                        layer.SeqToANNContainer(_DenseBlock(
                            num_layers=num_layers,
                            num_input_features=self.out_channels[3] // 2,
                            bn_size=4,
                            growth_rate=growth_rate,
                            drop_rate=0,
                            norm_layer=nn.BatchNorm2d,
                            bias=False,
                            T=T,
                            neuron=neuron.MultiStepParametricLIFNode,
                            backend="cupy",
                        )),
                        SpikingBlock(self.out_channels[3] // 2 + num_layers * growth_rate, self.out_channels[4],
                                     kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[4], self.out_channels[4] // 2, kernel_size=1),
                        layer.SeqToANNContainer(_DenseBlock(
                            num_layers=num_layers,
                            num_input_features=self.out_channels[4] // 2,
                            bn_size=4,
                            growth_rate=growth_rate,
                            drop_rate=0,
                            norm_layer=nn.BatchNorm2d,
                            bias=False,
                            T=T,
                            neuron=neuron.MultiStepParametricLIFNode,
                            backend="cupy",
                        )),
                        SpikingBlock(self.out_channels[4] // 2 + num_layers * growth_rate, self.out_channels[5],
                                     kernel_size=3, padding=1, stride=2)
                    )
                ]
            )
        elif mode == 'res':
            self.pyramid_ext = nn.ModuleList(
                [
                    nn.Sequential(
                        SpikingBlock(in_channel, in_channel // 2, kernel_size=1),
                        SEWBlock(in_channel // 2, in_channel // 4),
                        SpikingBlock(in_channel // 2, self.out_channels[0], kernel_size=3, padding=1, stride=1),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[0], self.out_channels[0] // 2, kernel_size=1),
                        SEWBlock(self.out_channels[0] // 2, self.out_channels[0] // 4),
                        SpikingBlock(self.out_channels[0] // 2, self.out_channels[1], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[1], self.out_channels[1] // 2, kernel_size=1),
                        SEWBlock(self.out_channels[1] // 2, self.out_channels[1] // 4),
                        SpikingBlock(self.out_channels[1] // 2, self.out_channels[2], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[2], self.out_channels[2] // 2, kernel_size=1),
                        SEWBlock(self.out_channels[2] // 2, self.out_channels[2] // 4),
                        SpikingBlock(self.out_channels[2] // 2, self.out_channels[3], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[3], self.out_channels[3] // 2, kernel_size=1),
                        SEWBlock(self.out_channels[3] // 2, self.out_channels[3] // 4),
                        SpikingBlock(self.out_channels[3] // 2, self.out_channels[4], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SpikingBlock(self.out_channels[4], self.out_channels[4] // 2, kernel_size=1),
                        SEWBlock(self.out_channels[4] // 2, self.out_channels[4] // 4),
                        SpikingBlock(self.out_channels[4] // 2, self.out_channels[5], kernel_size=3, padding=1, stride=2)
                    )
                ]
            )
        elif mode == 'skip':
            self.pyramid_ext = nn.ModuleList(
                [
                    SEWSkipBlock(in_channel, self.out_channels[0]),
                    SEWSkipBlock(self.out_channels[0], self.out_channels[1]),
                    SEWSkipBlock(self.out_channels[1], self.out_channels[2]),
                    SEWSkipBlock(self.out_channels[2], self.out_channels[3]),
                    SEWSkipBlock(self.out_channels[3], self.out_channels[4]),
                    SEWSkipBlock(self.out_channels[4], self.out_channels[5]),
                ]
            )


        self.pyramid_ext.apply(init_weights)

    def forward(self, x):
        pyramid_fea = []
        x = x.permute(1, 0, 2, 3, 4)  # Before inputting, change to T N C H W
        T = x.shape[0]
        for k, v in enumerate(self.pyramid_ext):
            x = v(x)
            pyramid_fea.append(x.permute(1, 0, 2, 3, 4).sum(dim=1) / T)

        return pyramid_fea

    def add_hooks(self):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()

            return hook

        self.hooks = {}
        for name, module in self.pyramid_ext.named_modules():
            self.nz[name], self.numel[name] = 0, 0
            self.hooks[name] = module.register_forward_hook(get_nz(name))

    def reset_nz_numel(self):
        for name, module in self.pyramid_ext.named_modules():
            self.nz[name], self.numel[name] = 0, 0

    def get_nz_numel(self):
        return self.nz, self.numel

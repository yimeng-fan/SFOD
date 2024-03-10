import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from spikingjelly.clock_driven import layer, neuron, surrogate
from .spiking_densenet import *

class SEWSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, connect_f='ADD'):
        super(SEWSkipBlock, self).__init__()
        self.connect_f = connect_f

        self.conv1 = nn.Sequential(
            SpikingBlock(in_channels, in_channels // 2, kernel_size=1),
            SpikingBlock(in_channels // 2, out_channels, kernel_size=3, padding=1, stride=2),
        )

        self.conv2 = SpikingBlock(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x: torch.Tensor):
        out1 = self.conv1(x)
        out2 = self.conv2(x)

        if self.connect_f == 'ADD':
            out = out1 + out2
        elif self.connect_f == 'AND':
            out = out1 * out2
        elif self.connect_f == 'IAND':
            out = out2 * (1. - out1)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f='ADD'):
        super(SEWBlock, self).__init__()
        self.connect_f = connect_f
        self.conv = nn.Sequential(
            SpikingBlock(in_channels, mid_channels, kernel_size=3, padding=1, stride=1),
            SpikingBlock(mid_channels, in_channels, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.connect_f == 'ADD':
            out = out + x
        elif self.connect_f == 'AND':
            out = out * x
        elif self.connect_f == 'IAND':
            out = x * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class ConstantDilation2d(nn.Module):
    """
        Performs fill operations (calculations can only be performed on cuda devices)
    """

    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation

    def forward(self, features):
        # Getting the shape
        H, W = features.shape[-2:]
        new_h = H + (H - 1) * (self.dilation - 1)
        new_w = W + (W - 1) * (self.dilation - 1)

        # Creating a new tensor
        N, T, C = 0, 0, 0
        if features.dim() == 5:
            N, T, C = features.shape[:-2]  # Getting the shape
        elif features.dim() == 4:
            N, C = features.shape[:-2]  # Getting the shape

        # Filling the tensor with original values
        # filler row
        output_tmp = torch.zeros(N, T, C, new_h, W) if features.dim == 5 else torch.zeros(N, C, new_h, W)
        i_output = 0
        for i in range(H):
            output_tmp[..., i_output, :] = features[..., i, :]
            i_output += self.dilation
        # filler colum
        output = torch.zeros(N, T, C, new_h, new_w) if features.dim == 5 else torch.zeros(N, C, new_h, new_w)
        j_output = 0
        for j in range(W):
            output[..., j_output] = output_tmp[..., j]
            j_output += self.dilation

        return output.cuda()


class SpikingNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 bias=False, stride=2, padding=1, groups=1, backend='cupy', up_flag=True):
        super().__init__()

        # bottleneck
        self.bottleneck = layer.SeqToANNContainer(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias,
                      stride=1, padding=0, groups=groups),
        )

        self.neuron = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=1.,
            surrogate_function=surrogate.ATan(),
            detach_reset=True, backend=backend,
        )

        # deconv
        if up_flag:
            self.up_sample = layer.SeqToANNContainer(
                ConstantDilation2d(dilation=stride),
                nn.ConstantPad2d(
                    padding=kernel_size[0] - 1 - padding if isinstance(kernel_size,
                                                                       tuple) else kernel_size - 1 - padding,
                    value=0.),
                nn.BatchNorm2d(out_channels),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size, bias=bias, stride=1,
                                   padding=kernel_size[0] - 1 if isinstance(kernel_size, tuple) else kernel_size - 1)
            )
        else:
            self.up_sample = nn.Identity()

    def forward(self, x, up_flag=True):
        out = self.bottleneck(x)
        out = self.neuron(out)
        if up_flag:
            out = self.up_sample(out)
        return out


class SpikingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=False, stride=1, padding=0, groups=1, backend='cupy'):
        super().__init__()

        self.bn_conv = layer.SeqToANNContainer(
            nn.ConstantPad2d(padding, 0.),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                      stride=stride, padding=0, groups=groups),
        )

        self.neuron = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=1.,
            surrogate_function=surrogate.ATan(),
            detach_reset=True, backend=backend,
        )

    def forward(self, x):
        out = self.bn_conv(x)
        out = self.neuron(out)
        return out


def get_model(args):
    norm_layer = nn.BatchNorm2d if args.bn else None
    ms_neuron = neuron.MultiStepParametricLIFNode

    # Getting Network Parameters
    family, version = args.model.split('-')
    depth, growth_rate = version.split('_')
    blocks = {"121": [6, 12, 24, 16], "169": [6, 12, 32, 32]}
    return multi_step_spiking_densenet_custom(
        2 * args.tbin, norm_layer=norm_layer,
        T=args.T, multi_step_neuron=ms_neuron,
        growth_rate=int(growth_rate), block_config=blocks[depth],
        num_classes=2, backend="cupy",
    )

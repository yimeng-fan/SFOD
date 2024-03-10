import torch
import torch.nn as nn

from torchvision.models.detection.anchor_utils import DefaultBoxGenerator


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def filter_boxes(tensors, min_box_diag=30, min_box_side=20):
    widths = tensors['boxes'][:, 2] - tensors['boxes'][:, 0]  # get all widths
    heights = tensors['boxes'][:, 3] - tensors['boxes'][:, 1]  # get all heights
    diag_square = widths ** 2 + heights ** 2
    mask = (diag_square >= min_box_diag ** 2) * (widths >= min_box_side) * (heights >= min_box_side)
    return {k: v[mask] for k, v in tensors.items()}


class GridSizeDefaultBoxGenerator(DefaultBoxGenerator):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, feature_maps, image_size):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]  # H W
        n_images = feature_maps[0].shape[0]  # N
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        default_boxes = self._grid_default_boxes(grid_sizes, image_size, dtype=dtype)  # Anchor 4
        default_boxes = default_boxes.to(device)

        dboxes = []
        for _ in range(n_images):  # each batch
            dboxes_in_image = default_boxes
            dboxes_in_image = torch.cat(
                [
                    dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:],
                    dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:],
                ],
                -1,
            )
            dboxes_in_image[:, 0::2] *= image_size[1]
            dboxes_in_image[:, 1::2] *= image_size[0]
            dboxes.append(dboxes_in_image)
        return dboxes  # N A 4 (real size)

import os
import random
import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from prophesee_utils.io.psee_loader import PSEELoader

# modified from https://github.com/loiccordone/object-detection-with-spiking-neural-networks/blob/main/datasets/classification_datasets.py

class ClassificationDataset(Dataset):
    def __init__(self, args, mode):
        self.mode = mode
        self.tbin = args.tbin  # number of micro time bins
        self.C, self.T = 2 * args.tbin, args.T  # channel T
        self.sample_size = args.sample_size  # duration of a sample in µs
        self.quantization_size = [args.sample_size // args.T, 1, 1]  # Time per T, y scaling, x scaling
        self.w, self.h = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]

        save_file_name = f"{args.dataset}_{mode}_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
        save_file = os.path.join(args.path, save_file_name)

        if os.path.isfile(save_file):
            self.samples = torch.load(save_file)
            print("File loaded.")
        else:
            data_dir = os.path.join(args.path, mode)
            self.samples = self.build_dataset(data_dir, save_file)
            torch.save(self.samples, save_file)
            print(f"Done! File saved as {save_file}.")

    def __getitem__(self, index):
        coords, feats, target = self.samples[index]

        sample = torch.sparse_coo_tensor(coords.t(), feats.to(torch.float32)).coalesce()
        sample = sample.sparse_resize_(
            (self.T, sample.size(1), sample.size(2), self.C), 3, 1
        ).to_dense().permute(0, 3, 1, 2)

        sample = T.Resize((64, 64), T.InterpolationMode.NEAREST)(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def build_dataset(self, data_dir, save_file):
        raise NotImplementedError("The method build_dataset has not been implemented.")


class NCARSClassificationDataset(ClassificationDataset):
    def __init__(self, args, mode="train"):
        super().__init__(args, mode)

    def build_dataset(self, data_dir, save_file):
        classes_dir = [os.path.join(data_dir, class_name) for class_name in os.listdir(data_dir)]
        samples = []
        for class_id, class_dir in enumerate(classes_dir):
            self.files = [os.path.join(class_dir, time_seq_name) for time_seq_name in os.listdir(class_dir)]
            target = class_id

            print(f'Building the class number {class_id + 1}')
            pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)
            for file_name in self.files:
                print(f"Processing {file_name}...")
                video = PSEELoader(file_name)
                events = video.load_delta_t(self.sample_size)  # Load data

                if events.size == 0:
                    print("Empty sample.")
                    continue

                events['t'] -= events['t'][0]
                coords = torch.from_numpy(
                    structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.float32))

                # Bin the events on T timesteps
                coords = torch.floor(coords / torch.tensor(self.quantization_size))
                coords[:, 1].clamp_(min=0, max=self.quantized_h - 1)
                coords[:, 2].clamp_(min=0, max=self.quantized_w - 1)

                # TBIN computations
                tbin_size = self.quantization_size[0] / self.tbin

                # get for each ts the corresponding tbin index
                tbin_coords = (events['t'] % self.quantization_size[0]) // tbin_size
                # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
                polarity = events['p'].copy().astype(np.int8)
                polarity[events['p'] == 0] = -1
                tbin_feats = (polarity * (tbin_coords + 1))
                tbin_feats[tbin_feats > 0] -= 1
                tbin_feats += (tbin_coords + 1).max()

                feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2 * self.tbin).to(bool)

                samples.append([coords.to(torch.int16), feats, target])
                pbar.update(1)

            pbar.close()

        return samples

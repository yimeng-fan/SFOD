import os
import tqdm

import torch
from torch.utils.data import Dataset

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from prophesee_utils.io.psee_loader import PSEELoader

# modified from https://github.com/loiccordone/object-detection-with-spiking-neural-networks/blob/main/datasets/gen1_od_dataset.py

class GEN1DetectionDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size  # duration of a sample in µs
        self.quantization_size = [args.sample_size // args.T, 1, 1]  # Time per T, y scaling, x scaling
        self.h, self.w = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]

        if mode != 'train':
            save_file_name = \
                f"gen1_{mode}_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
            save_file = os.path.join(args.path, save_file_name)

            if os.path.isfile(save_file):
                self.samples = torch.load(save_file)
                print("File loaded.")
            else:
                data_dir = os.path.join(args.path, mode)
                self.samples = self.build_dataset(data_dir, save_file)
                torch.save(self.samples, save_file)
                print(f"Done! File saved as {save_file}")
        else:  # Since the train is divided into 3 parts, it is processed separately.
            save_file_name_a = \
                f"gen1_train_a_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
            save_file_name_b = \
                f"gen1_train_b_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
            save_file_name_c = \
                f"gen1_train_c_{self.sample_size // 1000}_{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt"
            save_file_a = os.path.join(args.path, save_file_name_a)
            save_file_b = os.path.join(args.path, save_file_name_b)
            save_file_c = os.path.join(args.path, save_file_name_c)

            if os.path.isfile(save_file_a) and os.path.isfile(save_file_b) and os.path.isfile(save_file_c):
                self.samples = torch.load(save_file_a)
                self.samples.extend(torch.load(save_file_b))
                self.samples.extend(torch.load(save_file_c))
                print("File loaded.")
            else:
                print('Processing train_a ...')
                data_dir = os.path.join(args.path, 'train_a')
                self.samples = self.build_dataset(data_dir, save_file_a)
                torch.save(self.samples, save_file_a)
                print(f"Done! File saved as {save_file_a}")
                self.samples = []

                print('Processing train_b ...')
                data_dir = os.path.join(args.path, 'train_b')
                self.samples = self.build_dataset(data_dir, save_file_b)
                torch.save(self.samples, save_file_b)
                print(f"Done! File saved as {save_file_b}")
                self.samples = []

                print('Processing train_c ...')
                data_dir = os.path.join(args.path, 'train_c')
                self.samples = self.build_dataset(data_dir, save_file_c)
                torch.save(self.samples, save_file_c)
                print(f"Done! File saved as {save_file_c}")

                self.samples.extend(torch.load(save_file_a))
                print("File train_a loaded.")
                self.samples.extend(torch.load(save_file_b))
                print("File train_b loaded.")

    def __getitem__(self, index):
        (coords, feats), target = self.samples[index]

        sample = torch.sparse_coo_tensor(
            coords.t(),
            feats.to(torch.float32),
            size=(self.T, self.quantized_h, self.quantized_w, self.C)
        )
        sample = sample.coalesce().to_dense().permute(0, 3, 1, 2)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def build_dataset(self, path, save_file):
        # Remove duplicates (.npy and .dat)
        files = [os.path.join(path, time_seq_name[:-9]) for time_seq_name in os.listdir(path)
                 if time_seq_name[-3:] == 'npy']

        print('Building the Dataset')
        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        samples = []
        for file_name in files:
            print(f"Processing {file_name}...")
            events_file = file_name + '_td.dat'
            video = PSEELoader(events_file)

            boxes_file = file_name + '_bbox.npy'
            boxes = np.load(boxes_file)
            # Rename 'ts' in 't' if needed (Prophesee GEN1)
            boxes.dtype.names = [dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names]

            boxes_per_ts = np.split(boxes, np.unique(boxes['t'], return_index=True)[1][1:])

            samples.extend([sample for b in boxes_per_ts if (sample := self.create_sample(video, b)) is not None])
            pbar.update(1)

        pbar.close()
        return samples

    def create_sample(self, video, boxes):
        ts = boxes['t'][0]
        video.seek_time(ts - self.sample_size)
        events = video.load_delta_t(self.sample_size)

        targets = self.create_targets(boxes)

        if targets['boxes'].shape[0] == 0:
            print(f"No boxes at {ts}")
            return None
        elif events.size == 0:
            print(f"No events at {ts}")
            return None
        else:
            return (self.create_data(events), targets)

    def create_targets(self, boxes):
        torch_boxes = torch.from_numpy(structured_to_unstructured(boxes[['x', 'y', 'w', 'h']], dtype=np.float32))

        # keep only last instance of every object per target
        _, unique_indices = np.unique(np.flip(boxes['track_id']), return_index=True)  # keep last unique objects
        unique_indices = np.flip(-(unique_indices + 1))
        torch_boxes = torch_boxes[[*unique_indices]]

        torch_boxes[:, 2:] += torch_boxes[:, :2]  # implicit conversion to xyxy
        torch_boxes[:, 0::2].clamp_(min=0, max=self.w)
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)

        # valid idx = width and height of GT bbox aren't 0
        valid_idx = (torch_boxes[:, 2] - torch_boxes[:, 0] != 0) & (torch_boxes[:, 3] - torch_boxes[:, 1] != 0)
        torch_boxes = torch_boxes[valid_idx, :]

        torch_labels = torch.from_numpy(boxes['class_id']).to(torch.long)
        torch_labels = torch_labels[[*unique_indices]]
        torch_labels = torch_labels[valid_idx]

        return {'boxes': torch_boxes, 'labels': torch_labels}

    def create_data(self, events):
        events['t'] -= events['t'][0]
        feats = torch.nn.functional.one_hot(torch.from_numpy(events['p']).to(torch.long), self.C)

        coords = torch.from_numpy(
            structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))

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

        return coords.to(torch.int16), feats

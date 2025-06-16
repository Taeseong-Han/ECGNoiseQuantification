import os
import ast
import torch

import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from utils.file_metrics_sort import FileData, FileDataset
from utils.scalogram_utils import scalogram_to_model_input

class PTBXLDataset(Dataset):
    # def __init__(self, ref_min, ref_max, superlet_dir, quantization):
    def __init__(self):

        # self.path = DATABASE_PTBXL / superlet_dir
        # Y = pd.read_csv(Path(__file__).resolve().parent / 'train_clean.csv')
        print(Path(__file__).resolve().parents[1])

    #     self.idx = Y.idx
    #     self.ref_min = ref_min
    #     self.ref_max = ref_max
    #     self.files = Y.filename_hr
    #     self.quantization = quantization
    #
    # def __len__(self):
    #     return len(self.files)
    #
    # def __getitem__(self, idx):
    #     scalogram = np.load(os.path.join(self.path, f'{self.files[idx]}_{self.idx[idx]}.npz'))['scalogram']
    #     scalogram = scalogram_to_model_input(
    #         scalogram,
    #         self.ref_min,
    #         self.ref_max,
    #         discretize=self.quantization
    #     )
    #     scalogram = torch.FloatTensor(scalogram).unsqueeze(0)
    #
    #     return scalogram


# def sort_ptbxl_metric_data(filename, is_train=False):
#     """
#     :param filename:
#     :return: -> all / clean / noise
#     """
#     df = pd.read_csv(PTBXL_METRIC_DIR / filename)
#
#     if is_train:
#         df = df[df['strat_fold'].isin([i for i in range(1, 10)])].reset_index(drop=True)
#
#     container = FileDataset()
#
#     for idx in range(len(df)):
#         metric_values = df.iloc[idx, -12:]
#
#         # clean / noise
#         clean_idx = ast.literal_eval(df.clean[idx])
#         noise_idx = sorted(set(range(12)) - set(clean_idx))
#
#         # noise type
#         baseline_idx = ast.literal_eval(df.baseline_drift[idx])
#         static_idx = ast.literal_eval(df.static_noise[idx])
#         burst_idx = ast.literal_eval(df.burst_noise[idx])
#         electrode_idx = ast.literal_eval(df.electrodes_problems[idx])
#
#         # norm / abnormal
#         is_norm = 'norm' if "NORM': 100" in df.scp_codes[idx] else 'abnorm'
#
#         if is_train:
#             for i in clean_idx:
#                 metadata = f'clean,{is_norm},lead{i}'
#                 filedata = [FileData(
#                     filename=f'{df.filename_hr[idx]}_{i}.npz',
#                     metric_value=metric_values[i],
#                     metadata=metadata)]
#
#                 container.add_files(filedata)
#
#         else: # train이 아닐때에는 fold=10 clean만
#             if int(df.strat_fold[idx]) == 10:
#                 for i in clean_idx:
#                     metadata = f'clean,{is_norm},lead{i}'
#                     filedata = [FileData(
#                         filename=f'{df.filename_hr[idx]}_{i}.npz',
#                         metric_value=metric_values[i],
#                         metadata=metadata)]
#
#                     container.add_files(filedata)
#
#         for i in noise_idx:
#             metadata = f'noise,{is_norm},lead{i}'
#             if i in baseline_idx:
#                 metadata += ',baseline'
#             if i in static_idx:
#                 metadata += ',static'
#             if i in burst_idx:
#                 metadata += ',burst'
#             if i in electrode_idx:
#                 metadata += ',electrode'
#
#             filedata = [FileData(
#                 filename=f'{df.filename_hr[idx]}_{i}.npz',
#                 metric_value=metric_values[i],
#                 metadata=metadata)]
#
#             container.add_files(filedata)
#
#     container.sort_by_value()
#
#     return container

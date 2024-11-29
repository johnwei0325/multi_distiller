# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/bucket_dataset.py ]
#   Synopsis     [ the general acoustic dataset with bucketing ]
#   Author1      [ Andy T. Liu (https://github.com/andi611) ]
#   Author2      [ Heng-Jui Chang (https://github.com/vectominist) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random
import pandas as pd
from torch.utils.data.dataset import Dataset


HALF_BATCHSIZE_TIME = 99999


################
# FEAT DATASET #
################
class FeatDataset(Dataset):
    """Base On-the-fly feature dataset by Andy T. Liu"""
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, **kwargs):
        super(FeatDataset, self).__init__()

        self.extracter = extracter
        self.task_config = task_config
        self.libri_root = libri_root
        self.sample_length = task_config['sequence_length'] 
        if self.sample_length > 0:
            print('[Dataset] - Sampling random segments for training, sample length:', self.sample_length)
        
        # Read file
        self.root = file_path
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        print('[Dataset] - Training data from these sets:', str(sets))

        # Drop seqs that are too long
        if max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
        # Drop seqs that are too short
        if max_timestep < 0:
            self.table = self.table[self.table.length > (-1 * max_timestep)]

        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        self.num_samples = len(X)
        print('[Dataset] - Number of individual training instances:', self.num_samples)

        # Use bucketing to allow different batch size at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)
            
            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_length == 0:
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 1: 
            self.X.append(batch_x)

    def _sample(self, x):
        if self.sample_length <= 0: return x
        if len(x) < self.sample_length: return x
        idx = random.randint(0, len(x)-self.sample_length)
        return x[idx:idx+self.sample_length]

    def __len__(self):
        return len(self.X)

    def collate_fn(self, items):
        items = items[0] # hack bucketing
        return items


################
# WAVE DATASET #
################
class WaveDataset(Dataset):
    """Base waveform dataset for Disiller by Heng-Jui Chang"""

    def __init__(
        self,
        task_config,
        bucket_size,
        file_path,
        sets,
        max_timestep=0,
        libri_root=None,
        **kwargs
    ):
        super().__init__()

        self.task_config = task_config
        self.libri_root = libri_root
        self.sample_length = task_config["sequence_length"]
        if self.sample_length > 0:
            print(
                "[Dataset] - Sampling random segments for training, sample length:",
                self.sample_length,
            )

        # Read file
        self.root = file_path

        # =============================================
        # tables = [pd.read_csv(os.path.join(file_path, s + ".csv")) for s in sets]
        # # self.table = pd.concat(tables, ignore_index=True).sort_values(
        # #     by=["length"], ascending=False
        # # )
        # self.table = pd.concat(tables, ignore_index=True)
        # print("[Dataset] - Training data from these sets:", str(sets))

        # # Drop seqs that are too long
        # if max_timestep > 0:
        #     self.table = self.table[self.table.length < max_timestep]
        # # Drop seqs that are too short
        # if max_timestep < 0:
        #     self.table = self.table[self.table.length > (-1 * max_timestep)]

        # X = self.table["file_path"].tolist()
        # random.shuffle(X)
        # X_lens = self.table["length"].tolist()
        # self.num_samples = len(X)
        # print("[Dataset] - Number of individual training instances:", self.num_samples)

        # # Use bucketing to allow different batch size at run time
        # self.X = []
        # batch_x, batch_len = [], []

        # for x, x_len in zip(X, X_lens):
        #     batch_x.append(x)
        #     batch_len.append(x_len)

        #     # Fill in batch_x until batch is full
        #     if len(batch_x) == bucket_size:
        #         print(max(batch_len))
        #         # Half the batch size if seq too long
        #         if (
        #             (bucket_size >= 2)
        #             and (max(batch_len) > HALF_BATCHSIZE_TIME)
        #             and self.sample_length == 0
        #         ):
        #             self.X.append(batch_x[: bucket_size // 2])
        #             self.X.append(batch_x[bucket_size // 2 :])
        #         else:
        #             self.X.append(batch_x)
        #         batch_x, batch_len = [], []
        # =============================================
        tables = [pd.read_csv(os.path.join(file_path, s + ".csv")) for s in sets]
        print("[Dataset] - Training data from these sets:", str(sets))
        
        # Drop sequences based on max_timestep
        for i, table in enumerate(tables):
            if max_timestep > 0:
                tables[i] = table[table.length < max_timestep]
            elif max_timestep < 0:
                tables[i] = table[table.length > (-1 * max_timestep)]
        
        # Initialize lists for storing batches
        self.X = []
        self.num_samples = sum(len(table) for table in tables)
        print("[Dataset] - Number of individual training instances:", self.num_samples)
        
        # Define per-dataset sample count for each batch
        samples_per_dataset = bucket_size // len(sets)
        print(samples_per_dataset)
        # Prepare data for balanced batching
        tables_shuffled = [table.sample(frac=1).reset_index(drop=True) for table in tables]  # Shuffle each dataset
        self.X = []
        data_indices = [0] * len(tables_shuffled)  # Track the current index for each dataset

        while True:
            batch_x, batch_len = [], []

            # Iterate over each table to pull `samples_per_dataset` entries
            for i, table in enumerate(tables_shuffled):
                start_idx = data_indices[i]
                end_idx = start_idx + samples_per_dataset

                # Ensure we don’t exceed the dataset length
                if end_idx > len(table):
                    end_idx = len(table)  # Adjust if the remaining samples are less than needed

                # Add samples to the current batch
                batch_x.extend(table["file_path"].iloc[start_idx:end_idx].tolist())
                batch_len.extend(table["length"].iloc[start_idx:end_idx].tolist())

                # Update the index for the next batch from this dataset
                data_indices[i] = end_idx

            # Add the completed batch to self.X, with sequence length handling
            if len(batch_x) == bucket_size:
                # Split the batch if sequences are too long
                random.shuffle(batch_x)
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_length == 0:
                    self.X.append(batch_x[:bucket_size // 2])
                    self.X.append(batch_x[bucket_size // 2:])
                else:
                    self.X.append(batch_x)

            # Exit when all datasets are exhausted
            if all(idx >= len(table) for idx, table in zip(data_indices, tables_shuffled)):
                # If there are any remaining samples that don't fill a full batch, add them
                # if batch_x:
                #     self.X.append(batch_x)
                break

        print(len(self.X))

        # # Gather the last batch
        # if len(batch_x) > 1:
        #     self.X.append(batch_x)

    # def _sample(self, x):
    #     if self.sample_length <= 0:
    #         return x
    #     if len(x) < self.sample_length:
    #         return x
    #     idx = random.randint(0, len(x) - self.sample_length)
    #     return x[idx : idx + self.sample_length]

    def _sample(self, x, sample_rate):
        if sample_rate != 16000:
            adjusted_sample_length = int(self.sample_length * sample_rate / 16000)
        else:
            adjusted_sample_length = self.sample_length

        if adjusted_sample_length <= 0:
            return x

        if len(x) < adjusted_sample_length:
            return x

        idx = random.randint(0, len(x) - adjusted_sample_length)
        return x[idx: idx + adjusted_sample_length]


    def __len__(self):
        return len(self.X)

    def collate_fn(self, items):
        items = items[0]  # hack bucketing
        assert (
            len(items) == 5
        ), "__getitem__ should return (wave_input, wave_orig, wave_len, pad_mask)"
        return items

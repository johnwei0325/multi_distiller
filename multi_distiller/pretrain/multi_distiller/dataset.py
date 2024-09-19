"""
    Dataset for distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from pretrain.bucket_dataset import WaveDataset


class OnlineWaveDataset(WaveDataset):
    """Online waveform dataset"""

    def __init__(
        self,
        task_config,
        bucket_size,
        file_path,
        sets,
        max_timestep=0,
        libri_root=None,
        target_level=-25,
        **kwargs
    ):
        super().__init__(
            task_config,
            bucket_size,
            file_path,
            sets,
            max_timestep,
            libri_root,
            **kwargs
        )
        self.target_level = target_level

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
        if wav.shape[0] == 2:  # Check if stereo (2 channels)
            wav = wav.mean(dim=0, keepdim=True)  # Average across channels to get mono
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav  # Add channel dimension if missing

        max_len = 250000  # Set a maximum length (e.g., 1 second of audio at 16kHz)
        if wav.size(1) > max_len:
            wav = wav[:, :max_len]
        return wav.squeeze()  # (seq_len)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        x_lens = [len(x) for x in x_batch]
        x_lens = torch.LongTensor(x_lens)
        x_pad_batch = pad_sequence(x_batch, batch_first=True)

        pad_mask = torch.ones(x_pad_batch.shape)  # (batch_size, seq_len)
        # zero vectors for padding dimension
        for idx in range(x_pad_batch.shape[0]):
            pad_mask[idx, x_lens[idx] :] = 0

        return [x_pad_batch, x_batch, x_lens, pad_mask]

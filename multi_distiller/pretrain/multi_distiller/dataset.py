"""
    Dataset for distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""
import soundfile as sf
import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
#import torchaudio
from pretrain.bucket_dataset import WaveDataset
import torch.nn.functional as F 


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
        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

    def _load_feat(self, feat_path):
        # if self.libri_root is None:
        #     return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        try:
            wav, sample_rate = sf.read(feat_path)
            wav = torch.from_numpy(wav)
            
            wav = wav.to(torch.float32)
        except Exception as e:
            print(f"Connection issue when loading file {feat_path}: {e}, retrying...")
        if wav.shape[0] == 2:  # Check if stereo (2 channels)
            wav = wav.mean(dim=0, keepdim=True)  # Average across channels to get mono
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav  # Add channel dimension if missing


        return [wav.squeeze(), sample_rate]  # (seq_len)

    def __getitem__(self, index):
        try:
            x_batch = []
            x_batch_24k = []
            target_sample_rate_16k = 16000  # Target sample rate (16 kHz)
            target_sample_rate_24k = 24000  # Target sample rate for x_batch_24k (24 kHz)
            sample_domain = []
            # print(self.X)
            for x_file in self.X[index]:
                # Load the waveform
                # print(x_file)
                if 'librispeech' in x_file.lower():
                    sample_domain.append('hubert_base')
                elif 'audioset' in x_file.lower():
                    sample_domain.append('ssast_frame')
                elif 'music4all' in x_file.lower():
                    sample_domain.append('mert_v0_public')
                else:
                    print('Can not recognize data domain')
                waveform, sample_rate = self._load_feat(x_file)
                waveform = self._sample(waveform, sample_rate)
                #waveform = waveform.type(torch.float64)
                # Resample to 16 kHz if the sample rate is different
                if sample_rate != target_sample_rate_16k:
                    resample_transform_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate_16k)
                    waveform_16k = resample_transform_16k(waveform)
                else:
                    waveform_16k = waveform

                # Resample to 24 kHz if the sample rate is different
                # if sample_rate != target_sample_rate_24k:
                    # resample_transform_24k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate_24k)
                    # waveform_24k = resample_transform_24k(waveform)
                # else:
                    # waveform_24k = waveform

                # Handle multi-channel (stereo) audio: downmix to mono if more than 1 channel
                if waveform_16k.ndim == 2 and waveform_16k.shape[0] > 1:  # Check if multi-channel for 16k
                    print(f"Multi-channel audio detected: {waveform_16k.shape}, downmixing to mono.", index)
                    waveform_16k = torch.mean(waveform_16k, dim=0, keepdim=True).squeeze(0)  # Downmix to mono
                # if waveform_24k.ndim == 2 and waveform_24k.shape[0] > 1:  # Check if multi-channel for 24k
                    # waveform_24k = torch.mean(waveform_24k, dim=0, keepdim=True).squeeze(0)  # Downmix to mono

                # Normalize waveforms and append to the respective lists
                x_batch.append(waveform_16k - waveform_16k.mean())
                # x_batch_24k.append(waveform_24k - waveform_24k.mean())
                x_batch_24k = []
            # Pad the sequences to the length of the longest sequence for 16 kHz
            x_pad_batch = pad_sequence(x_batch, batch_first=True)

            # Create the padding mask for 16 kHz
            x_lens = torch.LongTensor([wave.shape[-1] for wave in x_batch])  # Sequence lengths for 16 kHz
            pad_mask = torch.ones(x_pad_batch.shape[:2], dtype=torch.bool)  # (batch_size, seq_len)
            for idx in range(x_pad_batch.shape[0]):
                pad_mask[idx, x_lens[idx]:] = 0  # Mask out padding regions with zeros
            # print(sample_domain)
            return [x_pad_batch, x_batch, sample_domain, x_lens, pad_mask]

        except Exception as e:
            print(f"Error occurred at index {index}: {e}")
            raise e  # Re-raise the exception to stop the process

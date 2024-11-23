"""
    Upstream expert for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import io
import yaml

from ..interfaces import UpstreamBase
from torch.nn.utils.rnn import pad_sequence
# from .builder import PretrainedDistiller
from .model import MultiDistillerModel, MultiDistillerConfig

import torch

class UpstreamExpert(UpstreamBase):
    """
    The Distiller wrapper
    """

    def __init__(self, ckpt, model_config=None, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # with open("./upstream/multi_distiller/config_model.yaml", "r") as f:
        self.model = MultiDistillerModel(MultiDistillerConfig(model_config["multi_distiller"]))
        self.model.load_state_dict(torch.load(ckpt)["Distiller"])

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs, no_pred=False):
        x_pad_batch = pad_sequence(wavs, batch_first=True)

        # Create the padding mask for 16 kHz
        x_lens = torch.LongTensor([wave.shape[-1] for wave in wavs])  # Sequence lengths for 16 kHz
        pad_mask = torch.ones(x_pad_batch.shape[:2], dtype=torch.bool).to(self.device)  # (batch_size, seq_len)
        for idx in range(x_pad_batch.shape[0]):
            pad_mask[idx, x_lens[idx]:] = 0  # Mask out padding regions with zeros

        _, feat_final, pred, pad_mask, layer_hidden = self.model(
            x_pad_batch, pad_mask=pad_mask, get_hidden=True, no_pred=no_pred
        )
        # pred: B x N x T x D
        # if not no_pred:
        #     hidden_feats = pred.transpose(0, 1).split(1, 0)
        #     hidden_feats = [hid.squeeze(0) for hid in hidden_feats]
        # else:
        #     
        hidden_feats = []
        hidden_feats = [feat_final] + layer_hidden + hidden_feats

        states = {
            "last_hidden_state": None if no_pred else hidden_feats[-1],
            "hidden_states": hidden_feats,
            "pad_mask": pad_mask,
            "paper": layer_hidden[-1],  # DistilHuBERT: https://arxiv.org/abs/2110.01900
        }

        return states


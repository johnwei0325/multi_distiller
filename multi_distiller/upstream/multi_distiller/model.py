"""
    Distiller Model
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import torch
from torch import nn

from .module import (
    ConvFeatureExtractionModel,
    GradMultiply,
    SplitLinear,
    TransformerEncoder,
)
from .feature_size import get_model_feature_size
from omegaconf import DictConfig, OmegaConf
from .feature_translators import build_feature_translator
from einops.layers.torch import Rearrange



class MultiDistillerConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        # Feature extractor
        self.extractor_mode = str(config.get("extractor_mode", "default"))
        self.extractor_conv_feature_layers = str(
            config.get(
                "extractor_conv_feature_layers",
                "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
            )
        )
        self.extractor_dropout = float(config.get("extractor_dropout", 0.0))
        self.feature_grad_mult = float(config.get("feature_grad_mult", 1.0))

        # Convolutional relative positional encoding
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))

        # Transformer encoder
        self.encoder_layers = int(config.get("encoder_layers", 1))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))
        self.attention_type = str(config.get("attention_type", "original"))

        # Dropout
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.1))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))

        # Output
        self.final_dim = int(config.get("final_dim", 768))
        self.out_layer_type = str(config.get("out_layer_type", "expand-last"))
        self.out_layer_inter_dim = int(config.get("out_layer_inter_dim", -1))

        # Task & loss
        self.n_tasks = int(config.get("n_tasks", 12))
        self.task_emb_type = str(config.get("task_emb_type", "expand-last"))
        self.task_emb_size = int(config.get("task_emb_size", 0))
        self.layer_emb_size = int(config.get("layer_emb_size", 0))
        self.loss_type = str(config.get("loss_type", "l1"))
        self.feat_pen_loss = float(config.get("feat_pen_loss", 0.0))
        self.cosine_loss = float(config.get("cosine_loss", 0.0))

        # When task_emb_type == 'expand-last' only
        self.pred_layer_id = list(
            config.get("pred_layer_id", range(1, self.n_tasks + 1))
        )

        # Initialization
        self.init_teacher_conv_layers = bool(
            config.get("init_teacher_conv_layers", False)
        )
        self.init_teacher_encoder_layers = bool(
            config.get("init_teacher_encoder_layers", False)
        )

        self.teacher_names = config.get('teacher_names')
        self.translator_kwargs = config.get('translator_kwargs')
        self.translator_type = config.get('translator_type')
        self.layerwise_translator = config.get('layerwise_translator')
        self.initialize_from = config.get('initialize_from')
        self.train_on_all_data = config.get('train_on_all_data')

        # self.initialize_from = [['hubert_base']]

        self.fbank_mean = 0
        self.fbank_std = 0

class MultiDistillerModel(nn.Module):
    """
    Multi-Distiller Model
    """

    def __init__(self, config: MultiDistillerConfig):
        super().__init__()

        self.config = config
        self.teacher_names = config.teacher_names

        self.conv_layers = eval(config.extractor_conv_feature_layers)
        feat_emb_dim = self.conv_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            self.conv_layers,
            dropout=config.extractor_dropout,
            mode=config.extractor_mode,
            conv_bias=False,
        )
        self.feature_grad_mult = config.feature_grad_mult

        self.n_tasks = config.n_tasks
        self.task_emb_type = config.task_emb_type

        final_emb_size = config.encoder_embed_dim
        if self.task_emb_type == "add":
            self.task_embedding = nn.Embedding(config.n_tasks, config.encoder_embed_dim)
            nn.init.normal_(self.task_embedding.weight, 0.0, 0.1)
        elif self.task_emb_type == "concat":
            assert config.task_emb_size > 0
            feat_emb_dim += config.task_emb_size
            self.task_embedding = nn.Embedding(config.n_tasks, config.task_emb_size)
        elif self.task_emb_type == "concat-last":
            assert config.task_emb_size > 0
            self.task_embedding = nn.Embedding(config.n_tasks, config.task_emb_size)
            final_emb_size += config.task_emb_size
        elif self.task_emb_type == "expand-last":
            self.pred_layer_id = config.pred_layer_id
            assert self.n_tasks == len(self.pred_layer_id)
            print(
                f"[DistillerModel] - Expands the output dimension by {self.n_tasks} times"
            )
            print(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "self-hidden":
            self.pred_layer_id = config.pred_layer_id
            assert self.n_tasks == len(self.pred_layer_id)
            assert self.n_tasks == config.encoder_layers + 1
            print("[DistillerModel] - Predicting with self-hidden layers")
            print(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "none":
            print(
                f"[DistillerModel] - Disabled task embedding (predicts only layer {self.n_tasks})"
            )
        else:
            raise NotImplementedError(f"Unknown task emb type {self.task_emb_type}")

        self.post_extract_proj = (
            nn.Linear(feat_emb_dim, config.encoder_embed_dim)
            if feat_emb_dim != config.encoder_embed_dim
            else None
        )

        if config.encoder_layers > 0:
            self.encoder = TransformerEncoder(config)
        else:
            self.encoder = nn.GELU()

        final_dim = config.final_dim * (
            1 if self.task_emb_type != "expand-last" else self.n_tasks
        )

        inter_dim = config.out_layer_inter_dim
        inter_dim = inter_dim if inter_dim > 0 else final_emb_size

        print(f"[DistillerModel] - Out layer type: {config.out_layer_type}")
        if config.out_layer_type == "expand-last":
            assert self.task_emb_type == "expand-last"
            print(f"[DistillerModel] - Inter dim = {inter_dim}")
            # self.output_layer = nn.Sequential(
            #     nn.Linear(final_emb_size, inter_dim * self.n_tasks),
            #     nn.GELU(),
            #     SplitLinear(inter_dim, self.n_tasks, config.final_dim),
            # )
            if config.translator_type == 'lconv' and not config.layerwise_translator:
                self.output_layers = nn.ModuleDict({
                    teacher: nn.Sequential(
                        nn.Linear(final_emb_size, inter_dim * self.n_tasks // 2),  # Linear projection for each teacher
                        nn.GELU(),  # Non-linear activation
                        # Conditionally add a more complex model for 'sasst_frame' teacher
                        nn.LayerNorm(inter_dim * self.n_tasks // 2),  # Normalization for stability
                        Rearrange('b t d -> b d t'),  # Rearrange for Conv1d
                        nn.Conv1d(inter_dim * self.n_tasks // 2, inter_dim * self.n_tasks // 2, kernel_size=3, padding=1),
                        nn.ReLU(),
                        Rearrange('b d t -> b t d'),
                        nn.Linear(inter_dim * self.n_tasks // 2, inter_dim * self.n_tasks),
                        nn.LayerNorm(inter_dim * self.n_tasks),
                        nn.GELU(),
                        SplitLinear(inter_dim, self.n_tasks, config.final_dim)
                    )
                    # ) if teacher == 'ssast_frame' else nn.Sequential(
                    #     nn.Linear(final_emb_size, inter_dim * self.n_tasks),  # Simpler model for other teachers
                    #     nn.GELU(),
                    #     SplitLinear(inter_dim, self.n_tasks, config.final_dim)
                    # )
                for teacher in self.teacher_names  # For each teacher model
                })
            elif config.translator_type == 'none' and not config.layerwise_translator:
                self.output_layers = nn.ModuleDict({
                    teacher: nn.Sequential(
                        nn.Linear(final_emb_size, inter_dim * self.n_tasks),
                        nn.GELU(),
                        SplitLinear(inter_dim, self.n_tasks, config.final_dim),
                    )
                for teacher in self.teacher_names  # For each teacher model
                })
            elif config.translator_type == 'lconv' and config.layerwise_translator:
                self.output_layers = nn.ModuleDict({
                    teacher: nn.ModuleDict({
                        f"task_{task}": nn.Sequential(
                            nn.Linear(final_emb_size, inter_dim),
                            Rearrange('b t d -> b d t'),  # Rearrange for Conv1d
                            nn.Conv1d(inter_dim, inter_dim, kernel_size=3, padding=1),
                            nn.ReLU(),
                            Rearrange('b d t -> b t d'),
                            # nn.Linear(inter_dim // 2, inter_dim),
                            nn.LayerNorm(inter_dim),
                            nn.GELU(),
                            nn.Linear(inter_dim, config.final_dim),
                        )
                        for task in range(self.n_tasks)  # Creating a head for each task
                    })
                    for teacher in self.teacher_names  # For each teacher model
                })
            elif config.translator_type == 'none' and config.layerwise_translator:
                self.output_layers = nn.ModuleDict({
                    teacher: nn.ModuleDict({
                        f"task_{task}": nn.Sequential(
                            nn.Linear(final_emb_size, inter_dim * 2),
                            nn.GELU(),
                            nn.Linear(inter_dim * 2, config.final_dim),
                        )
                        for task in range(self.n_tasks)  # Creating a head for each task
                    })
                    for teacher in self.teacher_names  # For each teacher model
                })
            print(self.output_layers)
            
        elif config.out_layer_type in {"none", "self-hidden"}:
            self.output_layer = None
        else:
            raise NotImplementedError(f"Unknown out layer type {config.out_layer_type}")

        # self.target_feature_sizes = {t: get_model_feature_size(t) for t in self.teacher_names}
        # self.backbone_feature_size = [12, 2, 176, 768]
        # self.feature_translators = nn.ModuleDict({
        #     'hubert_base': FeatureTranslator(feat_emb_dim, config.hubert_hidden_size),
        #     'mert_v0_public': FeatureTranslator(feat_emb_dim, config.mert_hidden_size),
        #     'ast': FeatureTranslator(feat_emb_dim, config.ast_hidden_size)
        # })
        # if self.target_feature_sizes:
        #     translator_kwargs = {} if config.translator_kwargs is None else config.translator_kwargs
        #     translator_kwargs["input_size"] = self.backbone_feature_size
        #     translator_kwargs["target_model_names"] = self.teacher_names
        #     self.translator = build_feature_translator(config.translator_type, **translator_kwargs)

    def forward_feature(self, wave, pad_mask):
        """Forward feature extractor"""

        if self.feature_grad_mult > 0:
            feat = self.feature_extractor(wave)
            if self.feature_grad_mult != 1.0:
                feat = GradMultiply.apply(feat, self.feature_grad_mult)
        else:
            with torch.no_grad():
                feat = self.feature_extractor(wave)

        feat = feat.transpose(1, 2)  # B x T x D
        pad_mask = self.cal_pad_mask(pad_mask, feat.shape[1])

        return feat, pad_mask

    def forward(self, wave, pad_mask, task_id=None, get_hidden=False, no_pred=False):
        """
        Forward function
        Input:
            wave (FloatTensor): B x T_wave
            pad_mask (BoolTensor): B x T_wave
            task_id (LongTensor): N >= 1
        """

        feat, pad_mask = self.forward_feature(wave, pad_mask)

        if self.task_emb_type not in ["none", "expand-last", "self-hidden"]:
            if task_id is None:
                task_id = self.generate_task_id(feat.device)
            elif isinstance(task_id, list):
                task_id = torch.LongTensor(task_id).to(feat.device)
            task_embs = self.task_embedding(task_id)
            # N x D
            n_sz = len(task_id)
        else:
            n_sz = 1
        b_sz, t_sz, _ = feat.shape

        if self.task_emb_type == "add":
            # Add embs to feature
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1) + task_embs.unsqueeze(0).unsqueeze(2)
        elif self.task_emb_type == "concat":
            # Concatenates embs to feature
            feat_final = torch.cat(
                [
                    feat.unsqueeze(1).expand(-1, n_sz, -1, -1),
                    task_embs.unsqueeze(0).unsqueeze(2).expand(b_sz, -1, t_sz, -1),
                ],
                dim=-1,
            )
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat_final)
        else:
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1)
        # feat_final: B x N x T x D or B x 1 x T x D

        pad_mask = pad_mask.unsqueeze(1).expand(-1, n_sz, -1).reshape(b_sz * n_sz, t_sz) ### to mask on time dim,.
        # BN x T
        feat_final = feat_final.reshape(b_sz * n_sz, t_sz, -1)
        # BN x T x D

        layer_hiddens = []
        if self.config.encoder_layers > 0:
            get_hidden_tmp = (
                True if (self.task_emb_type == "self-hidden") else get_hidden
            )
            hidden, layer_hiddens = self.encoder(
                feat_final, ~pad_mask.bool(), get_hidden=get_hidden_tmp
            )
        else:
            hidden = self.encoder(feat_final)
        
        
        if not no_pred:
            if self.task_emb_type == "self-hidden":
                pred = torch.stack([feat_final] + layer_hiddens, dim=1)
            else:
                pred = {}
                if self.config.layerwise_translator:
                    for teacher in self.teacher_names:
                        layer_preds = []
                        for task_num in range(self.n_tasks):
                            layer_pred = self.output_layers[teacher][f"task_{task_num}"](hidden).reshape(b_sz, t_sz, -1)
                            layer_preds.append(layer_pred)
                        pred[teacher] = torch.stack(layer_preds, dim=1)
                else:
                    for teacher in self.teacher_names:
                        pred[teacher] = self.output_layers[teacher](hidden).reshape(b_sz, t_sz, -1)
            # B x N x T x D
        else:
            pred = None

        if (not no_pred) and self.task_emb_type == "expand-last":
            assert n_sz == 1, n_sz
            for teacher in self.teacher_names:
                pred[teacher] = (
                    pred[teacher]
                    .squeeze(1)
                    .reshape(b_sz, t_sz, self.n_tasks, -1)
                    .permute(0, 2, 1, 3)
                )
            # B x N x T x D

        # translated_predictions = self.translator(pred, self.teacher_names) 

        if get_hidden:
            return feat, feat_final, pred, pad_mask, layer_hiddens
        else:
            return feat, feat_final, pred, pad_mask
        # if get_hidden:
            # return feat, feat_final, pred, pad_mask, layer_hiddens
        # else:
            # return feat, feat_final, pred, pad_mask

    def cal_pad_mask(self, pad_mask, max_len):
        """Calculates pad mask after conv."""
        pad_len = (pad_mask > 0).sum(1).long()
        for _, k_size, s_size in self.conv_layers:
            pad_len = torch.div((pad_len - k_size), s_size, rounding_mode="trunc") + 1

        new_pad_mask = torch.ones(
            (pad_mask.shape[0], max_len), dtype=pad_mask.dtype, device=pad_mask.device
        )

        for idx in range(pad_len.shape[0]):
            new_pad_mask[idx, pad_len[idx] :] = 0

        return new_pad_mask

    def generate_task_id(self, device):
        return torch.arange(self.n_tasks, device=device, dtype=torch.long)

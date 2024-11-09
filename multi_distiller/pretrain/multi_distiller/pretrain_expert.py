"""
    Pre-train expert for distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

from easydict import EasyDict as edict
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pretrain.multi_distiller.dataset import OnlineWaveDataset
from upstream.multi_distiller.model import MultiDistillerConfig, MultiDistillerModel
from transformers import AutoModel, AutoConfig
import torchaudio
import torchaudio.transforms as transforms
from transformers import ASTForAudioClassification
from transformers import AutoProcessor, ASTModel


# from audiossl.models.atst.atst import ATST
# from ......MERT.mert_fairseq.models.mert.mert_model import MERTConfig

def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False

def remap_keys(state_dict, prefix):
    """Remap keys in the state_dict to match the model's expected structure."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.backbone'):
            new_key = key.replace('module.backbone', 'encoder')
        elif key.startswith('module.head'):
            new_key = key.replace('module.head', 'projector')  # Adjust based on your model's needs
        elif key.startswith('module.'):
            new_key = key.replace('module.', '', 1)
        else:
            new_key = key
        new_state_dict[f'{prefix}.{new_key}'] = value
    return new_state_dict

def get_ATST_teacher_model(arch, ncrops, atst_model_path, target_device):
    """Configure the ATST model by loading weights, disabling dropouts, and freezing the model."""
    # Initialize the model
    kwargs = {}  # Additional arguments if needed
    teacher_3 = ATST(arch=arch, ncrops=ncrops, **kwargs)

    # Load the full state dictionary from the file
    full_state_dict = torch.load(atst_model_path, map_location='cpu')

    # Extract the 'student' and 'teacher' state dictionaries
    student_state_dict = full_state_dict.get('student', {})
    teacher_state_dict = full_state_dict.get('teacher', {})

    # Remap keys
    student_state_dict = remap_keys(student_state_dict, 'student')
    teacher_state_dict = remap_keys(teacher_state_dict, 'teacher')

    # Combine the remapped state dictionaries
    combined_state_dict = {**student_state_dict, **teacher_state_dict}

    # Load the combined state dict into your model with strict=False to allow minor mismatches
    try:
        missing_keys, unexpected_keys = teacher_3.load_state_dict(combined_state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")

    # Move the model to the desired GPU
    teacher_3.to(target_device)

    # Disable dropouts in the student encoder's blocks
    for block in teacher_3.student.encoder.blocks:
        block.attn.attn_drop.p = 0.0
        block.attn.proj_drop.p = 0.0
        block.mlp.drop.p = 0.0
        if hasattr(block, 'drop_path') and isinstance(block.drop_path, torch.nn.Dropout):
            block.drop_path.p = 0.0

    # Disable dropouts in the teacher encoder's blocks
    for block in teacher_3.teacher.encoder.blocks:
        block.attn.attn_drop.p = 0.0
        block.attn.proj_drop.p = 0.0
        block.mlp.drop.p = 0.0
        if hasattr(block, 'drop_path') and isinstance(block.drop_path, torch.nn.Dropout):
            block.drop_path.p = 0.0

    print("[ATST] - Disabled all dropouts in the encoder's blocks")

    # Freeze the parameters of the teacher and student models

    return teacher_3

def disable_MERT_encoder_dropout(model):
    """Disable all dropouts in the encoder layers of the model by setting their probabilities to 0.0."""

    # Disable encoder layer dropout if available in config
    if hasattr(model.config, 'encoder_layerdrop'):
        model.config.encoder_layerdrop = 0.0  # Set encoder layer dropout to 0
        print("[MERT] - Disabled all dropouts in the encoder's blocks via config")

    # Iterate through all encoder layers and disable their dropouts by setting p=0.0
    for layer in model.encoder.layers:
        # Disable attention dropout
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'dropout'):
            if isinstance(layer.attention.dropout, nn.Dropout):
                layer.attention.dropout.p = 0.0  # Correctly set the probability to 0
            elif isinstance(layer.attention.dropout, float):
                layer.attention.dropout = 0.0  # Directly set the float value

        # Disable intermediate and output dropouts in the feed-forward layer
        if hasattr(layer, 'feed_forward'):
            if hasattr(layer.feed_forward, 'intermediate_dropout'):
                if isinstance(layer.feed_forward.intermediate_dropout, nn.Dropout):
                    layer.feed_forward.intermediate_dropout.p = 0.0
                elif isinstance(layer.feed_forward.intermediate_dropout, float):
                    layer.feed_forward.intermediate_dropout = 0.0  # Directly set the float value

            if hasattr(layer.feed_forward, 'output_dropout'):
                if isinstance(layer.feed_forward.output_dropout, nn.Dropout):
                    layer.feed_forward.output_dropout.p = 0.0
                elif isinstance(layer.feed_forward.output_dropout, float):
                    layer.feed_forward.output_dropout = 0.0  # Directly set the float value

        # Disable general dropout in the layer if applicable
        if hasattr(layer, 'dropout'):
            if isinstance(layer.dropout, nn.Dropout):
                layer.dropout.p = 0.0
            elif isinstance(layer.dropout, float):
                layer.dropout = 0.0  # Directly set the float value

    print("[MERT] - Disabled all dropouts in the encoder's layers by setting p=0.0 where applicable")


def disable_AST_encoder_dropout(model):
    """Disable all dropouts in the encoder layers of the ASTModel by setting their probabilities to 0.0."""

    # Disable encoder layer dropout if available in config
    if hasattr(model.config, 'encoder_layerdrop'):
        model.config.encoder_layerdrop = 0.0  # Set encoder layer dropout to 0

    # Iterate through all encoder layers and disable their dropouts by setting p=0.0
    for layer in model.encoder.layer:  # Access each ASTLayer in the encoder
        # Disable attention dropout within ASTAttention
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'attention'):
            attention = layer.attention.attention
            if hasattr(attention, 'dropout') and isinstance(attention.dropout, nn.Dropout):
                attention.dropout.p = 0.0  # Set attention dropout to 0.0

        # Disable output dropout within ASTSelfOutput
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'output'):
            output = layer.attention.output
            if hasattr(output, 'dropout') and isinstance(output.dropout, nn.Dropout):
                output.dropout.p = 0.0  # Set output dropout to 0.0

        # Disable intermediate dropout within ASTIntermediate
        if hasattr(layer, 'intermediate'):
            intermediate = layer.intermediate
            if hasattr(intermediate, 'intermediate_act_fn'):
                act_fn = intermediate.intermediate_act_fn
                if isinstance(act_fn, nn.Dropout):
                    act_fn.p = 0.0  # Set intermediate activation dropout to 0.0

        # Disable output dropout within ASTOutput
        if hasattr(layer, 'output'):
            output = layer.output
            if hasattr(output, 'dropout') and isinstance(output.dropout, nn.Dropout):
                output.dropout.p = 0.0  # Set output dropout to 0.0

        # Disable any general dropout in the layer if applicable
        if hasattr(layer, 'dropout') and isinstance(layer.dropout, nn.Dropout):
            layer.dropout.p = 0.0

    print("[AST] - Disabled all dropouts in the encoder's layers by setting p=0.0 where applicable")

class UpstreamPretrainExpert(nn.Module):
    """
    The Distiller pretrain expert
    """

    def __init__(
        self, datarc, upstream_config, device="cuda", multi_gpu=False, **kwargs
    ):
        super().__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(
                open(upstream_config, "r"), Loader=yaml.FullLoader
            )
            print(
                "[UpstreamPretrainExpert] - Using upstream config from:",
                upstream_config,
            )
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print(
                "[UpstreamPretrainExpert] - Using upstream config from the previous experiment."
            )
        else:
            raise ValueError

        self._get_train_dataloader()

        print("[UpstreamPretrainExpert] - Initializing model...")
        model_config = MultiDistillerConfig(self.upstream_config["multi_distiller"])
        self.model = MultiDistillerForPretrain(
            model_config, edict(self.upstream_config["teacher"])
        )

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print(
                "[UpstreamPretrainExpert] - Multi-GPU training Enabled: "
                + str(torch.cuda.device_count())
            )
        print(
            "[UpstreamPretrainExpert] - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        )

    def _get_train_dataloader(self):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            **self.datarc,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            self.model.module.distiller.load_state_dict(all_states["Distiller"])
        else:
            self.model.distiller.load_state_dict(all_states["Distiller"])

    # Interface
    def add_state_to_save(self, all_states):
        all_states["Distiller"] = (
            self.model.float().distiller.state_dict()
            if not self.multi_gpu
            else self.model.float().module.distiller.state_dict()
        )
        all_states["Config"] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [wave_input, pad_mask]

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss
        """

        wave_input, wave_orig, wave_len, pad_mask = data
        wave_input = wave_input.to(self.device)
        wave_len = wave_len.to(self.device)
        pad_mask = pad_mask.type(wave_input.dtype).to(self.device)

        loss, other_res = self.model(
            wave_input,
            wave_orig,
            wave_len,
            pad_mask,
            return_other=global_step % log_step == 0,
        )

        if global_step % log_step == 0:
            for key, value in other_res.items():
                if isinstance(value, torch.Tensor):
                    value = float(value.mean().cpu().item())
                records[key] = value

        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                logger.add_image(f"{prefix}{key}", values, global_step=global_step)
            elif isinstance(values, float):
                logger.add_scalar(f"{prefix}{key}", values, global_step=global_step)


class MultiDistillerForPretrain(nn.Module):
    """
    Distiller for pretraining with flexible number of teacher models.
    """

    def __init__(self, config: MultiDistillerConfig, teacher_config: edict):
        super().__init__()
        self.config = config
        self.distiller = MultiDistillerModel(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher_config = teacher_config
        self.teachers = teacher_config.models  # Expecting a list of teacher model names

        # Dictionary to store teacher models and processors
        self.teacher_models = {}
        self.teacher_processors = {}

        # Load teacher models based on self.teachers
        for model_name in self.teachers:
            if model_name == 'hubert_base':
                teacher_1 = torch.hub.load("s3prl/s3prl",model_name).to(device)
                if model_name.find("hubert") >= 0 or model_name.find("wav2vec2") >= 0:
                    teacher_1.model.encoder.layerdrop = 0
                    print("[HuBERT] - Disabled all dropouts in the encoder's blocks")
                self.teacher_models[model_name] = teacher_1
            elif model_name == 'mert_v0_public':
                temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
                temp_config.output_hidden_states = True  # Enable hidden states in the output
                teacher_2 = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to(device)
                disable_MERT_encoder_dropout(teacher_2)
                self.teacher_models[model_name] = teacher_2
            elif model_name == 'ast':
                temp_config = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                temp_config.output_hidden_states = True  # Enable output of hidden states
                teacher_3 = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=temp_config).to(device)
                teacher_3_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                disable_AST_encoder_dropout(teacher_3)
                self.teacher_models[model_name] = teacher_3
                self.teacher_processors[model_name] = teacher_3_processor
            else:
                print(f"Warning: Unknown teacher model {model_name} specified.")

        # Freeze all teacher models
        for teacher in self.teacher_models.values():
            freeze_model(teacher)

        # Initialize loss function
        if config.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="none")
        elif config.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(config.loss_type)

        self.cosine_loss = config.cosine_loss
        if self.cosine_loss > 0:
            print("[DistillerForPretrain] - Enabled cosine similarity loss.")

        if config.init_teacher_conv_layers:
            print("[DistillerForPretrain] - Initializing feature extractor from teacher")
            self.distiller.feature_extractor.load_state_dict(
                self.teacher_models['hubert_base'].model.feature_extractor.state_dict()
            )
            if self.distiller.post_extract_proj is not None:
                self.distiller.post_extract_proj.load_state_dict(
                    self.teacher_models['hubert_base'].model.post_extract_proj.state_dict()
                )

        if config.init_teacher_encoder_layers:
            print("[DistillerForPretrain] - Initializing encoder from teacher")
            self.distiller.encoder.pos_conv.load_state_dict(
                self.teacher_models['hubert_base'].model.encoder.pos_conv.state_dict()
            )
            for l in range(config.encoder_layers):
                self.distiller.encoder.layers[l].load_state_dict(
                    self.teacher_models['hubert_base'].model.encoder.layers[l].state_dict()
                )

    def forward(
        self,
        wave_input: torch.Tensor,
        wave_orig: list,
        wave_len: torch.Tensor,
        pad_mask: torch.Tensor,
        return_other: bool = False,
    ):
        """
        Forward function.
        """
        feat, feat_final, pred, pad_mask = self.distiller(wave_input, pad_mask)

        teachers_hidden_states = {}
        with torch.no_grad():
            wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
            if isinstance(wave_orig, list):
                    max_length = max(wave.size(0) for wave in wave_orig)
                    padded_wave_orig = [F.pad(wave, (0, max_length - wave.size(0))) for wave in wave_orig]
                    wave_orig = torch.stack(padded_wave_orig).to(wave_input.device)
            with torch.cuda.amp.autocast(False):
                # Loop through the teacher models to gather hidden states
                for model_name, teacher in self.teacher_models.items():
                    if model_name == 'hubert_base':
                        teacher_hiddens = teacher(wave_orig)
                    elif model_name == 'mert_v0_public':
                        teacher_hiddens = teacher(wave_orig)
                    elif model_name == 'ast':
                        inputs = self.teacher_processors[model_name](wave_orig.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
                        inputs = {key: value.to('cuda:0') for key, value in inputs.items()}
                        teacher_hiddens = teacher(**inputs)

                    # Extract hidden states based on task embedding type
                    if self.config.task_emb_type in ["expand-last", "hnet", "self-hidden"]:
                        teacher_hiddens = [
                            teacher_hiddens["hidden_states"][i]
                            for i in self.distiller.pred_layer_id
                        ]
                        teachers_hidden_states[model_name] = torch.stack(teacher_hiddens, dim=1)

        # Compute all objectives
        (
            total_loss,
            rec_loss,
            rec_layer_loss_dict,
            feat_pen,
            sim_loss,
            sim_layer_loss_dict,
        ) = self.compute_loss(feat, pred, teachers_hidden_states, return_other)

        if return_other:
            with torch.no_grad():
                other_res = {
                    "rec_loss": rec_loss,
                    "feat_pen": feat_pen,
                    "sim_loss": sim_loss,
                    "norm_feat_final": feat_final.pow(2).mean(),
                }

                # Initialize a dictionary to keep norms for each teacher
                teacher_norms = {}

                # Calculate norms for each teacher and add to teacher_norms
                for model_name, hidden_states in teachers_hidden_states.items():
                    teacher_norms[model_name] = torch.abs(hidden_states).mean((0, 2, 3))

                # Log metrics for each teacher
                for model_name, norm in teacher_norms.items():
                    # Retrieve the layer-wise losses from the dictionaries
                    rec_layer_loss = rec_layer_loss_dict.get(model_name, None)
                    sim_layer_loss = sim_layer_loss_dict.get(model_name, None)

                    if rec_layer_loss is not None:
                        # If task_emb_type is 'none', log only the first layer as before
                        if self.config.task_emb_type == "none":
                            other_res[f"rec_l_{model_name}_{self.config.n_tasks}"] = rec_layer_loss[0]
                            other_res[f"tar_norm_l_{model_name}_{self.config.n_tasks}"] = norm[0]
                            if sim_layer_loss is not None:
                                other_res[f"sim_l_{model_name}_{self.config.n_tasks}"] = sim_layer_loss[0]
                        else:
                            # Otherwise, log all layers or based on pred_layer_id
                            for i in range(min(self.config.n_tasks, len(rec_layer_loss))):
                                layer_id = i + 1
                                if self.config.task_emb_type in ["expand-last", "hnet", "self-hidden"]:
                                    layer_id = self.distiller.pred_layer_id[i]

                                # Logging for each layer of each teacher
                                other_res[f"rec_l_{model_name}_{layer_id}"] = rec_layer_loss[i]
                                other_res[f"tar_norm_l_{model_name}_{layer_id}"] = norm[i]
                                if sim_layer_loss is not None and i < len(sim_layer_loss):
                                    other_res[f"sim_l_{model_name}_{layer_id}"] = sim_layer_loss[i]

                # Additional task embedding logging if applicable
                if self.config.task_emb_type not in ["expand-last", "hnet", "self-hidden"]:
                    other_res["norm_task_emb"] = self.distiller.task_embedding.weight.pow(2).mean()
        else:
            other_res = None



        return total_loss, other_res

    def compute_loss(self, feat, pred, target, return_other=False):
        """
        Computes loss for multiple teachers.
        Inputs:
            feat: B x T x D
            pred: Dict containing predictions from multiple teachers
            target: Dict containing targets corresponding to each teacher
            return_other: Flag to indicate if additional losses should be returned
        """
        # Initialize variables to accumulate losses
        total_loss = 0
        total_rec_loss = 0
        total_sim_loss = 0
        total_feat_pen = 0

        rec_layer_loss_dict = {}
        sim_layer_loss_dict = {}

        # Iterate over each teacher's predictions and targets
        for teacher_key in pred.keys():
            teacher_pred = pred[teacher_key]  # Prediction from the current teacher
            teacher_target = target[teacher_key]  # Target corresponding to the current teacher

            # Ensure shapes match
            assert teacher_pred.shape == teacher_target.shape, (teacher_pred.shape, teacher_target.shape)

            # Compute reconstruction loss
            rec_loss = self.loss_func(teacher_pred, teacher_target)  # B x N x T x D
            total_rec_loss += rec_loss.mean()

            # Optionally compute layer-wise reconstruction loss
            if return_other:
                with torch.no_grad():
                    rec_layer_loss = rec_loss.mean((0, 2, 3))  # Per-layer reconstruction loss
                rec_layer_loss_dict[teacher_key] = rec_layer_loss
            else:
                rec_layer_loss_dict[teacher_key] = None

            # Compute cosine similarity loss if applicable
            if self.cosine_loss > 0:
                sim_loss = -F.logsigmoid(F.cosine_similarity(teacher_pred, teacher_target, dim=-1))  # B x N x T
                total_sim_loss += sim_loss.mean()

                # Optionally compute layer-wise similarity loss
                if return_other:
                    with torch.no_grad():
                        sim_layer_loss = sim_loss.mean((0, 2))  # Per-layer similarity loss
                    sim_layer_loss_dict[teacher_key] = sim_layer_loss
                else:
                    sim_layer_loss_dict[teacher_key] = None
            else:
                sim_layer_loss = 0
                sim_layer_loss_dict[teacher_key] = None

            # Compute feature penalty loss
            feat_pen = feat.float().pow(2).mean()
            total_feat_pen += feat_pen

        # Sum up the total loss components
        total_loss = (
            total_rec_loss
            + total_feat_pen * self.config.feat_pen_loss
            + total_sim_loss * self.cosine_loss
        )

        # print("=====================================", total_loss)
        return total_loss, total_rec_loss, rec_layer_loss_dict, total_feat_pen, total_sim_loss, sim_layer_loss_dict
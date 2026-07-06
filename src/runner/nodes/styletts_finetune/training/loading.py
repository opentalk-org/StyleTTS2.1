#coding:utf-8

import os
import os.path as osp

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from runner.nodes.styletts_finetune.training.modules.asr.models import ASRCNN
from runner.nodes.styletts_finetune.training.modules.jdc import JDCNet

from runner.nodes.styletts_finetune.training.modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from runner.nodes.styletts_finetune.training.modules.diffusion.modules import Transformer1d, StyleTransformer1d
from runner.nodes.styletts_finetune.training.modules.diffusion.diffusion import AudioDiffusionConditional

from runner.nodes.styletts_finetune.training.modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

from runner.nodes.styletts_finetune.training.modules.encoders import TextEncoder, StyleEncoder
from runner.nodes.styletts_finetune.training.modules.predictors import ProsodyPredictor
from runner.nodes.styletts_finetune.training.modules.hifigan import Decoder as HifiganDecoder
from runner.nodes.styletts_finetune.training.modules.istftnet import Decoder as IstftnetDecoder
from runner.nodes.styletts_finetune.training.modules.plbert import load_plbert
from runner.nodes.styletts_finetune.training.state_dict_resize import merge_state_dict_with_dim0_resize

from munch import Munch
import yaml

_ASR_N_TOKEN_DIM0_KEYS = frozenset({
    "ctc_linear.2.linear_layer.weight",
    "ctc_linear.2.linear_layer.bias",
    "asr_s2s.embedding.weight",
    "asr_s2s.project_to_n_symbols.weight",
    "asr_s2s.project_to_n_symbols.bias",
})

_CHECKPOINT_DIM0_RESIZE_KEYS_BY_MODULE = {
    "bert": frozenset({
        "embeddings.word_embeddings.weight",
        "module.embeddings.word_embeddings.weight",
    }),
    "text_encoder": frozenset({
        "embedding.weight",
        "module.embedding.weight",
    }),
    "text_aligner": frozenset({
        "ctc_linear.2.linear_layer.weight",
        "ctc_linear.2.linear_layer.bias",
        "asr_s2s.embedding.weight",
        "asr_s2s.project_to_n_symbols.weight",
        "asr_s2s.project_to_n_symbols.bias",
        "module.ctc_linear.2.linear_layer.weight",
        "module.ctc_linear.2.linear_layer.bias",
        "module.asr_s2s.embedding.weight",
        "module.asr_s2s.project_to_n_symbols.weight",
        "module.asr_s2s.project_to_n_symbols.bias",
    }),
}
def _merge_checkpoint_state_with_dim0_resize(module_name, model_module, ckpt_sd):
    resize_keys = _CHECKPOINT_DIM0_RESIZE_KEYS_BY_MODULE.get(module_name)
    if not resize_keys:
        return ckpt_sd
    return merge_state_dict_with_dim0_resize(
        model_module,
        ckpt_sd,
        resize_keys,
        error_scope=module_name,
    )

def _maybe_normalize_module_prefix(model_module, state_dict):
    model_keys = list(model_module.state_dict().keys())
    if not model_keys:
        return state_dict
    state_has_module_prefix = any(key.startswith("module.") for key in state_dict)
    model_has_module_prefix = any(key.startswith("module.") for key in model_keys)
    if state_has_module_prefix == model_has_module_prefix:
        return state_dict
    if state_has_module_prefix and not model_has_module_prefix:
        return {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
    return {
        (f"module.{key}" if not key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _filter_state_dict_by_model_shape(model_module, state_dict):
    model_state = model_module.state_dict()
    filtered_state = {}
    skipped_missing = []
    skipped_shape = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append(
                (key, tuple(value.shape), tuple(model_state[key].shape))
            )
            continue
        filtered_state[key] = value
    return filtered_state, skipped_missing, skipped_shape


def _preview_list(values, limit=5):
    if not values:
        return []
    if len(values) <= limit:
        return values
    return values[:limit] + [f"... ({len(values) - limit} more)"]


def _preview_shape_mismatches(values, limit=5):
    if not values:
        return []
    if len(values) <= limit:
        return values
    return values[:limit] + [("...", "...", f"{len(values) - limit} more")]


def _warn_checkpoint_mismatch(module_name, skipped_missing, skipped_shape, missing_keys, unexpected_keys):
    if skipped_missing:
        print(
            f"[checkpoint] {module_name}: dropped missing-in-model keys: "
            f"{_preview_list(skipped_missing)}"
        )
    if skipped_shape:
        print(
            f"[checkpoint] {module_name}: dropped shape-mismatch keys: "
            f"{_preview_shape_mismatches(skipped_shape)}"
        )
    if missing_keys:
        print(
            f"[checkpoint] {module_name}: model keys left uninitialized from checkpoint: "
            f"{_preview_list(missing_keys)}"
        )
    if unexpected_keys:
        print(
            f"[checkpoint] {module_name}: unexpected checkpoint keys after filtering: "
            f"{_preview_list(unexpected_keys)}"
        )


def load_F0_models(path):
    F0_model = JDCNet(num_class=1, seq_len=192)
    if path is not None:
        params = torch.load(path, map_location='cpu', weights_only=False)['net']
        F0_model.load_state_dict(params)
    else:
        print("No F0 model found, using default F0 model from checkpoint")
    
    F0_model.train()
    
    return F0_model

def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    def _load_config(path_or_dict):
        if isinstance(path_or_dict, dict):
            config = path_or_dict
        else:
            with open(path_or_dict) as f:
                config = yaml.safe_load(f)
        return config["model_params"]

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        if model_path is not None:
            params = torch.load(model_path, map_location='cpu', weights_only=False)['model']
            adapted = merge_state_dict_with_dim0_resize(
                model,
                params,
                _ASR_N_TOKEN_DIM0_KEYS,
                error_scope="ASR",
            )
            model.load_state_dict(adapted)
        else:
            print("No ASR model found, using default ASR model from checkpoint")
        return model

    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model

def build_model(args, text_aligner, pitch_extractor, bert):
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    generator_checkpointing = bool(getattr(args.decoder, "gradient_checkpointing", False))
    discriminators_checkpointing = bool(getattr(args, "discriminators_checkpointing", False))
    
    if args.decoder.type == "istftnet":
        decoder = IstftnetDecoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size,
                gradient_checkpointing=generator_checkpointing) 
    else:
        decoder = HifiganDecoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
                gradient_checkpointing=generator_checkpointing) 
        
    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    
    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
    
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
        
    # define diffusion model
    if args.multispeaker:
        transformer = StyleTransformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    context_features=args.style_dim*2, 
                                    **args.diffusion.transformer)
    else:
        transformer = Transformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    **args.diffusion.transformer)
    
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
        channels=args.style_dim*2,
        context_features=args.style_dim*2,
    )
    
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0 
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    
    nets = Munch(
            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

            predictor=predictor,
            decoder=decoder,
            text_encoder=text_encoder,

            predictor_encoder=predictor_encoder,
            style_encoder=style_encoder,
            diffusion=diffusion,

            text_aligner = text_aligner,
            pitch_extractor=pitch_extractor,

            mpd = MultiPeriodDiscriminator(gradient_checkpointing=discriminators_checkpointing),
            msd = MultiResSpecDiscriminator(gradient_checkpointing=discriminators_checkpointing),
        
            # slm discriminator head
            wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
       )
    
    return nets

def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location="cpu", weights_only=False)
    params = state["net"]
    for key in model:
        if key in params and key not in ignore_modules:
            print("%s loaded" % key)
            normalized_params = _maybe_normalize_module_prefix(model[key], params[key])
            adapted_params = _merge_checkpoint_state_with_dim0_resize(key, model[key], normalized_params)
            filtered_params, skipped_missing, skipped_shape = _filter_state_dict_by_model_shape(
                model[key],
                adapted_params,
            )
            load_result = model[key].load_state_dict(filtered_params, strict=False)
            missing_keys = [
                missing_key for missing_key in load_result.missing_keys
                if not missing_key.endswith("dummy_tensor")
            ]
            unexpected_keys = [
                unexpected_key for unexpected_key in load_result.unexpected_keys
                if not unexpected_key.endswith("dummy_tensor")
            ]
            _warn_checkpoint_mismatch(
                key,
                skipped_missing,
                skipped_shape,
                missing_keys,
                unexpected_keys,
            )
    _ = [model[key].eval() for key in model]    

    if not load_only_params:
        optimizer.load_state_dict(state["optimizer"])

    return model, optimizer

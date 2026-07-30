import torch
import torch.nn as nn
import yaml
from munch import Munch

from runner.nodes.training.styletts.finetune.training.modules.asr.models import ASRCNN
from runner.nodes.training.styletts.finetune.training.modules.diffusion.diffusion import AudioDiffusionConditional
from runner.nodes.training.styletts.finetune.training.modules.diffusion.modules import StyleTransformer1d, Transformer1d
from runner.nodes.training.styletts.finetune.training.modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from runner.nodes.training.styletts.finetune.training.modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator
from runner.nodes.training.styletts.finetune.training.modules.encoders import TextEncoder, StyleEncoder
from runner.nodes.training.styletts.finetune.training.modules.hifigan import Decoder as HifiganDecoder
from runner.nodes.training.styletts.finetune.training.modules.istftnet import Decoder as IstftnetDecoder
from runner.nodes.training.styletts.finetune.training.modules.jdc import JDCNet
from runner.nodes.training.styletts.finetune.training.modules.predictors import ProsodyPredictor
from runner.nodes.training.styletts.finetune.training.state_dict_resize import merge_state_dict_with_dim0_resize

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
    if module_name not in _CHECKPOINT_DIM0_RESIZE_KEYS_BY_MODULE:
        return ckpt_sd
    return merge_state_dict_with_dim0_resize(
        model_module,
        ckpt_sd,
        _CHECKPOINT_DIM0_RESIZE_KEYS_BY_MODULE[module_name],
        error_scope=module_name,
        appended_source_index=0,
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


def load_F0_models(path):
    F0_model = JDCNet(num_class=1, seq_len=192)
    if path is not None:
        params = torch.load(path, map_location='cpu', weights_only=False)['net']
        F0_model.load_state_dict(params)
    F0_model.train()
    return F0_model


def _load_asr_config(path_or_dict):
    if isinstance(path_or_dict, dict):
        return path_or_dict["model_params"]
    with open(path_or_dict) as config_file:
        return yaml.safe_load(config_file)["model_params"]


def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    asr_model = ASRCNN(**_load_asr_config(ASR_MODEL_CONFIG))
    if ASR_MODEL_PATH is None:
        asr_model.train()
        return asr_model
    params = torch.load(ASR_MODEL_PATH, map_location='cpu', weights_only=False)['model']
    adapted = merge_state_dict_with_dim0_resize(
        asr_model,
        params,
        _ASR_N_TOKEN_DIM0_KEYS,
        error_scope="ASR",
        appended_source_index=0,
    )
    asr_model.load_state_dict(adapted)
    asr_model.train()
    return asr_model

def build_model(args, text_aligner, pitch_extractor, bert):
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    generator_checkpointing = bool(args.decoder.gradient_checkpointing)
    discriminators_checkpointing = bool(args.discriminators_checkpointing)
    
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
    
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim)
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim)
        
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
        embedding_mask_proba=args.diffusion.embedding_mask_proba,
        channels=args.style_dim*2,
        context_features=args.style_dim*2,
    )
    
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data,
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
        
            wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
       )
    
    return nets

def load_checkpoint(model, optimizer, path, load_only_params, ignore_modules):
    state = torch.load(path, map_location="cpu", weights_only=False)
    params = state["net"]
    for key in model:
        if key in ignore_modules:
            continue
        normalized_params = _maybe_normalize_module_prefix(model[key], params[key])
        adapted_params = _merge_checkpoint_state_with_dim0_resize(key, model[key], normalized_params)
        load_result = model[key].load_state_dict(adapted_params, strict=False)
        missing_keys = [
            item for item in load_result.missing_keys
            if not item.endswith("dummy_tensor")
        ]
        unexpected_keys = [
            item for item in load_result.unexpected_keys
            if not item.endswith("dummy_tensor")
        ]
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"checkpoint module {key} does not match: "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )
    for module in model.values():
        module.eval()

    if not load_only_params:
        optimizer.load_state_dict(state["optimizer"])

    return model, optimizer

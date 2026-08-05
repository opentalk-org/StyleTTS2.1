import logging

import torch
import torch.nn as nn
import yaml
from munch import Munch

from runner.nodes.training.styletts.finetune.training.modules.asr.models import ASRCNN
from runner.nodes.training.styletts.finetune.training.modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    WavLMDiscriminator,
)
from runner.nodes.training.styletts.finetune.training.modules.encoders import TextEncoder
from runner.nodes.training.styletts.finetune.training.modules.hifigan import Decoder as HifiganDecoder
from runner.nodes.training.styletts.finetune.training.modules.istftnet import Decoder as IstftnetDecoder
from runner.nodes.training.styletts.finetune.training.modules.jdc import JDCNet
from runner.nodes.training.styletts.finetune.training.modules.latent.alpha_flow import AlphaFlow
from runner.nodes.training.styletts.finetune.training.modules.latent.factorization import FactorizationHeads
from runner.nodes.training.styletts.finetune.training.modules.latent.prosody import (
    DurationPredictor,
    ProsodyDiscriminator,
    ProsodyPredictor,
    TVStyleEncoder,
)
from runner.nodes.training.styletts.finetune.training.modules.latent.rfsq import (
    ResidualFiniteScalarQuantizer,
)
from runner.nodes.training.styletts.finetune.training.modules.latent.voice import VoiceEncoder
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

logger = logging.getLogger(__name__)
_FACTORIZED_MODULES = frozenset({
    "alpha_flow",
    "duration_predictor",
    "factorization",
    "position_embedding",
    "prosody_encoder",
    "prosody_discriminator",
    "prosody_predictor",
    "quantizer",
    "duration_discriminator",
})
_OBSOLETE_ALPHA_FLOW_KEYS = frozenset({
    "denoiser.fixed_embedding.embedding.weight",
    "denoiser.fixed_feature.embedding.weight",
})


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


def _voice_checkpoint_source(params):
    for candidate in ("voice_encoder", "style_encoder"):
        if candidate not in params:
            continue
        names = {
            name[7:] if name.startswith("module.") else name
            for name in params[candidate]
        }
        if "mel_proj.weight" in names:
            return candidate
    return None


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
        
    reference_token_count = min(args.n_token, 178)
    text_encoder = TextEncoder(
        channels=args.hidden_dim,
        kernel_size=5,
        depth=args.n_layer,
        n_symbols=reference_token_count,
    )
    if args.n_token > reference_token_count:
        base_weight = text_encoder.embedding.weight.detach()
        appended = base_weight[0].expand(
            args.n_token - reference_token_count,
            -1,
        )
        text_encoder.embedding = nn.Embedding.from_pretrained(
            torch.cat((base_weight, appended), dim=0),
            freeze=False,
        )
    voice_encoder = VoiceEncoder(
        mel_dim=args.n_mels,
        text_dim=args.hidden_dim,
        voice_dim=args.style_dim,
    )
    wd = WavLMDiscriminator(
        slm_hidden=args.slm.hidden,
        slm_layers=args.slm.nlayers,
        initial_channel=args.slm.initial_channel,
    )
    mpd = MultiPeriodDiscriminator(
        gradient_checkpointing=discriminators_checkpointing,
    )
    msd = MultiResSpecDiscriminator(
        gradient_checkpointing=discriminators_checkpointing,
    )
    prosody_encoder = TVStyleEncoder(mel_dim=514)
    duration_predictor = DurationPredictor(max_dur=50)
    prosody_predictor = ProsodyPredictor()
    quantizer = ResidualFiniteScalarQuantizer(
        input_dim=512,
        latent_dim=args.prosody_quantizer.latent_dim,
        stages=args.prosody_quantizer.stages,
        levels=args.prosody_quantizer.levels,
        stage_dropout=args.prosody_quantizer.stage_dropout,
    )
    alpha_flow = AlphaFlow(
        text_dim=bert.config.hidden_size,
        style_dim=args.prosody_quantizer.latent_dim,
        style_scale=args.alpha_flow.get("style_scale", 1.0),
        transition_start=args.alpha_flow.transition_start,
        transition_end=args.alpha_flow.transition_end,
        temperature=args.alpha_flow.temperature,
        conditional_dropout=args.alpha_flow.conditional_dropout,
    )

    
    nets = Munch(
            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

            duration_predictor=duration_predictor,
            prosody_predictor=prosody_predictor,
            prosody_discriminator=ProsodyDiscriminator(mel_dim=514),
            duration_discriminator=ProsodyDiscriminator(mel_dim=513),
            decoder=decoder,
            text_encoder=text_encoder,
            position_embedding=nn.Embedding(512, 512),
            prosody_encoder=prosody_encoder,
            quantizer=quantizer,
            voice_encoder=voice_encoder,
            alpha_flow=alpha_flow,
            factorization=FactorizationHeads(
                language_count=args.language_count,
                content_dim=args.n_token,
                voice_dim=args.style_dim,
            ),
            text_aligner = text_aligner,
            pitch_extractor=pitch_extractor,

            mpd=mpd,
            msd=msd,
        
            wd=wd,
       )
    
    return nets

def load_checkpoint(model, optimizer, path, load_only_params, ignore_modules):
    state = torch.load(path, map_location="cpu", weights_only=False)
    params = state["net"]
    ignored = set(ignore_modules)
    quantizer_state = params["quantizer"] if "quantizer" in params else {}
    quantizer_keys = {
        name.removeprefix("module.")
        for name in quantizer_state
    }
    if "to_latent.weight" not in quantizer_keys:
        ignored.update(("quantizer", "alpha_flow"))
        logger.info(
            "initialized residual FSQ and continuous AlphaFlow; "
            "checkpoint contains a different prosody bottleneck"
        )
    reinitialized = set()
    for key in model:
        if key in ignored:
            reinitialized.add(key)
            continue
        source_key = _voice_checkpoint_source(params) if key == "voice_encoder" else key
        if key == "voice_encoder" and source_key is None:
            logger.info(
                "initialized voice encoder; checkpoint has no compatible weights"
            )
            reinitialized.add(key)
            continue
        if source_key not in params:
            if key in _FACTORIZED_MODULES:
                logger.info("initialized factorized module=%s parameters=%s", key, len(model[key].state_dict()))
                continue
            raise ValueError(f"checkpoint is missing unchanged module {key}")
        normalized_params = _maybe_normalize_module_prefix(model[key], params[source_key])
        if key == "alpha_flow":
            normalized_params = {
                name: value
                for name, value in normalized_params.items()
                if name not in _OBSOLETE_ALPHA_FLOW_KEYS
            }
        adapted_params = _merge_checkpoint_state_with_dim0_resize(key, model[key], normalized_params)
        if key == "factorization":
            current_keys = model[key].state_dict().keys()
            adapted_params = {
                name: value
                for name, value in adapted_params.items()
                if name in current_keys
            }
        load_result = model[key].load_state_dict(adapted_params, strict=False)
        missing_keys = [
            item for item in load_result.missing_keys
            if not item.endswith("dummy_tensor")
            and not (
                key == "alpha_flow"
                and item in {"style_scale", "style_scale_updates"}
            )
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

    resume_step = 0
    if not load_only_params:
        optimizer_state = [
            item for item in state["optimizer"]
            if item[0] not in reinitialized
        ]
        optimizer.load_state_dict(optimizer_state)
        # External base checkpoints use their own iteration counters. Only a
        # Studio checkpoint's explicit step belongs to this stage schedule.
        resume_step = int(state["step"]) if "step" in state else 0

    return model, optimizer, resume_step

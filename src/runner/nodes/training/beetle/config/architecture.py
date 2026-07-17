from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioConfig(StrictConfigModel):
    sample_rate: int = Field(gt=0)
    n_fft: int = Field(gt=0)
    win_length: int = Field(gt=0)
    hop_length: int = Field(gt=0)
    mel_channels: int = Field(gt=0)
    f_min: float = Field(ge=0)
    f_max: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_spectrum(self) -> "AudioConfig":
        if self.win_length > self.n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if self.f_max > self.sample_rate / 2:
            raise ValueError("f_max must not exceed Nyquist frequency")
        return self


class PosteriorEncoderConfig(StrictConfigModel):
    mel_channels: int = Field(gt=0)
    latent_channels: int = Field(gt=0)
    hidden_channels: int = Field(gt=0)
    downsample_rate: int = Field(default=2, gt=0)
    downsample_kernel_size: int = Field(default=4, gt=1)
    kernel_size: int = Field(gt=1)
    dilation_cycle: tuple[int, ...] = Field(min_length=1)
    cycles: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    log_scale_min: float
    log_scale_max: float

    @model_validator(mode="after")
    def validate_log_scale(self) -> "PosteriorEncoderConfig":
        if self.log_scale_min >= self.log_scale_max:
            raise ValueError("log_scale_min must be below log_scale_max")
        return self


class FeatureConfig(StrictConfigModel):
    latent_channels: int = Field(gt=0)
    upsample_rate: int = Field(default=2, gt=0)


class DecoderConfig(StrictConfigModel):
    latent_channels: int = Field(gt=0)
    hidden_channels: int = Field(gt=0)
    residual_channels: int = Field(gt=0)
    generator_channels: int = Field(gt=0)
    decode_block_count: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    f0_smoothing_kernel_sizes: tuple[int, ...] = Field(min_length=1)
    n_smoothing_kernel_sizes: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_smoothing_kernels(self) -> "DecoderConfig":
        fields = (
            ("f0_smoothing_kernel_sizes", self.f0_smoothing_kernel_sizes),
            ("n_smoothing_kernel_sizes", self.n_smoothing_kernel_sizes),
        )
        for field, kernels in fields:
            invalid = any(
                kernel < 0 or (kernel != 0 and kernel % 2 == 0)
                for kernel in kernels
            )
            if invalid:
                raise ValueError(
                    f"{field} must contain only zero or positive odd integers"
                )
        return self


class GeneratorConfig(StrictConfigModel):
    input_channels: int = Field(gt=0)
    frame_channels: int = Field(gt=0)
    temporal_channels: int = Field(gt=0)
    temporal_upsample_rate: int = Field(gt=0)
    temporal_upsample_kernel_size: int = Field(gt=1)
    resblock_kernel_sizes: tuple[int, ...] = Field(min_length=1)
    resblock_dilations: tuple[tuple[int, ...], ...] = Field(min_length=1)
    initial_frequency_bins: int = Field(gt=0)
    frequency_upsample_kernel_sizes: tuple[int, ...] = Field(min_length=1)
    harmonic_count: int = Field(gt=0)
    subbands: int = Field(gt=0)
    istft_n_fft: int = Field(gt=0)
    istft_hop_length: int = Field(gt=0)
    source_n_fft: int = Field(gt=0)
    source_hop_length: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_generator_geometry(self) -> "GeneratorConfig":
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilations):
            raise ValueError("each resblock kernel needs a dilation sequence")
        if len(self.frequency_upsample_kernel_sizes) != 3:
            raise ValueError("iSTFTNet2-MB requires three frequency upsampling stages")
        if self.temporal_channels % 4:
            raise ValueError("generator temporal_channels must be divisible by four")
        if self.istft_n_fft % self.subbands != 0:
            raise ValueError("istft_n_fft must be divisible by subbands")
        frequency_bins = self.initial_frequency_bins
        for kernel_size in self.frequency_upsample_kernel_sizes:
            frequency_bins = (frequency_bins - 1) * 2 - 2 + kernel_size
        if frequency_bins != self.istft_n_fft // 2 + 1:
            raise ValueError("frequency upsampling must match the iSTFT bins")
        if self.output_hop() % self.source_hop_length:
            raise ValueError("source_hop_length must divide output hop")
        if self.output_hop() // self.source_hop_length != self.temporal_upsample_rate:
            raise ValueError("harmonic-source frames must match temporal upsampling")
        return self

    def output_hop(self) -> int:
        return self.temporal_upsample_rate * self.istft_hop_length * self.subbands


class PhonemeConfig(StrictConfigModel):
    model_path: str = Field(min_length=1)
    projection_channels: int = Field(gt=0)
    cnn_hidden_channels: int = Field(gt=0)
    cnn_layers: int = Field(gt=0)
    cnn_kernel_size: int = Field(gt=1)
    dropout: float = Field(ge=0, lt=1)


class ContextConfig(StrictConfigModel):
    hidden_channels: int = Field(gt=0)
    output_channels: int = Field(gt=0)
    layers: int = Field(gt=0)
    kernel_size: int = Field(gt=1)
    dropout: float = Field(ge=0, lt=1)


class EmbeddingEncoderConfig(StrictConfigModel):
    input_channels: int = Field(gt=0)
    hidden_channels: int = Field(gt=0)
    embedding_channels: int = Field(gt=0)
    attention_channels: int = Field(gt=0)
    layers: int = Field(gt=0)
    speaker_classes: int = Field(gt=1)


class ConditionDropoutConfig(StrictConfigModel):
    phoneme_embedding: float = Field(ge=0, le=1)
    style: float = Field(ge=0, le=1)
    voice: float = Field(ge=0, le=1)
    pooled_phoneme: float = Field(ge=0, le=1)
    pre_text: float = Field(ge=0, le=1)
    post_text: float = Field(ge=0, le=1)
    pre_audio: float = Field(ge=0, le=1)
    post_audio: float = Field(ge=0, le=1)


class ConditioningConfig(StrictConfigModel):
    common_channels: int = Field(gt=0)
    boundary_k_min: int = Field(ge=1, le=32)
    boundary_k_max: int = Field(ge=1, le=32)
    concat_layers: tuple[int, ...] = Field(min_length=1)
    dropout: ConditionDropoutConfig

    @model_validator(mode="after")
    def validate_boundary(self) -> "ConditioningConfig":
        if self.boundary_k_min > self.boundary_k_max:
            raise ValueError("boundary_k_min must not exceed boundary_k_max")
        return self


class DurationFlowConfig(StrictConfigModel):
    condition_channels: int = Field(gt=0)
    hidden_channels: int = Field(gt=0)
    flow_count: int = Field(gt=0)
    posterior_flow_count: int = Field(gt=0)
    convolution_layers: int = Field(gt=0)
    kernel_size: int = Field(gt=1)
    spline_bins: int = Field(gt=1)
    spline_tail_bound: float = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)


class LatentFlowConfig(StrictConfigModel):
    latent_channels: int = Field(gt=0)
    hidden_channels: int = Field(gt=0)
    condition_channels: int = Field(gt=0)
    time_embedding_channels: int = Field(gt=0)
    layer_count: int = Field(gt=0)
    kernel_size: int = Field(gt=1)
    dilation_cycle: tuple[int, ...] = Field(min_length=1)
    minimum_steps: int = Field(gt=1)
    base_case_probability: float = Field(gt=0, lt=1)
    ema_decay: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_dyadic_steps(self) -> "LatentFlowConfig":
        if self.minimum_steps & (self.minimum_steps - 1):
            raise ValueError("minimum_steps must be a power of two")
        return self


class AlignerConfig(StrictConfigModel):
    checkpoint_asset_id: str = Field(min_length=1)
    hidden_channels: int = Field(gt=0)
    blank_id: int = Field(ge=0)


class TextEncoderConfig(StrictConfigModel):
    pretrained_model: str = Field(min_length=1)
    hidden_channels: int = Field(gt=0)
    projection_channels: int = Field(gt=0)


class ArchitectureConfig(StrictConfigModel):
    posterior: PosteriorEncoderConfig
    feature: FeatureConfig
    decoder: DecoderConfig
    generator: GeneratorConfig
    phoneme_token_count: int = Field(gt=1)
    phoneme: PhonemeConfig
    context: ContextConfig
    embeddings: EmbeddingEncoderConfig
    conditioning: ConditioningConfig
    duration_flow: DurationFlowConfig
    latent_flow: LatentFlowConfig
    aligner: AlignerConfig
    text_encoder: TextEncoderConfig

    @model_validator(mode="after")
    def validate_audio_rates(self) -> "ArchitectureConfig":
        if self.posterior.downsample_rate != 2:
            raise ValueError("posterior downsample_rate must equal 2")
        if self.posterior.downsample_kernel_size != 4:
            raise ValueError("posterior downsample_kernel_size must equal 4")
        if self.feature.upsample_rate != 2:
            raise ValueError("feature upsample_rate must equal 2")
        return self

    @model_validator(mode="after")
    def validate_channels(self) -> "ArchitectureConfig":
        latent = self.posterior.latent_channels
        if self.feature.latent_channels != latent:
            raise ValueError("feature latent_channels must match posterior latent_channels")
        if self.decoder.latent_channels != latent:
            raise ValueError("decoder latent_channels must match posterior latent_channels")
        if self.embeddings.input_channels != latent:
            raise ValueError("embedding input_channels must match posterior latent_channels")
        if self.latent_flow.latent_channels != latent:
            raise ValueError("latent-flow latent_channels must match posterior latent_channels")
        if self.generator.input_channels != self.decoder.generator_channels:
            raise ValueError("generator input_channels must match decoder generator_channels")
        condition = self.conditioning.common_channels
        if self.duration_flow.condition_channels != condition:
            raise ValueError("duration-flow condition_channels must match conditioning")
        if self.latent_flow.condition_channels != condition:
            raise ValueError("latent-flow condition_channels must match conditioning")
        if max(self.conditioning.concat_layers) >= self.latent_flow.layer_count:
            raise ValueError("conditioning concat layer is outside latent flow")
        return self

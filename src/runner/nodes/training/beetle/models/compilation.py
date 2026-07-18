from torch import nn

from .model import Stage1Models


def compile_stage1(models: Stage1Models) -> None:
    modules: tuple[nn.Module, ...] = (
        models.audio_encoder,
        models.feature_linear,
        models.decoder.encode,
        *models.decoder.decode,
    )
    for module in modules:
        module.compile()

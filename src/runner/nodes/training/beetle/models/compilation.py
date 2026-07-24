import torch
from torch import nn

from .model import AcousticModels


def compile_acoustic(models: AcousticModels) -> None:
    modules: tuple[nn.Module, ...] = (
        models.feature_linear,
        models.decoder,
        models.generator,
    )
    for module in modules:
        # Module.compile targets nn.Module._call_impl, so independently compiled
        # modules share one Dynamo code cache and exhaust its recompile budget.
        # Compiling each concrete forward keeps graph ownership with the model.
        module.forward = torch.compile(
            module.forward,
            dynamic=False,
        )

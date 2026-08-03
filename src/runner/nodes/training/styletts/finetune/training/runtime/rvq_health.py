from torch import Tensor


COLLAPSE_THRESHOLD = 1e-3
COLLAPSE_WINDOW = 20


def check_rvq_health(history: list[float], style_std: Tensor) -> None:
    value = float(style_std.detach().item())
    history.append(value)
    if len(history) > COLLAPSE_WINDOW:
        del history[:-COLLAPSE_WINDOW]
    if len(history) == COLLAPSE_WINDOW and max(history) < COLLAPSE_THRESHOLD:
        raise RuntimeError(
            "StyleTTS-ZS RVQ collapsed: quantized style batch standard "
            f"deviation stayed below {COLLAPSE_THRESHOLD} for "
            f"{COLLAPSE_WINDOW} steps"
        )

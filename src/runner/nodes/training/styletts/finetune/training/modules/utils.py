import torch

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        with torch.no_grad():
            if hasattr(m, "weight_v") and hasattr(m, "weight_g"):
                m.weight_v.normal_(mean, std)
                dimensions = tuple(range(1, m.weight_v.ndim))
                norm = torch.linalg.vector_norm(
                    m.weight_v,
                    dim=dimensions,
                    keepdim=True,
                )
                m.weight_g.copy_(norm)
            else:
                m.weight.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def checkpoint_with_mixed_precision(function, *args):
    if not torch.is_autocast_enabled("cuda"):
        return torch.utils.checkpoint.checkpoint(
            function,
            *args,
            use_reentrant=False,
        )
    autocast_dtype = torch.get_autocast_dtype("cuda")
    def _wrapped(*inner_args):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            return function(*inner_args)

    return torch.utils.checkpoint.checkpoint(
        _wrapped,
        *args,
        use_reentrant=False,
    )

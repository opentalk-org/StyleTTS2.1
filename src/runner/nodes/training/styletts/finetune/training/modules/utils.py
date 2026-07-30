import torch

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def checkpoint_with_mixed_precision(function, *args):
    if not torch.is_autocast_enabled("cuda"):
        return torch.utils.checkpoint.checkpoint(
            function,
            *args,
            use_reentrant=True,
        )
    autocast_dtype = torch.get_autocast_dtype("cuda")
    def _wrapped(*inner_args):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            return function(*inner_args)

    return torch.utils.checkpoint.checkpoint(
        _wrapped,
        *args,
        use_reentrant=True,
    )

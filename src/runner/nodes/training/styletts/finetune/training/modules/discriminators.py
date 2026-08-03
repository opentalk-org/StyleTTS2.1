import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm

from ..profiling import profiling_fn
from .utils import checkpoint_with_mixed_precision, get_padding

LRELU_SLOPE = 0.1

def stft(x, fft_size, hop_size, win_length, window):
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window,
            return_complex=True)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.abs(x_stft).transpose(2, 1)

class SpecDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        use_spectral_norm=False,
        gradient_checkpointing=False,
    ):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer(
            "window",
            getattr(torch, window)(win_length),
            persistent=False,
        )
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))
        self.gradient_checkpointing = gradient_checkpointing
        self.dummy_tensor = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _conv_block(self, y, i, dummy):
        y = self.discriminators[i](y)
        return F.leaky_relu(y, LRELU_SLOPE, inplace=True)

    def _out_block(self, y, dummy):
        return self.out(y)

    def forward(self, y, return_features=True):

        fmap = []
        y = y.squeeze(1)
        y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        y = y.unsqueeze(1)
        dummy = self.dummy_tensor
        for i in range(len(self.discriminators)):
            if self.gradient_checkpointing:
                y = checkpoint_with_mixed_precision(self._conv_block, y, i, dummy)
            else:
                y = self._conv_block(y, i, dummy)
            if return_features:
                fmap.append(y)

        if self.gradient_checkpointing:
            y = checkpoint_with_mixed_precision(self._out_block, y, dummy)
        else:
            y = self._out_block(y, dummy)
        if return_features:
            fmap.append(y)

        return torch.flatten(y, 1, -1), fmap

class MultiResSpecDiscriminator(torch.nn.Module):

    def __init__(self,
                 gradient_checkpointing=False,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):

        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window, gradient_checkpointing=gradient_checkpointing),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window, gradient_checkpointing=gradient_checkpointing),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window, gradient_checkpointing=gradient_checkpointing)
            ])

    def forward(self, y, y_hat, return_features=True):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        batch_size = y.shape[0]
        for discriminator in self.discriminators:
            with profiling_fn(f"stft_{discriminator.fft_size}"):
                if discriminator.dummy_tensor.requires_grad:
                    with profiling_fn("real_and_generated"):
                        combined = torch.cat((y, y_hat), dim=0)
                        scores, feature_maps = discriminator(combined, return_features)
                        y_d_r, y_d_g = scores.split(batch_size)
                        fmap_r = []
                        fmap_g = []
                        if return_features:
                            for feature_map in feature_maps:
                                real_map, generated_map = feature_map.split(batch_size)
                                fmap_r.append(real_map)
                                fmap_g.append(generated_map)
                else:
                    with profiling_fn("real"):
                        y_d_r, fmap_r = discriminator(y, return_features)
                    with profiling_fn("generated"):
                        y_d_g, fmap_g = discriminator(y_hat, return_features)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, gradient_checkpointing=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.gradient_checkpointing = gradient_checkpointing
        self.dummy_tensor = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _conv_block(self, x, i, dummy):
        x = self.convs[i](x)
        return F.leaky_relu(x, LRELU_SLOPE, inplace=True)

    def _post_block(self, x, dummy):
        return self.conv_post(x)

    def forward(self, x, return_features=True):
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        dummy = self.dummy_tensor
        for i in range(len(self.convs)):
            if self.gradient_checkpointing:
                x = checkpoint_with_mixed_precision(self._conv_block, x, i, dummy)
            else:
                x = self._conv_block(x, i, dummy)
            if return_features:
                fmap.append(x)
        if self.gradient_checkpointing:
            x = checkpoint_with_mixed_precision(self._post_block, x, dummy)
        else:
            x = self._post_block(x, dummy)
        if return_features:
            fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, gradient_checkpointing=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, gradient_checkpointing=gradient_checkpointing),
            DiscriminatorP(3, gradient_checkpointing=gradient_checkpointing),
            DiscriminatorP(5, gradient_checkpointing=gradient_checkpointing),
            DiscriminatorP(7, gradient_checkpointing=gradient_checkpointing),
            DiscriminatorP(11, gradient_checkpointing=gradient_checkpointing),
        ])

    def forward(self, y, y_hat, return_features=True):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        batch_size = y.shape[0]
        for discriminator in self.discriminators:
            with profiling_fn(f"period_{discriminator.period}"):
                if discriminator.dummy_tensor.requires_grad:
                    with profiling_fn("real_and_generated"):
                        combined = torch.cat((y, y_hat), dim=0)
                        scores, feature_maps = discriminator(combined, return_features)
                        y_d_r, y_d_g = scores.split(batch_size)
                        fmap_r = []
                        fmap_g = []
                        if return_features:
                            for feature_map in feature_maps:
                                real_map, generated_map = feature_map.split(batch_size)
                                fmap_r.append(real_map)
                                fmap_g.append(generated_map)
                else:
                    with profiling_fn("real"):
                        y_d_r, fmap_r = discriminator(y, return_features)
                    with profiling_fn("generated"):
                        y_d_g, fmap_g = discriminator(y_hat, return_features)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
class WavLMDiscriminator(nn.Module):
    def __init__(self, slm_hidden=768, 
                 slm_layers=13, 
                 initial_channel=64, 
                 use_spectral_norm=False):
        super(WavLMDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.pre = norm_f(Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
            norm_f(nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)),
            norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
        ])

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))
        
    def forward(self, x):
        x = self.pre(x)
        
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x

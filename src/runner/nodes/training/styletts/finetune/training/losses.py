import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel

from .profiling import profiling_fn

class SpectralConvergengeLoss(torch.nn.Module):
    def __init__(self):
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class STFTLoss(torch.nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=fft_size, win_length=win_length, hop_length=shift_size, window_fn=window)

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)    
        return sc_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window=torch.hann_window):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
    
    
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)

    return loss


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss


class GeneratorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(GeneratorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat, discriminator):
        real_scores, generated_scores, real_maps, generated_maps = discriminator(
            y,
            y_hat,
        )
        feature = feature_loss(real_maps, generated_maps)
        adversarial, _ = generator_loss(generated_scores)
        relative = generator_TPRLS_loss(real_scores, generated_scores)
        return (feature + adversarial + relative).mean()

class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(DiscriminatorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat, discriminator):
        real_scores, generated_scores, _, _ = discriminator(
            y,
            y_hat,
            return_features=False,
        )
        adversarial = discriminator_loss(real_scores, generated_scores)
        relative = discriminator_TPRLS_loss(real_scores, generated_scores)
        return (adversarial + relative).mean()

   
    
class WavLMLoss(torch.nn.Module):

    def __init__(
        self,
        model,
        wd,
        model_sr,
        slm_sr=16000,
    ):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wavlm.requires_grad_(False)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
     
    def forward(self, wav, y_rec):
        with torch.no_grad():
            with profiling_fn("wavlm.reference_embedding"):
                wav_16 = self.resample(wav)
                wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
        with profiling_fn("wavlm.generated_embedding"):
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

        with profiling_fn("wavlm.feature_loss"):
            floss = 0
            for er, eg in zip(wav_embeddings, y_rec_embeddings):
                floss += torch.mean(torch.abs(er - eg))
        
        return floss.mean()

    def generator(self, y_rec):
        with profiling_fn("wavlm.generator_embedding"):
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        with profiling_fn("wavlm.generator_discriminator"):
            y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1-y_df_hat_g)**2)
        
        return loss_gen
    
    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            with profiling_fn("wavlm.discriminator_embeddings"):
                wav_16 = self.resample(wav)
                wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
                y_rec_16 = self.resample(y_rec)
                y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

                y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
                y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        with profiling_fn("wavlm.discriminator_network"):
            y_d_rs = self.wd(y_embeddings)
            y_d_gs = self.wd(y_rec_embeddings)
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
                        
        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            with profiling_fn("wavlm.discriminator_forward_embedding"):
                wav_16 = self.resample(wav)
                wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
                y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        with profiling_fn("wavlm.discriminator_forward_network"):
            y_d_rs = self.wd(y_embeddings)
        
        return y_d_rs

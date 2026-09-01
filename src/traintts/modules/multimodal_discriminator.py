import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchaudio.models import Conformer


class WavLMDiscriminator(nn.Module):
    """Conformer discriminator over WavLM and every acoustic decoder input."""

    def __init__(
        self,
        slm_hidden: int = 768,
        slm_layers: int = 13,
        hidden_dim: int = 512,
        style_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.wavlm_projection = nn.Conv1d(
            slm_hidden * slm_layers,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.prosody_projection = nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1)
        self.style_projection = nn.Linear(style_dim, hidden_dim)
        self.separator = nn.Embedding(1, hidden_dim)
        self.conformers = nn.ModuleList(
            Conformer(
                hidden_dim,
                num_heads,
                hidden_dim * 2,
                1,
                15,
                use_group_norm=True,
            )
            for _ in range(num_layers)
        )
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        wavlm: Tensor,
        aligned_text: Tensor,
        global_style: Tensor,
        pitch: Tensor,
        energy: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        condition_length = aligned_text.size(-1)
        prosody = torch.stack((pitch, energy), dim=1)
        prosody = F.interpolate(
            prosody,
            size=condition_length,
            mode="linear",
            align_corners=False,
        )
        separator = self.separator(
            torch.zeros(wavlm.size(0), dtype=torch.long, device=wavlm.device)
        ).unsqueeze(-1)
        tokens = torch.cat(
            (
                self.wavlm_projection(wavlm),
                separator,
                self.style_projection(global_style).unsqueeze(-1),
                self.prosody_projection(prosody),
                separator,
                aligned_text,
            ),
            dim=-1,
        ).transpose(1, 2)
        lengths = torch.full(
            (tokens.size(0),),
            tokens.size(1),
            dtype=torch.long,
            device=tokens.device,
        )
        feature_maps = []
        for conformer in self.conformers:
            tokens, _ = conformer(tokens, lengths)
            feature_maps.append(tokens)
        return self.score(tokens).squeeze(-1), feature_maps


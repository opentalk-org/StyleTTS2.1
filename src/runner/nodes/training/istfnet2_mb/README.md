# iSTFTNet2-MB training

This package trains the multi-band iSTFTNet2 generator described in Section
4.3 of Kaneko et al., *iSTFTNet2: Faster and More Lightweight iSTFT-Based
Neural Vocoder Using 1D-2D CNN*.

The complete paper generator is contained in `model.py`. Its fixed contract is:

- 80-channel log-mel input at a 256-sample hop;
- one `C4` temporal upsampling stage with HiFi-GAN V2 channels;
- three 1-D MRF branches with kernels 3, 7, and 11, concatenated;
- conversion to a four-bin frequency representation;
- three multi-band 2-D ShuffleBlocks with doubled 2-D channels and `C`
  active-branch expansion;
- frequency transposed convolutions producing 8, 16, then 33 bins;
- eight output channels representing magnitude and phase for four subbands;
- four 64-point, hop-16 iSTFTs followed by four-band PQMF synthesis.

The generator has 827,048 parameters, corresponding to the paper's reported
0.83M. Training intentionally uses the HiFTNet loss system: multi-period and
multi-resolution spectral discriminators, least-squares adversarial loss,
TPRLS, feature matching weighted by two, and log-mel L1 weighted by 45.
Every 500 steps, validation logs mel and STFT spectrogram comparisons plus
`pred.wav` and `gt.wav` for each of 16 held-out samples.

Run through the repository environment:

```bash
./nix/run-venv.sh python -m runner.nodes.training.istfnet2_mb.train \
  --dataset-id <uuid> \
  --output-dir .data/istftnet2-mb/checkpoints \
  --cache-dir .data/istftnet2-mb/audio-cache
```

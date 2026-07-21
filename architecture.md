# ARCHITECTURE
all modules:
- AudioEncoder - similar to pipertts audio encoder
- Generator - based on styletts2 base generator and customized for istftnet2-MB - multiband
- Decoder - based on istftnet2 architecture + parts of original styletts2 f0/N conditioning, 300 hop size, multiband
- DurationPredictor - normalizing flow similar to pipertts
- LatentFlowModel - diffusion forcing + shorctut models + flow matching + cnn. so it accept x_t, t - (noise level 0.0 - 1.0 different per every token), d (step size), cond (vector by linear projection of all inputs (embeddings, style, voice vectors, etc.))
- PhonemeAligner - pretrained model (the same as styletts2)
- PhonemeEncoder - phoneme bert initialization (multilingual bert - the same albert as original styletts2)
- LatentPhonemeEncoder - few cnn layers (to decide of architecture)
- DurationPhonemeEncoder - few cnn layers (to decide of architecture) 
- ContextPhonemeEncoder - few cnn layers (to decide of architecture)
- ContextAudioEncoder - few cnn layers (to decide of architecture)
- TextEncoder - multilingual bert - normal text

- StyleEncoder - similar to original styletss2 but works on audio encoder output instead waveform so customized and resized for that
- VoiceEncoder - the same as StyleEncoder

- F0Extractor - to get ground truth f0, the same as styletts2, N you can get similar to styletts2 (log norm)
- FeatureLinear (linear layer, z -> f0, N)

The flow model is cnn, it accept input x_t, t (0.0 - 1.0) per every token, 

phonemes -PhonemeEncoder> -LatentPhonemeEncoder/DurationPhonemeEncoder> text embeddings + text pool vector (mean)

audio mel -AudioEncoder> z -VoiceEncoder> voice embedding
audio mel -AudioEncoder> z -StyleEncoder 2> style embedding

voice prompt -TextEncoder> voice embedding
style prompt -TextEncoder> style embedding

pre and post encoders for audio: 
phonemes -PhonemeEncoder> vectors -ContextPhonemeEncoder> mean; 
audio -AudioEncoder> vectors -ContextPhonemeEncoder> mean

phoneme embeddings + style vector + voice vector + text pool vector + pre/post vectors (apply to first/last k phonemes) -DurationPredictor> durations

phoneme embeddings duplicated by alignment matrix + style vector + voice vector + phoneme pool vector + pre/post vectors (apply to first / last k phonemes) -LatentFlowModel> z

each conditioning (phoneme embedding, style, voice, phoneme pool, pre/post vectors have drop chance during training, so cfg is possible), the model is conditioned on single vector, so all features first pass through linear layer. noise -> z; conditioning works by conv channel concat at some layer + adaLN-Zero.

audio mel -AudioEncoder> z, z_std -Linear> z + [f0, N] -Decoder> h + [f0] -Generator> audio

z -Linear> [f0, N]

losses:

- mel/full spectogram losses at different scales (recon_loss)
- GAN wave unet discriminator 9600 samples (gen_loss, disc_loss)
- f0_loss, N_loss - f0 mse loss, N mse loss, f0 go from f0 extraction model, N is the same as styletts2 so from log of norm of mel spectogram or something like that.
- KL loss for encoder latent (kl_loss)
- flow matching losses for flow model 1 and 2 (shortcut models and diffusion forcing setup) (dur_flow_loss - log duration prediction, main_flow_loss)
- phoneme aligner loss (align_s2s_loss - phoneme/time alignment loss cross entropy, align_mono_loss - hard alignment loss so soft and hard match, align_ctc_loss - obvious)
- slm discriminator (finetuning - slm_disc_loss, slm_gen_loss)

- voice encoder require that 2 audio of the same person have equal voice vector and different have the same (contrastive loss, GE2E loss) + ugmented time stretch, pitch shift, audio gain.
- style encoder (the same audio, different cuts of it with distance weight - the same embedding, different random audio == different vector - contrastive loss, GE2E loss)
- style encoder GAN (it classify if 2 vectors come from the same speaker, negative gradient from classifier so it doesn't contain info about speaker)
- mean f0, N, std of f0, std of N linear classifier from style vector.
- reencoding consistency loss (style -diffusion> generated latent audio -StyleEncoder> style)

- voice / style loss, mse: (audio style - text style)^2 / maybe diffusion loss - two losses: voice_loss and style_loss - different training stage, can be trained after model.

Stages:
- AudioEncoder + Generator + [f0, N] + Decoder training + GANs
- everything except above (up to latents from audioencoder training) - AudioEncoder, Generator, Decoder, Gans frozen / non used
- e2e finetuning
- prompt -> voice/style encoder training (skip for now)

# Two-BERT G2P with frozen-ASR PPO

This isolated experiment initializes two copies of the custom multilingual PL-BERT bucket asset: a bidirectional text-byte encoder and a causal phoneme decoder with cross-attention. It first learns text-to-phoneme generation from the original Hetzner Storage Box PL-BERT parquets, then applies PPO to backend text/audio rows using the frozen StyleTTS checkpoint aligner as reward.

The PPO objective uses the sampled-token likelihood ratio, `clip(ratio, 1-epsilon, 1+epsilon)`, a frozen supervised-policy KL penalty, and an entropy bonus. The aligner is frozen, and PPO maximizes the negative raw combined ASR loss without normalizing it to `[0, 1]`.

All commands must run through the development shell:

```bash
nix develop -c python -m experiments.bert_g2p_asr_ppo.cli download --train-files 1 --validation-files 1
nix develop -c python -m experiments.bert_g2p_asr_ppo.cli sft --steps 10000 --batch-size 4 --validation-interval 250 --validation-batches 16
nix develop -c python -m experiments.bert_g2p_asr_ppo.cli ppo CHECKPOINT MLFLOW_RUN_ID --steps 500 --batch-size 4 --validation-interval 25 --validation-batches 4
```

Or run both stages under one MLflow run:

```bash
nix develop -c python -m experiments.bert_g2p_asr_ppo.cli pipeline --sft-steps 1000 --ppo-steps 500 --batch-size 4
```

Defaults pin the separate custom PL-BERT asset `f4109860-a92d-47a9-9717-1d2f2febac4b`, the current StyleTTS aligner checkpoint `4a3f31d4-6fda-463f-9f8a-f9d85cbf84a9`, and backend training dataset `e25b39ac-3400-4f9f-9fac-3e9c94e1a92b`.

MLflow experiment: `bert_g2p_asr_ppo`. Supervised metrics include `sft/loss` and periodic `sft/validation_loss`. PPO uses a deterministic 90/10 backend audio split and logs raw `ppo/asr_loss` plus held-out `ppo/validation_asr_loss` and `ppo/validation_kl`.

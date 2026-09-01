# PL-BERT + BiLSTM RNN-T G2P

This experiment initializes and freezes the text encoder from the custom multilingual PL-BERT asset, contextualizes UTF-8 text bytes with BERT, and passes those representations through a trainable bidirectional LSTM. An autoregressive LSTM prediction network and additive joiner decode phonemes with the RNN-T objective.

Training uses 16 multilingual PL-BERT parquet shards and preserves each example's PL-BERT language ID. Validation reports RNN-T loss, phoneme error rate, aligned token accuracy, token F1, and exact sequence match. Configuration, metrics, and checkpoints are logged to the `bert_bilstm_rnnt` MLflow experiment.

```bash
nix develop -c python -m experiments.bert_bilstm_rnnt.cli download --train-files 1 --validation-files 1
nix develop -c python -m experiments.bert_bilstm_rnnt.cli train --steps 10000 --batch-size 2 --validation-interval 250 --validation-batches 16
```

The default PL-BERT checkpoint and data locations are inherited from the sibling experiment. Model and training settings live in `config.py`. Continue model weights from a saved step with `train --checkpoint PATH`; add `--resume-optimizer` when the checkpoint has the same trainable parameter groups. Checkpoint and MLflow steps advance from the loaded global step.

Training clips the global gradient norm to `1.0`. Non-finite loss or gradient batches are skipped before the optimizer update and counted in MLflow as `train/nonfinite_loss_skips` or `train/nonfinite_gradient_skips`, preventing one numerically invalid multilingual batch from corrupting later checkpoints.

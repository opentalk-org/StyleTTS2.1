# Beetle Language Conditioning Design

## Goal

Represent every configured language with one learned vector and provide the same
conditioning sources to the duration predictor and latent flow at their native
token rates.

## Configuration and identity

The architecture configuration owns an explicit ordered language vocabulary and
the language-vector width. Language names must be non-empty and unique. Their
configured order defines embedding IDs and is included in the existing config
fingerprint, so a resumed checkpoint cannot silently reinterpret IDs.

Database audio rows already store a nullable language string. Segment reference
queries will expose it, and the Beetle index will carry it into fetched examples.
Rows with a missing language or a language absent from the configured vocabulary
are ineligible for every training stage. The eligibility report will count these
rows separately. Language participates in the dataset fingerprint.

The collator resolves language strings to a `[B]` integer tensor. This keeps
mixed-language batches ordinary and avoids dataset-derived ID assignment.

## Shared representation

Stage 2 owns one `nn.Embedding`: each configured language ID selects one learned
vector. The same selected vector feeds both consumers; there are no one-hot
language features or consumer-specific language embeddings.

Both consumers receive these sources:

- phoneme features
- pooled phoneme embedding
- style vector
- voice vector
- pre/post text context
- pre/post audio context
- language vector

Each conditioning source retains independent per-item dropout. Dropped sources
are zero vectors, allowing different source combinations within a batch. The
language dropout probability is configured independently and defaults to 0.01.
One set of keep decisions is applied to both token rates for a training example.

## Duration predictor

Duration conditioning is built at phoneme rate. Duration-encoder phoneme features
remain one value per phoneme. Vector and context sources are repeated across the
phoneme tokens. After dropout, all sources are concatenated and passed to the
duration predictor.

The duration predictor already begins its condition encoder with a `1x1 Conv1d`,
which is a learned linear projection applied per phoneme token. Its configured
input width will match the complete concatenated condition width; no extra
duration-specific projection model is introduced.

## Latent flow

Latent-flow conditioning is built at posterior-latent rate. Phoneme features are
expanded through the hard alignment and pooled to the posterior rate. The same
vector and context values used by duration conditioning are repeated across the
latent tokens.

Every source, including language, has its own learned `1x1` projection into the
common condition width. The projected sources are summed for AdaLN modulation and
concatenated at the configured latent-flow layers. The latent-flow concatenation
width therefore accounts for nine sources instead of eight.

## Model accounting and documentation

The language embedding and inference-time conditioning parameters count toward
the full inference parameter total. Training-only exclusions remain unchanged.
The implementation will rerun the parameter and latent-to-audio GFLOPs checks and
report any configured limit violation. `main.md` and the default configuration
will describe the language vocabulary, rejection policy, dropout, and both
conditioning paths.

## Verification

Temporary tests will cover ordered ID resolution, missing and unknown language
rejection, mixed-language collation, independent dropout, rate-specific condition
shapes, shared language vectors, and both model inputs. The full temporary Beetle
suite, Ruff, compile checks, parameter counting, and latent-to-audio complexity
measurement will run after implementation.

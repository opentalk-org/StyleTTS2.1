<start/>
<end/>

<p/> (,)
<s/> (.)
<break time="400ms/0.4s/low/medium/strong/300ms..500ms"/> low/medium/strong is alias for x ms, 300ms..500ms - range
<mark name=""/>

<spell><spell> - spell text
<normalize><normalize> - model normalization
<phoneme ph="phonemes">word word word<phoneme> - use direct phoneme
<voice id="" blend=0.5 prompt="" stability=0.5>text</voice> - switch voice, blend 0.0-1.0 (default 0.5), prompt not compatible with id. can be voice sample but through extra api you receive voice id to use. stability (hidden alpha, beta, scalling embedding and heurestics)
<lang value="english"><lang> - token + control
<prosody speed="0.8" pitch="-2st" volume="-6db" duration=2.0s></prosody> - code control, speed - obvious, pitch - f0, volume - obvious, duration not compatible with speed, allow for forced duration.
<style value="sad/neutral/excited" prompt="" power=0.5></style> - only steer generation, emotions
<emphasis> - emphasis of word
<do value="laught/cry/breath/door slam"> - get own tokens
<turn speaker="someone"> (really other speakers, the same speakers don't get turns tokens)


## code only:
<break>
<mark>
<spell>
<normalize> - other model
<phoneme>
<prosody>

## model:
### tokens:
<start>
<end>
<p/>
<s/>
<do>

### tokens control (added to multiple tokens)
<style></style>

### tokens control + tokens
<lang>

### other
<voice>

### <normalize> (model)
- dates: 01/05/2026, Jan. 5, 2026-06-04
- numbers: positive numbers, negative numbers, decimals, large numbers, 1B, 2B
- currency: PLN, 1 USD, 20.5$, €9, £1.2M, ¥500
- ordinals: 1st, 2nd, 22nd, 100th
- fractions: 1/2, 4/5, ¾, 2 1/2
- Roman numerals: II, IV, XVI, Louis XIV
- times: 14:30, 2:30 PM, 08:00
- durations: 2h 30m, 01:45:00
- time zones: EST, UTC+2, CET
- percentages: 15%, 0.5%
- units and measurements: 5 kg, 10 km, 72°F, 3 tbsp, ms, µm, kWh, Gbps
- addresses: 123 Main St., Apt 4B
- postal codes: 00-001, SW1A 1AA, 90210
- coordinates/GPS: 52.2297° N, 21.0122° E, N 52°13′48″
- phone numbers: +48 123 456 789
- email addresses: indoxer.mk@gmail.com
- URLs/domains: https://openai.com, www.example.org
- hashtags: #AI, #ThrowBacks
- handles/usernames: @john_doe
- filenames/file paths: report.pdf, report_v2.pdf, /usr/bin
- abbreviations: Dr., Mr., Ave., etc., Prof., Hon., Rev., Sr.
- acronyms/initialisms: NASA, FBI, HTML, ICU, SLA, KPI
- contractions: don't, i'am, we'll, I’m, can’t
- symbols: +, =, -, &, @, /, *, #
- punctuation behavior: ., ?, !, …, —, –, quotes, parentheses, brackets
- bullet points/list markers: •, -, *, 1., a), (iii)
- math expressions: x², 3 + 4 = 7, √9, E=mc², a/b, f(x)
- inequalities/math symbols: ≤, ≥, ≠, ≈
- Greek letters: α, β, Δ, π
- superscripts/subscripts: x₂, m³, 10⁶
- chemical formulas: H2O, CO2, NaCl, CH₃COOH, C6H12O6
- medical/scientific notation: mg/dL, BP, HbA1c, 120/80, 5.6 mmol/L
- dosages: 2 tabs BID, 5 mg q.d.
- ranges: 5–10, Mon–Fri, pages 2-4, 20–25°C
- scores/ratios: 3–2, 16:9, 1:4, 10–2, 3-for-4
- versions: v2.1.0, Python 3.12
- product/model names: iPhone 15 Pro, RTX 4090
- emoji/emoticons: 🙂, ❤️, :)
- kaomoji/ASCII art: ¯\_(ツ)_/¯, (╯°□°）╯︵ ┻━┻
- foreign words/loanwords: croissant, schadenfreude
- legal references: § 12, Art. 5, 18 U.S.C. § 1030
- scripture references: John 3:16, Quran 2:255
- academic citations/references: Smith et al. 2020, [12]
- footnotes/endnotes: word¹, see note 4
- music notation: C#, Bb, 4/4
- chess notation: Nf3, O-O, Qxe5+
- betting odds: +150, 5/1, -110
- license plates: KR 12345
- flight numbers: LO388, BA2490
- train/bus routes: M2, Bus 175, S8
- room/building labels: B12, Rm. 204, 3F
- product SKUs/serial numbers: AB-123-X9, SN: 04A9-ZX
- order/tracking IDs: 1Z999AA10123456784
- bank/account/card endings: IBAN PL..., **** 1234, Visa •••• 4242
- crypto addresses: 0x742d...
- code identifiers: snake_case, camelCase, HTTPServer
- programming syntax: foo.bar(), ==, !=, &&, ||
- command-line flags/commands: --help, -v, npm install
- keyboard shortcuts: Ctrl+C, ⌘K, Alt+Tab
- UI paths/menus: File > Export > PDF
- Markdown/HTML/XML/JSON: # Title, <br>, **bold**, { "id": 123 }
- social counts: 1.2K likes, 3M views
- stock tickers: $AAPL, NASDAQ: MSFT
- financial shorthand: EPS, P/E, YoY, QoQ, bps, 10Y
- fiscal/period expressions: FY25, Q3 2026, H1 2026, 2026-W28
- calendar recurrences/relative business terms: Mon/Wed/Fri, biweekly, EOD, COB
- name initials: J. R. R. Tolkien
- names with particles: van Gogh, de Gaulle
- generational suffixes: Jr., Sr., III
- company suffixes: Inc., Ltd., GmbH, S.A.
- homographs/heteronyms: read, lead, Polish/polish, St. John, Reading
- profanity masking/censored words: f***, sh*t, @$$
- redactions/placeholders/template variables: [REDACTED], xxxxx, {name}, [insert date], {{ user.first_name }}
- tables/headings/captions: rows/columns, H1, Section 2, image alt text
- emphasis/capitalization: bold, italics, ALL CAPS, US vs us
- repeated characters: soooo, !!!, ???
- Unicode symbols: ™, ®, ©
- currency codes: USD, PLN, CHF
- country/language codes: en-US, pl-PL
- locale-specific separators: 1,234.56 vs 1.234,56
- decimal commas: 3,14
- numeral systems: Arabic-Indic digits, Devanagari digits
- full-width/half-width forms: ＡＢＣ, １２３
- ligatures: ﬁ, ﬂ
- diacritics: é, ł, ñ
- transliteration/language switching: Cyrillic names, Arabic names, Chinese names, English text inside Polish text
- whitespace/formatting: extra spaces, line breaks, nonbreaking spaces
### Lexicon support
dict that auto converts given words into phonemes

### previous_text and next_text context (audio and text support)
two texts with ssml -> bert style/diffusion
audio -small model> diffusion
### Voice Prompt
llm prompt explaining voice:
- age
- gender
- pitch and timbre
- accent/dialect
- formality
- resonance
- breathiness
- vocal range
- roughness
- articulation character

### Style prompt
emotion and intensity
speaking rate
loudness
tone
pitch movement
energy
rhythm
pause behavior
formality
conversational versus narrative delivery, audiobook, podcast
whispering, shouting, laughing, crying

#### Basic
neutral
happy
cheerful
excited
enthusiastic
elated
euphoric
triumphant
amazed
surprised
awe
flirtatious
curious
content
peaceful
serene
calm
grateful
affectionate
trust
sympathetic
anticipation
mysterious
mischievous
angry
mad
outraged
frustrated
agitated
threatened
disgusted
contempt
envious
sarcastic
ironic
sad
sorrowful
dejected
melancholic
disappointed
hurt
guilty
crying
bored
tired
rejected
nostalgic
wistful
apologetic
hesitant
indecisive
insecure
confused
quizzical
resigned
anxious
panicked
alarmed
scared
cautious
proud
confident
distant
erotic
skeptical
contemplative
determined
dramatic
laughing
sighing
whispering
shouting
groaning
exhaling
#### Advanced
llm prompts describing samples.
### Do
laughs
breath
wheezing
whispers
shout
beep
giggling
clear throat
yawn
sighs
exhales
curious
chuckles
crying
snorts
swallows
moan
gulps

woo,ummm,ohhh,hhmmm, wwwmm - vocalization + classifier of exact phonemes i will merge it into transcript



i need:
- "Do" classifier on audio
- score audio regression model MOS
- pass samples through gemini and get style and voice descriptions.
- text -> normalized text model
- "break/p/s" detection/handling/formatting
- text -> voice model


Data Pipeline

sources:
- yt (all langs)
- audiobooki
- synth (all langs)
- dataset'y gotowe (to co pobralem + english versions)

models to add:
- test other cleaning models
- test other diarization models
- global voice embedding and aggregation???

pipelines:
- yt pipeline
- audiobooki pipeline
- synth pipeline
- datasety gotowe
- datasety english
- elevenlabs ds
na

- mos score training
- mos score classifier usage
- silence processing (model? forced alignment) -> break/p/s
- audio effects tags classifier usage
- audio effect + style/emotions train
- style/emotions tags
- voice and style prompts
- text -> voice model

- normalization model for text
- normalization model training

models:
- mos classifier
- tags classifier (style/emotions + tags)
- text -> voice model
- text -> style model

all modules:
- AudioEncoder
- Generator
- Decoder
- DurationPredictor
- LatentFlowModel
- PhonemeAligner
- PhonemeEncoder - phoneme bert initialization
- TextEncoder - bert initialization

- StyleEncoder
- Generator
- Decoder

- F0Extractor (+ other models to get all features)
- FeatureLinear (linear layer)

DurationPredictor is normalizing flow.

LatentFlowMode model: diffusion forcing + shorctut models + flow matching + local cross attention inside model with tokens + cnn

The flow model is cnn, it accept input x_t, t (0.0 - 1.0) per every token, 

phonemes -PhonemeEncoder> text embeddings + text pool vector (mean)

audio mel -AudioEncoder> z -StyleEncoder 1> voice embedding
audio mel -AudioEncoder> z -StyleEncoder 2> style embedding

voice prompt -TextEncoder> voice embedding
style prompt -TextEncoder> style embedding

pre and post encoders for audio and text: text -PhonemeEncoder> vectors -linear> mean, audio -AudioEncoder> vectors -Linear> mean 

text embeddings + style vector + voice vector + text pool vector + pre/post vectors (apply to first/last n phonemes) -DurationPredictor> durations

text embeddings duplicated + text embeddings for local cross attention + style vector + voice vector + text pool vector + pre/post vectors (apply to first / last n phonemes) -LatentFlowModel> z

each conditioning (text embedding, style, voice, text pool, pre/post vectors have drop chance during training, so cfg is possible). noise -> z; conditioning works by conv channel concat at some layer + adaLN-Zero.

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
- AudioEncoder + Generator + All features + Decoder training + GANs
- everything except above (up to latents from audioencoder training)
- e2e finetuning.
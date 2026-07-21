Please download all (download to stage1/ don't upload to backend for now):

for each create folder in stage1/dataset_name/
- with wavs/
- data.json (all required backend data to import)
- src/ code to download all data.
- tmp/ (tmp downlowded data for example mp3, .txt, etc.)

start from biggest datasets so disk won't be filled now so you can collect more speakers.

iterate over all. limit of 45 min per dataset, if exceeded then move to next. also avoid slow download/upload. read order.md fully. don't use subagents / git worktrees. be sure so you would download all metadata. mark completed/failed/not possible/etc. datasets. avoid disk over 512 GB. the target dataset around 250GB so plan ahead/cleanup.

be sure to use 24khz, mono, 24 bit audio.  make sure that your code is optimized, so use multiprocessing if required, benchmark if something above 4 minutes, be sure that after switch to next tmp/ will be empty. don't give up during download easy.

.env contain hf token.


  Dataset           Hours to download
  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━
   NVSpeech                  50.00
  ────────────────  ───────────────
   SynParaSpeech             50.00
  ────────────────  ───────────────
   MNV-17                     7.55
  ────────────────  ───────────────
   EmoGator                  16.97
  ────────────────  ───────────────
   ASVP-ESD v2               18.00
  ────────────────  ───────────────
   VocalSound                24.40
  ────────────────  ───────────────
   Nonspeech7k                6.75
  ────────────────  ───────────────
   PodcastFillers            50.00
  ────────────────  ───────────────
   AudioSet                  50.00
  ────────────────  ───────────────
   FSD50K                    50.00
  ────────────────  ───────────────
   VGGSound                  50.00
  ────────────────  ───────────────
   ESC-50                     2.78
  ────────────────  ───────────────
   FSDKaggle2019             50.00
  ────────────────  ───────────────
   ehehe Corpus               5.13
  ────────────────  ───────────────
   COUGHVID                  35.00
  ────────────────  ───────────────
   Coswara                   50.00
  ────────────────  ───────────────
   ICBHI 2017                 5.50
  ────────────────  ───────────────
   Category total           522.08

  ### B. Emotion and expression datasets

   Dataset             hours to download
  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━
   MSP-Podcast v2.x            50.00
  ──────────────────  ───────────────
   BEAT                        50.00
  ──────────────────  ───────────────
   MEAD                        37.30
  ──────────────────  ───────────────
   ESD                         29.10
  ──────────────────  ───────────────
   EmoV-DB                      9.50
  ──────────────────  ───────────────
   SUBESCO                      7.68
  ──────────────────  ───────────────
   Emozionalmente               6.30
  ──────────────────  ───────────────
   CREMA-D                      5.00
  ──────────────────  ───────────────
   ShEMO                        3.42
  ──────────────────  ───────────────
   ASED                         2.10
  ──────────────────  ───────────────
   EMNS / Imz                   2.30
  ──────────────────  ───────────────
   JL Corpus                    1.40
  ──────────────────  ───────────────
   CaFE                         1.20
  ──────────────────  ───────────────
   eNTERFACE’05                 1.10
  ──────────────────  ───────────────
   AESDD                        0.70
  ──────────────────  ───────────────
   EmoDB                        0.40
  ──────────────────  ───────────────
   MESD                         0.20
  ──────────────────  ───────────────
   Category total             207.69

  ### C. MOS, naturalness, and quality datasets

   Dataset                          hours to download
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━
   PSTN Speech Quality Corpus               50.00
  ───────────────────────────────  ───────────────
   NISQA Corpus                             27.21
  ───────────────────────────────  ───────────────
   Tencent Speech Quality Corpus            23.51
  ───────────────────────────────  ───────────────
   SOMOS / SOMOS-clean                      18.32
  ───────────────────────────────  ───────────────
   TMHINT-QI                                11.35
  ───────────────────────────────  ───────────────
   BVCC / VoiceMOS 2022                      8.02
  ───────────────────────────────  ───────────────
   URGENT 2024 human-MOS                    13.80
  ───────────────────────────────  ───────────────
   SingMOS-Pro                              11.15
  ───────────────────────────────  ───────────────
   TCD-VoIP                                  0.87
  ───────────────────────────────  ───────────────
   TTSDS2                                    0.96
  ───────────────────────────────  ───────────────
   Blizzard Challenge 2019                   0.32
  ───────────────────────────────  ───────────────
   CHiME-7 UDASE                             0.84
  ───────────────────────────────  ───────────────
   Category total                          166.35

   Summary                  Hours
  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━
   Vocal events            522.08
  ────────────────────  ──────────
   Emotion/expression      207.69
  ────────────────────  ──────────
   MOS/quality             166.35
  ────────────────────  ──────────
   Grand total           896.12 h


# Extra Info

# Speech datasets for nonverbal vocalization, expression, and MOS labels

Research snapshot: **17 July 2026**. This document deliberately excludes ordinary ASR/transcript corpora. A dataset is included only when its audio has at least one useful **non-text** target:

- a vocal-event class or timestamp (`laugh`, `breath`, `wheeze`, `whisper`, `sigh`, `cough`, `cry`, `grunt`, `uh/um`, etc.);
- an emotion, affect, intensity, prosody, speaking-style, or expressiveness label; or
- a per-clip human opinion score (MOS), naturalness score, or multidimensional speech-quality rating.

Some datasets happen to ship transcripts too. That does **not** make them transcript datasets here: the proposed use is the audio plus the non-text label columns only.

Every dataset below can be obtained without another person approving or fulfilling the request. Direct downloads, source-URL download scripts, account creation, and automatic click-through license acceptance are allowed; email requests, manual approval, signed institutional agreements, and unreleased or access-uncertain corpora are excluded. URL-based web datasets may still have missing items when the original media has been deleted.

## How to read the hours

- **reported** means the publisher/paper states the duration;
- **measured** means an official downstream recipe summed the released audio;
- **computed** means duration follows directly from a published fixed clip count and duration. It is marked with `~` and the arithmetic is shown;
- when a corpus contains long source recordings but only short target events, both are stated where known. Do not interpret the full hours as hours of positive events.

## A. Direct nonverbal and paralinguistic vocal-event datasets

These are the closest matches to the labels originally listed in this file.

| # | Dataset | Language(s) | Size (hours) | What the audio and non-text annotations contain | Practical fit / limitations | Source |
|---:|---|---|---:|---|---|---|
| 1 | **NVSpeech** | Mandarin Chinese | **573.4 h reported**; includes a **76 h manually annotated seed set** | 174,179 utterances with word-aligned inline tags over 18 paralinguistic categories. The paper explicitly includes laughter, breathing, cough, crying, sigh, sniff, throat clearing, `uhm`, confirmation sounds, question-like `ah`, surprise `oh`, and other Mandarin interjections. | Exceptional match for learning event locations inside normal speech. The 76 h subset is human-labeled; the remaining 573 h corpus is model-labeled, so retain provenance as a confidence field. | [paper](https://arxiv.org/abs/2508.04195), [project](https://nvspeech170k.github.io/) |
| 2 | **SynParaSpeech** | Chinese conversational speech | **118.75 h reported** | Six event categories—**sigh, throat clearing, laugh, pause, tsk, gasp**—inserted in or mined from natural conversational speech, with millisecond-level timestamps. | Very strong for sequence tagging and TTS control. Labels were created by an automated construction pipeline rather than entirely by hand. | [paper](https://arxiv.org/abs/2509.14946), [data card](https://huggingface.co/datasets/shawnpi/SynParaSpeech) |
| 3 | **MNV-17** | Mandarin Chinese | **7.55 h reported** | Performative speech designed around 17 nonverbal-vocalization categories embedded in speech, including such events as sighs, laughs, and coughs. | Clean, deliberately covered classes and much less label noise than web audio; acted rather than spontaneous. The Hugging Face gate requires only automatic account/license acceptance. | [paper](https://arxiv.org/abs/2509.18196), [data card](https://huggingface.co/datasets/maimai11/MNV_17) |
| 4 | **EmoGator** | Mostly English-speaking contributors; vocal bursts themselves are non-lexical | **16.9654 h reported** | 32,130 short vocal bursts from 357 speakers. Speakers chose one of **30 emotion categories** for each self-produced laugh, cry, sigh, moan, groan, or related sound. | Broad emotion vocabulary and open source; self-assigned intended emotion is not the same as listener-perceived emotion. | [paper](https://arxiv.org/abs/2301.00508), [repository](https://github.com/fredbuhl/EmoGator) |
| 5 | **ASVP-ESD v2** | Chinese, English, French, Russian, and other/no-language media | **>11 h reported** in the release (a later standardized inventory measures **18.0 h**) | 13,964 speech and non-speech clips from media. Twelve emotions plus breath; filenames encode vocal channel, emotion, intensity, age, language, and noise. Explicit sounds include **sigh, yawn, laugh/giggle, cry, sniffle, scream, panic, grunt, gasp, groan, breath**. | One of the closest label inventories to the requested list and genuinely multilingual. Natural/media data can contain background sound, mixed voices, duplicates, and source/copyright concerns; use the filename flags. | [official release](https://zenodo.org/records/7132783), [18 h standardized inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 6 | **VocalSound** | Crowdsourced worldwide; metadata includes native language/country | **~24.4 h computed** (21,024 clips × published 4.18 s mean) | Six clean, isolated classes: **laughter, sigh, cough, throat clearing, sneeze, sniff**, from 3,365 people. Includes age, gender, native language, country, and health metadata. | The cleanest direct classifier-training set for six requested tags. It has one class per clip and no timestamps inside continuous speech. | [official download page](https://sls.csail.mit.edu/downloads/vocalsound/), [paper](https://arxiv.org/abs/2205.03433) |
| 7 | **Nonspeech7k** | Non-lexical; sources are international web audio | **6.75 h reported** | 7,014 strongly, manually, single-label clips (0.5–4 s) in seven classes: **breath, cough, cry, laugh, scream, sneeze, yawn**. | Near-perfect label match, compact and easy to train. It is assembled from Freesound/YouTube/Aigei, is class-imbalanced, and speaker/language provenance is limited. | [paper/data description](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/sil2.12233), [archive](https://doi.org/10.5281/zenodo.6967442) |
| 8 | **PodcastFillers** | English | **145 h reported** (199 episodes, >350 speakers) | 85,803 manually timestamped events: **uh (17,907), um (17,078), agreement sounds such as mm-hmm/ah-ha (3,755), breath (8,288), laughter (6,623)**, plus words, repetition, music, noise, overlap, “you know,” “like,” and other fillers. Also supplies centered 1-second event clips. | Best open long-form source for exact `ummm`/`uh`, agreement hums, breaths, and laughter. Annotation license is non-commercial and full episodes contain much ordinary speech. Ignore the optional ASR JSON. | [official overview](https://podcastfillers.github.io/), [detailed labels](https://podcastfillers.github.io/dataset/) |
| 9 | **AudioSet** | Worldwide YouTube; language-independent/multilingual | **5,800 h reported** (2,084,320 ten-second segments) | Weak clip-level multilabel ontology. Direct requested classes include **shout, scream, whisper, laughter** (and baby laugh/giggle/snicker/belly laugh/chuckle), **cry/sob, wail/moan, sigh, hum, groan, grunt, yawn, breathing, cough, sneeze, sniff, hiccup**, and more. For scale: laughter is 15.8 h, cough 2.4 h, sigh 0.9 h. | Largest useful pretraining source, but labels only assert presence somewhere in 10 s and YouTube availability decays. Class quality varies—Google estimates sigh at only 33% precision, while cough/laughter are high quality. | [dataset](https://research.google.com/audioset/), [human-voice ontology](https://research.google.com/audioset/ontology/human_voice_1.html), [class/hour table](https://research.google.com/audioset/dataset/index.html) |
| 10 | **FSD50K** | Worldwide Freesound; largely language-independent | **108.3 h reported** | 51,197 Creative-Commons audio clips with manually verified multilabel annotations over 200 AudioSet classes. Human-vocal labels include cough (and throat clear), sneeze, breath, sniff, laughter, scream, crying/sobbing, whisper, sigh and related classes. | Better waveform availability and licensing than AudioSet. Many recordings are long and labels are clip-level, not event timestamps. Preserve each clip’s individual CC license. | [paper](https://arxiv.org/abs/2010.00475), [official archive](https://zenodo.org/records/4060432), [dataset explorer](https://annotator.freesound.org/fsd/explore/) |
| 11 | **VGGSound** | Worldwide YouTube; multilingual/context-independent classes | **~583.3 h computed** (210,000 ten-second clips) | One label per audiovisual 10-second clip across 310 classes. Relevant classes include people **babbling, belly laughing, burping, coughing, crying, gasping, giggling, groaning, grunting, hiccupping, screaming, sighing, sneezing, snoring, whispering, whistling**, plus baby cry and child speech. | Large and visually verified, useful for pretraining. Labels are weak and the downloadable artifact is URLs/timestamps; many YouTube items have disappeared. | [paper](https://arxiv.org/abs/2004.14368), [official repository](https://github.com/hche11/VGGSound) |
| 12 | **ESC-50** | Language-independent Freesound clips | **2.78 h computed/reported** (2,000 × 5 s) | Balanced 50-way benchmark with 40 clips/class. Human non-speech subset includes **crying baby, sneezing, breathing, coughing, laughing, snoring, drinking/sipping, clapping, footsteps, brushing teeth**. | Tiny but clean and balanced; excellent sanity-test set, poor sole training source. Only 40 examples per target class. | [official repository](https://github.com/karolpiczak/ESC-50) |
| 13 | **FSDKaggle2019 / DCASE 2019 Task 2** | Worldwide Freesound | **103.4 h reported** | 29,266 variable-length clips tagged with 80 AudioSet classes. Target labels include **cough, sneeze, sigh, scream, whisper, laughter**, and related human sounds; a small curated portion is manually verified and the large portion is noisy-labeled. | Useful scale between FSD50K and ESC-50, but do not merge curated and noisy labels without a confidence/provenance column. | [task/paper](https://dcase.community/challenge2019/task-audio-tagging), [paper](https://arxiv.org/abs/1906.02975) |
| 14 | **ehehe Corpus** | Japanese; performed by Japanese voice actors | **5.13 h reported** | Character-organized, acted **laughter** audio from Japanese voice actors, intended for laughter detection and synthesis. | Valuable language/style expansion for laughter; performed anime/fiction style is not representative of everyday spontaneous laughter. | [data card](https://huggingface.co/datasets/litagin/ehehe-corpus) |
| 15 | **COUGHVID** | Worldwide crowdsourced, non-lexical cough | **~35 h reported** after filtering at cough probability >0.8 | More than 20,000 phone/web recordings. Metadata includes age, gender, location, symptoms/COVID status, automatic cough probability and estimated SNR; 2,800 samples were expert-assessed for cough type/severity and health status. | Excellent cough diversity and quality metadata, but it is a cough/health corpus—not a general vocalization taxonomy. Do not treat COVID label as acoustically definitive or use it for diagnosis without clinical validation. | [official archive](https://zenodo.org/records/4048312), [paper/data description](https://pmc.ncbi.nlm.nih.gov/articles/PMC8222356/) |
| 16 | **Coswara** | Mainly Indian participants; speech prompts plus non-lexical sounds | **65 h reported** | 2,635 people contribute nine modalities: shallow/deep breathing, shallow/heavy cough, sustained vowels, counting at normal/fast pace, and other voice samples. Includes demographics, symptoms, comorbidities, COVID status, and **manual quality labels for the entire audio collection**. | Direct source for breath/cough plus a useful clean/noisy quality target. Health labels require the same clinical caution as COUGHVID. | [official data repository](https://github.com/iiscleap/Coswara-Data), [paper](https://arxiv.org/abs/2305.12741) |
| 17 | **ICBHI 2017 Respiratory Sound Database** | Portuguese/Greek clinical collection; no lexical dependency | **5.5 h reported** | 920 recordings from 126 subjects, segmented into 6,898 respiratory cycles: **1,864 crackle, 886 wheeze, 506 both, remainder normal**. Includes acquisition-device and diagnosis metadata. | The most direct public source for the requested `wheezing` class. These are stethoscope/lung sounds, not mouth-level conversational breathing, so keep the domain separate. | [official download page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge), [database paper](https://doi.org/10.1088/1361-6579/ab03ea) |

## B. Emotion, intensity, prosody, and expressiveness datasets

These do not usually say “this is a sigh.” They are useful for the requested labels such as `curious`, excited, sad/crying-like, whispery/calm, and for an **expressiveness score or class derived from vocal delivery rather than words**. Train from audio and emotion/intensity columns; transcripts are unnecessary.

| # | Dataset | Language(s) | Size (hours) | Exact non-text target | Important caveat | Source |
|---:|---|---|---:|---|---|---|
| 18 | **MSP-Podcast v2.x** | English, natural podcast speech | **>400 h reported** | Speaking-turn clips rated by at least five listeners for primary and secondary categorical emotions and continuous **valence, arousal, dominance**; speaker IDs are available for most samples. | Best large naturalistic affect source. It is licensed for research; its Hugging Face gate is automatic after account/license acceptance. Emotion is perceived at utterance level, not a vocal-event boundary. | [2025 corpus paper](https://arxiv.org/abs/2509.09791), [official corpus page](https://www.lab-msp.com/MSP/MSP-Podcast.html), [data card](https://huggingface.co/datasets/wanchichen/msp_podcast) |
| 19 | **BEAT** | English (60 h), Mandarin (12 h), Spanish (2 h), Japanese (2 h) | **76 h reported** | 30 speakers in conversational/performative recordings with 8 emotions: **happiness, anger, disgust, sadness, contempt, surprise, fear, neutral**; also rich gesture/audio-visual data. | Strong multilingual expressive coverage, but designed for conversational gesture synthesis and distributed under non-commercial terms. | [paper/project listing](https://github.com/PantoMatrix/BEAT), [dataset inventory](https://github.com/SuperKogito/SER-datasets) |
| 20 | **MEAD** | English | **37.3 h measured** | 60 actors speak at three intensity levels in **neutral, happy, angry, sad, surprise, fear, disgust, contempt**. | Large and intensity-controlled but highly acted; frontal video is optional for this audio-only use. | [official repository](https://github.com/uniBruce/Mead), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 21 | **ESD (Emotional Speech Dataset)** | English and Mandarin Chinese | **29.1 h reported** | 20 speakers, 350 parallel utterances each, in **neutral, happy, angry, sad, surprise**. | Excellent cross-lingual and voice-conversion control because content is parallel; acted, studio-clean, and only five emotions. | [official download page](https://hltsingapore.github.io/ESD/download.html), [paper](https://arxiv.org/abs/2105.14762) |
| 22 | **EmoV-DB** | English (multiple accents) and a French male speaker | **9.5 h measured** | 6,887 standardized clips covering **amused, angry, disgusted, sleepy, neutral** (speaker subsets vary). | Especially relevant to amusement and sleepy/breathy style; acted TTS-oriented recordings and uneven emotion coverage by speaker. | [official repository](https://github.com/numediart/EmoV-DB), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 23 | **SUBESCO** | Bangla | **7 h 40 min 40 s reported** | 7,000 four-second clips from 20 speakers in **anger, disgust, fear, happiness, neutral, sadness, surprise**. Both intended and individual perceived-emotion ratings are released. | Balanced and CC BY 4.0, but acted and limited to ten sentences. | [paper and data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8087046/), [standardized loader](https://audeering.github.io/datasets/datasets/subesco.html) |
| 24 | **Emozionalmente** | Italian | **6.3 h measured** | 6,902 acted utterances with seven emotion categories, designed as a substantially larger Italian SER corpus than EMOVO. | Useful language expansion; acted studio style. | [official paper](https://www.nature.com/articles/s41597-023-02506-9), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 25 | **CREMA-D** | English; ethnically diverse US actors | **~5 h measured** | 7,442 clips from 91 actors in **anger, disgust, fear, happy, neutral, sad**, with low/medium/high/unspecified intended intensity. 2,443 listeners supplied audio-only categorical and real-valued perceived-intensity ratings. | The individual listener votes are more valuable than only the majority label. Acted phrases; audio files are open, full Git-LFS checkout with video is about 7.5 GB. | [official site](https://cheyneycomputerscience.github.io/CREMA-D/), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/) |
| 26 | **ShEMO** | Persian/Farsi | **3 h 25 min reported** | 3,000 semi-natural utterances from 87 speakers in radio plays, majority-labeled by 12 annotators as **anger, fear, happiness, sadness, surprise, neutral**. | More natural than actor prompt sets and freely available for academic use; class imbalance and radio-drama acting remain. | [paper](https://arxiv.org/abs/1906.01155), [official repository](https://github.com/mansourehk/ShEMO) |
| 27 | **ASED** | Amharic (Gojjam, Wollo, Shewa, and Gonder dialects) | **2.1 h measured** | 2,474 two-to-four-second acted utterances from 65 speakers, judged by eight raters as **neutral, fearful, happy, sad, angry**. | Valuable low-resource language and dialect coverage; still small and acted. | [release paper](https://arxiv.org/abs/2201.02710), [official repository](https://github.com/Ethio2021/ASED_V1), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 28 | **EMNS / Imz Corpus** | English | **2.3 h reported** (a standardized training inventory uses **1.9 h**) | A single female storyteller performs 1,181/1,205 released clips in eight balanced emotional states, with **expressiveness levels**, natural-language emotion descriptions, and word-emphasis labels. | The expressiveness level is unusually relevant, but one speaker means it cannot teach speaker-independent cues by itself. | [official OpenSLR download](https://www.openslr.org/136/), [paper](https://arxiv.org/abs/2305.13137) |
| 29 | **JL Corpus** | New Zealand English | **1.4 h measured** | 2,400 clips from four professional speakers with five primary emotions (**angry, happy, neutral, sad, excited**) and five nuanced secondary emotions (**anxious, apologetic, confident/pensive, enthusiastic, worried**). | The secondary classes directly help nuanced expressiveness; only four speakers makes speaker leakage a major risk. | [data](https://www.kaggle.com/datasets/tli725/jl-corpus), [corpus analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10053518/), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 30 | **CaFE (Canadian French Emotional)** | Canadian French | **1.2 h measured** | 936 audio/video clips, 12 actors, seven basic emotions plus multiple intensities. | Good French/accent expansion; small, acted, license requires checking at download. | [official project](https://zenodo.org/records/1478765), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 31 | **eNTERFACE’05 Emotion** | English spoken by international participants | **1.1 h measured** | 1,263 acted audiovisual utterances in **anger, disgust, fear, happiness, sadness, surprise**. | Speakers have many native languages but speak English; expressive exaggeration and recording artifacts are common. | [official corpus page](https://enterface.net/enterface05/), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 32 | **AESDD** | Greek | **0.7 h measured** | 604 acted clips from professional actors in five emotions: **anger, disgust, fear, happiness, sadness**. | Very small; useful only as cross-lingual evaluation or part of a pooled corpus. | [data mirror/card](https://huggingface.co/datasets/EdwardLin2023/AESDD), [corpus review](https://www.mdpi.com/2306-5729/10/10/164), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 33 | **EmoDB / Berlin Emotional Speech Database** | German | **0.4 h measured** | 535 validated acted utterances: **anger, boredom, disgust, fear/anxiety, happiness, sadness, neutral**. | `Boredom` is a useful proxy for low expressiveness, but the corpus is tiny and highly acted. | [standardized dataset page](https://audeering.github.io/datasets/datasets/emodb.html), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |
| 34 | **MESD (Mexican Emotional Speech Database)** | Mexican Spanish | **0.2 h measured** | 862/864 selected single-word clips from adult women, adult men, and children in **anger, disgust, fear, happiness, neutral, sadness**. | Useful adult/child Mexican-Spanish check set, far too small as the only training source. | [data page](https://www.kaggle.com/datasets/saurabhshahane/mexican-emotional-speech-database-mesd), [corpus description](https://www.mdpi.com/2076-3417/15/8/4340), [measured inventory](https://openreview.net/pdf?id=uBcHcM7Kzi) |

## C. Per-audio MOS, naturalness, and quality datasets

These are the exact sources for `MOS scores per audio (quality, expressiveness, etc.)`. Human MOS is subjective and protocol-specific. Keep **raw listener scores, listener ID, test condition, system ID, language, and rating question** where the release provides them; a 4.0 “naturalness” MOS is not interchangeable with a 4.0 telephony “overall quality” MOS.

The measured hours below use the current [URGENT 2026 speech-quality preparation table](https://github.com/urgent-challenge/urgent2026_challenge_track2) where noted; that recipe provides reproducible clip counts, duration and download/license links.

| # | Dataset | Language/domain | Size (hours) | Exactly what is scored | Practical fit / limitations | Source |
|---:|---|---|---:|---|---|---|
| 35 | **PSTN Speech Quality Corpus** | Worldwide narrowband telephone networks; speech material is mainly English | **163.08 h measured** (58,709 clips) | Each recording was sent through one of 80 PSTN networks in >50 countries and then VoIP; 69% also mix noise. Crowdsourced listeners rate per-file overall speech quality; cleaned files have 1–10 ratings (mean 4.6). | Largest accessible human-MOS telephony corpus. It learns network/noise quality, not TTS expressiveness, and licensing is not clearly stated. | [direct archive](https://challenge.blob.core.windows.net/pstn/train.zip), [paper](https://arxiv.org/abs/2007.14598), [automated preparation](https://github.com/urgent-challenge/urgent2026_challenge_track2) |
| 36 | **NISQA Corpus** | English/German and live international communication channels | **27.21 h measured** for the standard 11,020-file train preparation; >14,000 clips in the complete corpus | Human ratings for **overall quality, noisiness, coloration, discontinuity, loudness** under codecs, packet loss, noise, clipping, mobile calls, Zoom, Skype, WhatsApp and live talking. | Best multidimensional quality target. Several component datasets have different licenses/protocols; use the supplied corpus/split metadata. | [official repository](https://github.com/gabrielmittag/NISQA), [paper](https://arxiv.org/abs/2104.09494), [archive](https://depositonce.tu-berlin.de/items/b8908103-b0e8-4912-8144-aea65098fa1f) |
| 37 | **Tencent Speech Quality Corpus** | Mandarin Chinese | **23.51 h measured** (11,563 prepared clips) | Human MOS for clean, noisy, reverberant and enhanced Mandarin speech; commonly distributed as with-reverb and without-reverb subsets with roughly 28 ratings/clip. | Strong Mandarin complement to NISQA. The original host can be difficult outside China; an Apache-licensed prepared mirror is linked by URGENT. | [original corpus link](https://share.weiyun.com/B4IS0l3z), [URGENT preparation](https://github.com/urgent-challenge/urgent2026_challenge_track2) |
| 38 | **SOMOS / SOMOS-clean** | English neural TTS (LJ Speech voice) | **18.32 h measured** for the common 14,100-file clean training split; full release has 20,000 synthetic + 100 natural clips | 200 Tacotron-like systems synthesize 2,000 texts. Release contains 374,955 individual 1–5 **naturalness** judgments, anonymized listener data, system IDs, reliability controls, and unseen-system/listener/text splits. | Best open TTS-naturalness data and closest MOS proxy for “expressiveness,” though all samples use one voice and one LPCNet vocoder. Full-release hours are larger than the measured training subset. | [official archive](https://zenodo.org/records/7119400), [paper](https://arxiv.org/abs/2204.03040) |
| 39 | **TMHINT-QI** | Taiwan Mandarin | **11.35 h measured** for the standard 12,937-file train/validation preparation; full set is 24,408 ~3 s items (**~20.34 h nominal**) | Clean sentences mixed with babble/street/pink/white noise at several SNRs and processed by five enhancement systems. 226 listeners rate both **quality MOS and intelligibility**. | Rare dataset with human quality and intelligibility together. Multiple processed versions can share the same source utterance; split by source sentence/speaker, not random clip. | [paper](https://arxiv.org/abs/2111.02585), [VoiceMOS 2023 release](https://github.com/dhimasryan/TMHINT-QI-VoiceMOS2023) |
| 40 | **BVCC / VoiceMOS Challenge 2022 main track** | English TTS and voice conversion | **8.02 h measured for all 7,106 clips** | A unified listening test re-rates 187 systems from Blizzard Challenges, Voice Conversion Challenges, and ESPnet-TTS. Supplies utterance- and system-level MOS and listener/system-aware splits. | Diverse and standard for MOS prediction. Blizzard audio cannot be redistributed, so the release supplies download/preprocessing scripts; do not confuse its main and OOD archives or count them as separate datasets. | [official challenge](https://voicemos-challenge-2022.github.io/), [data archive](https://zenodo.org/records/10691660), [duration evidence](https://cslinzhang.github.io/home/files/TASLP_Qiao.pdf) |
| 41 | **URGENT 2024 human-MOS set** | English noisy/enhanced speech, 8–48 kHz | **13.8 h reported** (6,900 clips) | 300 noisy test items plus outputs of 22 enhancement systems. Each clip has 1–5 MOS, raw ratings from **eight distinct MTurk listeners**, and source/system IDs. | Clean, simple, downloadable per-clip MOS table. It is an evaluation set derived from shared sources, so group variants of the same source in one split. | [official data card](https://huggingface.co/datasets/urgent-challenge/urgent2024_mos), [challenge](https://urgent-challenge.github.io/urgent2024/data/) |
| 42 | **SingMOS-Pro** | Chinese and Japanese singing | **11.15 h reported** (7,981 clips) | Real and synthesized singing from synthesis, conversion, vocoder, codec and ground-truth systems. The release provides per-utterance and system MOS plus split, system and source metadata. | The current official SingMOS benchmark and strongest public multilingual singing-quality MOS source; singing MOS does not calibrate directly to spoken MOS. CC BY 4.0 and directly downloadable. | [data card](https://huggingface.co/datasets/TangRain/SingMOS-Pro), [paper](https://arxiv.org/abs/2510.01812) |
| 43 | **TCD-VoIP** | English wideband speech | **0.87 h measured** (384 clips) | Four speakers under five degradation families: chop, clipping, competing speaker, echo, and background noise. Each item has subjective opinion scores from **24 listeners** and the mean. | Small but carefully controlled, useful for validating degradation sensitivity; non-commercial license. | [official dataset](https://sigmedia.tv/datasets/tcd_voip_ltd/), [duration/method description](https://researchrepository.ucd.ie/server/api/core/bitstreams/47b177dc-6f1e-4924-8fb3-196967c9d777/content) |
| 44 | **TTSDS2 listening-test corpus** | **English** released listening set; associated benchmark separately supports 14 languages | **0.96 h measured** (460 clips from 80 system/configuration groups in the prepared set) | More than 11,000 listener judgments including **MOS, comparative MOS (CMOS), and speaker-similarity MOS (SMOS)** across recent voice-cloning systems and clean/noisy audiobooks, in-the-wild YouTube speech, and children’s dialogue. | Small audio set but modern and diverse; best as validation/OOD data, not the primary training corpus. The downloadable listening-test corpus is tagged English, so the benchmark's 14-language coverage must not be counted as 14 MOS-labelled languages. Do not collapse MOS, CMOS, and SMOS into one target. | [official data card](https://huggingface.co/datasets/ttsds/listening_test), [paper](https://www.isca-archive.org/ssw_2025/minixhofer25_ssw.html), [URGENT preparation](https://github.com/urgent-challenge/urgent2026_challenge_track2) |
| 45 | **Blizzard Challenge 2019 (BC19) MOS subset** | English TTS | **0.32 h measured** (136 selected clips, 21 systems) | Natural and synthesized system outputs with human listening-test naturalness/quality scores. | Very small and subject to Blizzard’s custom terms; useful as an out-of-domain system test. | [prepared archive](https://zenodo.org/records/6572573), [Blizzard data terms](https://www.cstr.ed.ac.uk/projects/blizzard/data.html) |
| 46 | **CHiME-7 UDASE Evaluation MOS Data** | English real noisy/enhanced conversational speech | **0.84 h measured** (640 clips, five systems) | Real-world speech-enhancement evaluation stimuli with subjective quality judgments across original/enhanced conditions. | Good real-noise validation, too small for training alone; CC BY-SA 4.0. | [official archive](https://zenodo.org/records/10418311), [challenge description](https://www.chimechallenge.org/challenges/chime7/task2/index) |

**Count:** 46 datasets with hours: 17 direct vocal-event/respiratory datasets, 17 affect/expressiveness datasets, and 12 MOS/quality datasets. BVCC and VoiceMOS are counted once because they refer to the same main-track dataset.

### Multilingual human ratings adjacent to MOS

| Dataset | Languages | Human rating | Size | Access and use |
|---|---|---|---:|---|
| **MANGO** | Hindi, Tamil, and English | **MUSHRA**, 0–100 continuous quality ratings rather than 1–5 MOS | About **246,000 individual human ratings** in ~51,000 released rating rows | CC BY 4.0 and directly downloadable from Hugging Face. It is valuable for multilingual TTS quality learning, but preserve `rating_protocol=MUSHRA` and do not merge its raw values directly with MOS. [data card](https://huggingface.co/datasets/ai4bharat/MANGO), [paper](https://arxiv.org/abs/2411.12719) |

Strictly multilingual, automatically downloadable human-MOS releases in this inventory are therefore **NISQA** (English/German) and **SingMOS-Pro** (Chinese/Japanese singing). Including MANGO's MOS-adjacent MUSHRA ratings expands coverage to six unique languages: English, German, Chinese, Japanese, Hindi, and Tamil. The seven-language **IndicMOS** project publishes code and a trained model, but no self-service release of its underlying audio-plus-human-MOS corpus was verified; **SQuId** covers many locales but is not publicly released.

### Coverage of the original requested labels

| Requested output | Strongest existing supervision | Status |
|---|---|---|
| `laughs`, `giggling`, `chuckles` | VocalSound/Nonspeech7k for generic laugh; AudioSet for giggle, snicker, belly laugh and chuckle; EmoGator for expressive bursts; ehehe for acted laugh styles | **Direct**, but subtype quality is best in AudioSet’s weak 10 s tags rather than strong timestamps. |
| `breath`, `exhales`, `sighs` | NVSpeech, PodcastFillers, VocalSound, Nonspeech7k, SynParaSpeech, ASVP-ESD | **Direct**. `Exhale` is usually subsumed by breath/sigh rather than independently labeled. |
| `wheezing` | ICBHI | **Direct in lung/stethoscope audio**, not ordinary close-mic conversational voice. Collect/label mouth-level wheezes separately if that is the intended sound. |
| `whispers`, `shout` | AudioSet/FSD50K/VGGSound; expressive speech datasets provide intensity/prosody but not usually a whisper event | **Direct weak tags**; create strong timestamps from a reviewed subset. |
| `clear throat` | VocalSound, NVSpeech, SynParaSpeech, FSD50K/AudioSet, MNV-17 | **Direct**. |
| `yawn` | Nonspeech7k, ASVP-ESD, AudioSet | **Direct**. |
| `crying`, `moan`, `snorts`, `gasp/gulps` | Nonspeech7k (cry), EmoGator/ASVP-ESD (cry/moan/groan/gasp), AudioSet (cry/moan/snort and digestive/respiratory neighbors) | **Partly direct**. Generic `gulp` is poorly covered and is not equivalent to gasp. |
| `swallows`, literal drinking `gulps` | The general sound sets may have eating/drinking/digestive neighbors, but this review found no mature, public, speech-embedded swallow corpus with a published total duration. A 2026 in-ear swallowing dataset exists but did not publish enough duration/access detail to count above. | **Important gap**: manually label this class or negotiate access; do not relabel breaths as swallows. See the [in-ear dataset paper](https://pubmed.ncbi.nlm.nih.gov/41977804/). |
| `beep` | AudioSet, FSD50K, VGGSound and FSDKaggle include electronic beep/alarm/bleep classes | **Direct but not human vocalization**. Keep it in a general sound-event head, not the human-vocal head. |
| `woo`, `ummm`, `ohhh`, `hhmmm`, `wwwmm`, exact vocal form | PodcastFillers (`uh`, `um`, agreement sounds), NVSpeech (Mandarin interjections), and SynParaSpeech `tsk` | **Only partially direct**. Most sound-event datasets give a semantic class, not the exact heard syllable. A small controlled token inventory needs new human annotation. |
| `curious` | No corpus above provides a large, clean, categorical `curious` speech label. MSP-Podcast VAD/secondary labels, JL secondary emotions and CLAP prompts can propose candidates. | **Gap / derived label**: define curiosity operationally and human-rate it; do not claim that “surprise” or high arousal equals curiosity. |
| Per-audio quality MOS | PSTN, NISQA, Tencent, TMHINT-QI, TCD-VoIP, URGENT | **Direct human scores** for signal/transmission/enhancement quality. |
| Per-audio naturalness/expressiveness | SOMOS, BVCC, TTSDS2, CREMA-D, EMNS | **Direct but different questions**: SOMOS/BVCC rate naturalness/quality, CREMA-D rates emotion intensity, and EMNS includes expressiveness levels. Preserve the question/protocol rather than merging raw numbers. |

## D. Models that return these data types

### 1. Direct vocal-event output

| Model | Output that is useful here | Availability and recommendation | Source |
|---|---|---|---|
| **OSUM** | Has explicit vocal-event detection (VED): **sigh, cry, scream, laugh, cough, throat clearing, sneeze, other**; also speech emotion and speaking-style tasks. It returns task text/labels rather than requiring a transcript-only workflow. | Open checkpoint/code (Whisper encoder + Qwen2). Best ready-made broad model to try first, especially for English/Mandarin; benchmark it on VocalSound/Nonspeech7k before trusting rare events. | [official repository](https://github.com/ASLP-lab/OSUM), [paper](https://arxiv.org/abs/2501.13306) |
| **NVSpeech paralinguistic-aware ASR** | Inline event tokens with word-level positions across 18 categories: laughter, breathing, cough, crying, sigh, sniff, throat clear, `uhm`, several exact Mandarin interjections. | Research code/model associated with the project; most precise match for “merge event token into transcript,” but Mandarin-centered and its large training set is pseudo-labeled. Keep the event stream separately even if lexical text is discarded. | [paper](https://arxiv.org/abs/2508.04195), [project](https://nvspeech170k.github.io/) |
| **FillerNet** | Frame/clip prediction over **uh, um, laughter, breath, words, music** on podcast audio. | Code/data accompany PodcastFillers. Best specialized model for fillers and breaths in English long-form audio; vocabulary deliberately omits rarer agreement sounds from its consolidated six classes. | [project](https://podcastfillers.github.io/), [paper/repository links](https://podcastfillers.github.io/dataset/) |
| **YAMNet** | 521 AudioSet probabilities every ~0.48 s, including laugh variants, cry/sob, whisper, sigh, cough, sneeze, breath, sniff, yawn, groan, grunt, moan, gasp and more. | Open TensorFlow model, small MobileNet and easiest baseline. Outputs weak window probabilities, not reliable exact boundaries; apply thresholds and temporal smoothing. | [official README](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md) |
| **PANNs (CNN14)** | 527 AudioSet tag probabilities and embeddings with nearly the same human-sound vocabulary as YAMNet. | Open PyTorch checkpoints. Usually a stronger batch/offline baseline than YAMNet but heavier; fine-tune on VocalSound/Nonspeech7k for the desired label set. | [paper](https://arxiv.org/abs/1912.10211), [official code](https://github.com/qiuqiangkong/audioset_tagging_cnn) |
| **BEATs AudioSet-finetuned** | 527-way AudioSet classification/embeddings, therefore the requested vocal events where present in AudioSet. | Open Microsoft PyTorch checkpoints and a very strong encoder. The released head is clip-level; add a sliding-window or sequence-detection head for timestamps. | [official README/checkpoints](https://github.com/microsoft/unilm/tree/master/beats), [paper](https://arxiv.org/abs/2212.09058) |
| **PaSST / EfficientAT** | AudioSet tag probabilities; EfficientAT offers small mobile-friendly CNNs, PaSST a high-accuracy transformer. | Open checkpoints. Choose EfficientAT when inference cost matters, PaSST for accuracy; both inherit AudioSet’s weak labels and class noise. | [PaSST](https://github.com/kkoutini/PaSST), [EfficientAT](https://github.com/fschmid56/EfficientAT) |
| **CLAP (Microsoft or LAION)** | Zero-shot similarity to arbitrary prompts such as “a person swallowing,” “a curious hmm,” or “a breathy chuckle,” so it can cover labels absent from a fixed ontology. | Open weights/code. Excellent candidate generator and bootstrap labeler, **not calibrated probabilities**; prompt wording changes results, so validate every class and include a none/other set. | [Microsoft CLAP](https://github.com/microsoft/CLAP), [LAION CLAP](https://github.com/LAION-AI/CLAP) |
| **voc2vec** | Nonverbal-vocalization embeddings pretrained on about 125 h from AudioSet vocalizations, babies, Nonspeech7k, ReCANVo, VocalSketch, VocalSound and related sources. Downstream examples cover VIVAE, Donate-a-Cry and other nonverbal emotion tasks. | Open representation model, not a turnkey label head. Fine-tune a small classifier/sequence head on the exact final taxonomy; likely a better starting representation than lexical wav2vec for rare non-speech sounds. | [model card](https://huggingface.co/alkiskoudounas/voc2vec), [paper](https://arxiv.org/abs/2502.16298) |

### 2. Emotion and expression output

| Model | Output | Availability and recommendation | Source |
|---|---|---|---|
| **emotion2vec+** | Nine-class emotion probabilities; the base emotion2vec supplies universal utterance/frame embeddings for a custom valence/arousal/expression head. | Open FunASR/ModelScope/Hugging Face models, trained at up to 42,526 h and intended to generalize across languages/environments. Best open general emotion starting point; labels are broad, not literal vocal events. | [official repository](https://github.com/ddlBoJack/emotion2vec), [paper](https://arxiv.org/abs/2312.15185) |
| **SpeechBrain wav2vec2-IEMOCAP** | Four-way **angry, happy, neutral, sad** classification from waveform. | Apache-2.0 and very easy local inference. Useful baseline only: English/IEMOCAP domain, four classes, no calibrated expressiveness intensity. | [model card](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) |
| **Hume vocal-expression model/API** | Dense scores for many nuanced vocal expressions from prosody (commercial model documentation describes 28 expression dimensions), without needing lexical text as the signal. | Hosted/proprietary API rather than open weights. Strong option for nuanced labels such as curiosity/amusement/awkwardness, but versioning, cost, privacy and reproducibility must be accepted. Store full score vectors, model version and segment times. | [model page](https://www.hume.ai/products/vocal-expression-model/), [API docs](https://dev.hume.ai/intro) |
| **openSMILE + supervised head** | Acoustic functionals (pitch, energy, spectral/voice-quality descriptors) from which emotion, arousal, speaking style or expressiveness can be trained. | Open-source for research with licensing conditions. It does not return semantic labels by itself; train the head on MSP-Podcast/ESD and split by speaker. Useful interpretable baseline alongside neural embeddings. | [official repository](https://github.com/audeering/opensmile) |

### 3. MOS and speech-quality output

| Model | Returned values | Best use / warning | Source |
|---|---|---|---|
| **NISQA v2 / NISQA-TTS** | Overall MOS plus **noisiness, coloration, discontinuity, loudness**; separate TTS model returns naturalness. | Downloadable pretrained weights. The main corpus contains **English and German** human-MOS speech. Select the transmitted-speech or TTS checkpoint correctly; scores outside its impairment/domain distribution are estimates, not human MOS. | [official repository](https://github.com/gabrielmittag/NISQA), [paper](https://arxiv.org/abs/2104.09494) |
| **IndicMOS** | Naturalness MOS for TTS/voice-conversion speech, optionally using language/task IDs and ASR error features. | The most directly multilingual spoken-TTS checkpoint found: trained for **Hindi, Marathi, Telugu, Kannada, Bengali, Indian English, and Chhattisgarhi**. CC BY 4.0 weights and inference code are directly downloadable, although the underlying human-rating corpus is not released. | [checkpoint](https://huggingface.co/SYSPIN/IndicMOS), [code](https://github.com/bloodraven66/IndicMOS), [paper](https://www.isca-archive.org/interspeech_2024/udupa24b_interspeech.html) |
| **Singing-SSL-MOS (`singmos_pro`)** | Utterance-level predicted naturalness MOS for singing. | Official pretrained **Chinese/Japanese** SingMOS-Pro model, loadable through `torch.hub`. Use only for singing/SVS/SVC; it is not calibrated for ordinary spoken speech. | [official repository and checkpoint loader](https://github.com/South-Twilight/SingMOS), [paper](https://arxiv.org/abs/2510.01812) |
| **DNSMOS P.835** | **SIG** (speech-signal quality), **BAK** (background quality), and **OVRL** (overall) MOS predictions; DNSMOS P.808 gives overall quality. | Excellent reference-free noisy/enhanced speech metric, widely used and fast. It is not a TTS expressiveness/naturalness model and should not score isolated laughs/breaths as though they were ordinary speech. | [official DNS Challenge repository](https://github.com/microsoft/DNS-Challenge), [paper](https://arxiv.org/abs/2010.15258) |
| **UTMOS22 / SpeechMOS** | Utterance-level predicted MOS for synthetic/converted speech, with strong and weak VoiceMOS systems. | Strong open TTS/VC baseline, but its human-MOS training is predominantly English; a multilingual SSL encoder does not make the MOS head multilingual-calibrated. Use system-disjoint and language-specific validation. | [paper](https://arxiv.org/abs/2204.02152), [implementation](https://github.com/tarepan/SpeechMOS) |
| **SSL-MOS** | Scalar predicted MOS from a fine-tuned self-supervised speech encoder. | Open and simple; intended for synthetic-speech MOS and can be fine-tuned. Use as an ensemble component rather than truth. | [official repository](https://github.com/nii-yamagishilab/mos-finetune-ssl), [paper](https://arxiv.org/abs/2110.02635) |
| **MOSNet** | Framewise estimates pooled to utterance MOS for TTS/voice conversion. | Classic open baseline with clear training code, but older and usually weaker out-of-domain than SSL models. | [official repository](https://github.com/lochenchou/MOSNet), [paper](https://arxiv.org/abs/1904.08352) |
| **TorchAudio SQUIM Subjective** | Predicted subjective MOS; SQUIM Objective separately estimates PESQ, STOI and SI-SDR. | Convenient official PyTorch bundle. “Subjective” inference needs a **non-matching clean reference**; it is not literally reference-free in the API despite not needing the paired original. | [official tutorial](https://docs.pytorch.org/audio/stable/tutorials/squim_tutorial.html), [paper](https://arxiv.org/abs/2304.01448) |
| **UrgentMOS** | Absolute MOS plus comparative MOS (CMOS), with unified prediction of complementary distortion, spectral, intelligibility and speaker-similarity metrics even when training labels are incomplete. | Open 2026 checkpoint/code and the strongest broad candidate found. Its human-MOS sources include at least **English, German, and Mandarin**, while additional multilingual enhancement data contribute weak/objective supervision. It still inherits incompatible source protocols, so calibrate separately per language/domain. | [checkpoint](https://huggingface.co/urgent-challenge/urgent-mos-f1c1m5dref), [code](https://github.com/vvwangvv/URGENT-MOS), [paper](https://arxiv.org/abs/2601.18438) |
| **Uni-VERSA / Uni-VERSA-Ext** | A single network for naturalness, intelligibility, speaker similarity, prosody/F0, noise/distortion and multiple objective or learned quality metrics. | Open weights and inference/training recipe. Its MOS training mixture includes **English, German, and Mandarin** corpora; broader multilingual enhancement data are mainly supervised by objective/pseudo metrics. Prefer it when more than one scalar is needed, and keep every output dimension separate. | [checkpoint](https://huggingface.co/vvwangvv/universa-ext_wavlm-base_5metric), [paper](https://arxiv.org/abs/2505.20741), [implementation](https://github.com/urgent-challenge/urgent2026_challenge_track2) |

**Multilingual pretrained shortlist:** use **IndicMOS** for seven Indian spoken languages, **Singing-SSL-MOS** for Chinese/Japanese singing, and **UrgentMOS or Uni-VERSA-Ext** for mixed speech-quality domains with English/German/Mandarin human-MOS supervision. NISQA is the lighter English/German diagnostic option. DNSMOS, UTMOS, SSL-MOS, MOSNet, and SQUIM may run on arbitrary waveforms, but their available documentation does not establish equivalent human-MOS calibration across many languages.

## E. Recommended construction for the labels in this file

Do not force all targets into one flat mutually exclusive classifier. One clip can contain speech + breath + laugh while also being excited and low quality. Use three synchronized outputs:

1. **Event timeline (multi-label, start/end/confidence):** `laugh`, `giggle`, `chuckle`, `breath`, `wheeze`, `whisper`, `shout`, `beep`, `clear_throat`, `yawn`, `sigh`, `exhale`, `cry`, `snort`, `swallow`, `moan`, `gulp`, `cough`, `sneeze`, `sniff`, `gasp`, `groan`, `grunt`, `hiccup`, `filler_uh`, `filler_um`, `agreement_hum`, `other_vocalization`.
2. **Vocal form token:** preserve the heard form when it is meaningful—e.g. `ha-ha`, `hehe`, `mmm`, `mm-hmm`, `uh`, `umm`, `oh`, `woo`—plus a broad event class. This is **not ordinary transcription**; it is a small, controlled vocalization lexicon. PodcastFillers, NVSpeech, and SynParaSpeech are the best seeds.
3. **Clip/segment attributes:** emotion/expression vector and quality vector. Prefer continuous valence/arousal/dominance or a corpus-specific emotion/intensity vector over a single “curious” class. Store human MOS separately as `mos_question`, `mos_mean`, `mos_std`, `n_ratings`, and raw ratings where legal.

### Suggested data order

- **First direct-label training:** VocalSound + Nonspeech7k + PodcastFillers event clips + MNV-17 + the 76 h human NVSpeech subset.
- **Then contextual/timestamp training:** SynParaSpeech, full PodcastFillers episodes, and NVSpeech auto-labeled data with lower sample weight.
- **Then weak-label pretraining:** FSD50K, selected high-quality AudioSet human-sound classes, and VGGSound. Do not give weak 10-second tags the same loss weight as strong timestamps.
- **Emotion head:** start emotion2vec+ and fine-tune on MSP-Podcast, ESD, EmoGator, and selected multilingual sets. EmoGator supplies the most relevant independently accessible non-lexical bursts.
- **Quality head:** keep at least two domains: NISQA/DNS/PSTN for signal quality and SOMOS/BVCC for synthetic-speech naturalness. A single merged MOS without a domain/protocol indicator will be poorly calibrated.

### Validation rules that prevent misleading results

- Split by **speaker, original source recording, and synthesis/enhancement system**, never by random derivative clip.
- Track `label_source = human | expert | crowd | weak_web | model_pseudo` and do not report them as equivalent ground truth.
- Include hard negatives: lexical speech, silence, music, animal vocalizations, and confusable pairs (`sigh`/`breath`, `grunt`/`groan`, `cry`/`scream`, `giggle`/`chuckle`, `sniff`/`inhale`).
- Evaluate event detection with onset/offset or collar-based F1, emotion with macro-F1/CCC, and MOS with utterance- and system-level SRCC/LCC plus calibration error.
- Audit languages separately. A “multilingual” web set may contain many countries but a nonverbal class; it does not prove equivalent performance on speech-embedded vocalizations in every language.
- Never interpret model-predicted MOS as a replacement for a human listening test in a new domain. Use it for filtering/ranking, then human-rate a stratified sample.

## Bottom line

For the original label list, the highest-value combination is **NVSpeech + SynParaSpeech + VocalSound + Nonspeech7k + PodcastFillers + EmoGator**, with **FSD50K/AudioSet** only as weak-label scale. For `wheezing`, use **ICBHI** as a separate clinical-domain head. For expression, use **EmoGator and MSP-Podcast**. For per-audio quality, train separate heads on **NISQA/PSTN/Tencent/TMHINT-QI** and **SOMOS/BVCC**, or run NISQA + DNSMOS + UTMOS as an explicitly model-estimated ensemble.

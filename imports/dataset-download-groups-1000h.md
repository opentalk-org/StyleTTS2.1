# Dataset download groups (approximately 1,000 h)

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

After each group verify that no metadata from data.json be lost (All send to backend) and all data was uploaded, then prune all stage1/ folder to make place for next group

Dataset-language rows belonging to the same logical source are combined before grouping. Mozilla Common Voice is the requested exception: its languages are distributed across three duration-balanced parts, with each language kept intact. Distinct FCBH APKs remain separate because each is a separate download.

Sources are sorted from the smallest to the largest by `Hours to get`, then consecutive small sources are packed without splitting a source. Groups stay at or below 1,000 h unless one source alone exceeds 1,000 h.

- Logical download sources with a positive target: **178**

- Dataset-language rows represented: **721**

- Total hours to get: **7646.7886 h**

- Download groups: **9**

## Group summary

| Group | Sources | Hours to get | Note |
|---:|---:|---:|---|
| 1 | 148 | 969.0902 |  |
| 2 | 16 | 949.2358 |  |
| 3 | 7 | 681.7995 |  |
| 4 | 2 | 907.6893 |  |
| 5 | 1 | 806.4034 |  |
| 6 | 1 | 806.6597 |  |
| 7 | 1 | 806.8715 |  |
| 8 | 1 | 831.0394 |  |
| 9 | 1 | 888.0000 |  |

## Group 1 — 969.0902 h

| Dataset / download source | Hours to get | Languages | Table rows | Included labels/configurations |
|---|---:|---:|---:|---|
| MESD | 0.1730 | `es` | 1 | — |
| French Oreau | 0.3000 | `fr` | 1 | — |
| emoUERJ | 0.3150 | `pt` | 1 | — |
| Arabic Natural Audio | 0.3840 | `ar` | 1 | — |
| FCBH/Amharic_Orthodox_Bible-1.0.apk | 0.5000 | `am` | 1 | — |
| FCBH/Arabic_OT_VDV-1.0.5.apk | 0.5000 | `ar` | 1 | — |
| FCBH/Arabic_Sharif_Bible-2.0.apk | 0.5000 | `ar` | 1 | — |
| FCBH/Arabic_VDV-1.0.apk | 0.5000 | `ar` | 1 | — |
| FCBH/Armenian_Western_Bible-1.0.apk | 0.5000 | `hyw` | 1 | — |
| FCBH/Azerbaijani_No-1.0.11.apk | 0.5000 | `az` | 1 | — |
| FCBH/Bengali_CLV-1.0.9.apk | 0.5000 | `bn` | 1 | — |
| FCBH/Bengali_ERV-1.0.13.apk | 0.5000 | `bn` | 1 | — |
| FCBH/Bengali_Mus-1.0.4.apk | 0.5000 | `bn` | 1 | — |
| FCBH/Bengali_RBV-1.0.4.apk | 0.5000 | `bn` | 1 | — |
| FCBH/Biblia_NVT_em_Portugues-1.0.1.apk | 0.5000 | `pt` | 1 | — |
| FCBH/Chinese_Cantonese-1.0.10.apk | 0.5000 | `zh-yue` | 1 | — |
| FCBH/Crimean_Tatar_IBT-1.0.1.apk | 0.5000 | `crh` | 1 | — |
| FCBH/Dutch_Arabic-1.0.3.apk | 0.5000 | `nl` | 1 | — |
| FCBH/Dutch_Bible-1.0.1.apk | 0.5000 | `nl` | 1 | — |
| FCBH/English_ESV-1.0.16.apk | 0.5000 | `en` | 1 | — |
| FCBH/English_ESV-1.0.17.apk | 0.5000 | `en` | 1 | — |
| FCBH/English_ESV_NT-1.0.1.apk | 0.5000 | `en` | 1 | — |
| FCBH/English_ESV_NT-1.0.apk | 0.5000 | `en` | 1 | — |
| FCBH/English_Full_WEB-1.0.32.apk | 0.5000 | `en` | 1 | — |
| FCBH/English_KJV_Bible-1.0.9.apk | 0.5000 | `en` | 1 | — |
| FCBH/English_NT_WEB-1.0.36.apk | 0.5000 | `en` | 1 | — |
| FCBH/Espanol_BDA-1.0.8.apk | 0.5000 | `es` | 1 | — |
| FCBH/Espanol_WTC-1.0.20.apk | 0.5000 | `es` | 1 | — |
| FCBH/French_1910_LS-1.0.7.apk | 0.5000 | `fr` | 1 | — |
| FCBH/German_GNB-1.0.2.apk | 0.5000 | `de` | 1 | — |
| FCBH/Greek_EPT-1.0.5.apk | 0.5000 | `el` | 1 | — |
| FCBH/Greek_OT_AVS-1.0.5.apk | 0.5000 | `el` | 1 | — |
| FCBH/Hindi_ERV-1.0.18.apk | 0.5000 | `hi` | 1 | — |
| FCBH/Hindi_KJV-1.0.1.apk | 0.5000 | `hi` | 1 | — |
| FCBH/Hindi_Sab_Ki-1.0.1.apk | 0.5000 | `hi` | 1 | — |
| FCBH/Hungarian_RHNT-1.0.4.apk | 0.5000 | `hu` | 1 | — |
| FCBH/Indonesian_Bible-1.0.apk | 0.5000 | `id` | 1 | — |
| FCBH/Indonesian_IND_Bible-1.0.1.apk | 0.5000 | `id` | 1 | — |
| FCBH/Indonesian_NTV-1.0.18.apk | 0.5000 | `id` | 1 | — |
| FCBH/Indonesian_TSI-1.0.9.apk | 0.5000 | `id` | 1 | — |
| FCBH/Kannada_Contemporary_Bible-1.0.apk | 0.5000 | `kn` | 1 | — |
| FCBH/Kannada_ERV_Bible-1.0.13.apk | 0.5000 | `kn` | 1 | — |
| FCBH/Karakalpak-1.0.10.apk | 0.5000 | `kaa` | 1 | — |
| FCBH/Kazakh-1.0.11.apk | 0.5000 | `kk` | 1 | — |
| FCBH/Korean_North_Chosun_Bible-1.0.4.apk | 0.5000 | `ko` | 1 | — |
| FCBH/Kurdish_Behdini-1.0.apk | 0.5000 | `ku` | 1 | — |
| FCBH/Latgalian_Bible-1.0.apk | 0.5000 | `ltg` | 1 | — |
| FCBH/Malayalam-1.0.15.apk | 0.5000 | `ml` | 1 | — |
| FCBH/Mandarin_UNV-1.0.12.apk | 0.5000 | `zh` | 1 | — |
| FCBH/Marathi_BLI-1.0.13.apk | 0.5000 | `mr` | 1 | — |
| FCBH/Nigerian_Pidgin_SC-1.0.apk | 0.5000 | `pcm` | 1 | — |
| FCBH/Oromo_West_Central-1.0.1.apk | 0.5000 | `om` | 1 | — |
| FCBH/Oromo_West_Central-1.0.apk | 0.5000 | `om` | 1 | — |
| FCBH/Romanian_Bible-1.0.1.apk | 0.5000 | `ro` | 1 | — |
| FCBH/Russian_1876_Synodal_Bible-1.0.apk | 0.5000 | `ru` | 1 | — |
| FCBH/Russian_CARS-1.0.2.apk | 0.5000 | `ru` | 1 | — |
| FCBH/Russian_NRT_Bible-1.0.apk | 0.5000 | `ru` | 1 | — |
| FCBH/Russian_NT_SB76-1.0.11.apk | 0.5000 | `ru` | 1 | — |
| FCBH/Spanish_NTV_Bible-1.0.apk | 0.5000 | `es` | 1 | — |
| FCBH/Spanish_NVI_Bible-1.0.1.apk | 0.5000 | `es` | 1 | — |
| FCBH/Spanish_PDT_Bible-1.0.apk | 0.5000 | `es` | 1 | — |
| FCBH/Spanish_PDT_Bible-2.0.apk | 0.5000 | `es` | 1 | — |
| FCBH/Swedish_Bible-1.0.apk | 0.5000 | `sv` | 1 | — |
| FCBH/Swedish_Bible-1.1.0.apk | 0.5000 | `sv` | 1 | — |
| FCBH/Swedish_SSF-1.0.4.apk | 0.5000 | `sv` | 1 | — |
| FCBH/Tamil_Contemporary_Bible-1.0.apk | 0.5000 | `ta` | 1 | — |
| FCBH/Tamil_ERV-1.0.12.apk | 0.5000 | `ta` | 1 | — |
| FCBH/Tatar-1.0.8.apk | 0.5000 | `tt` | 1 | — |
| FCBH/Telugu_Bible-1.0.apk | 0.5000 | `te` | 1 | — |
| FCBH/Telugu_ERV-1.0.14.apk | 0.5000 | `te` | 1 | — |
| FCBH/Thai_TSV-1.0.11.apk | 0.5000 | `th` | 1 | — |
| FCBH/Tigrinya_BSE-1.0.6.apk | 0.5000 | `ti` | 1 | — |
| FCBH/Turkish_BST-1.0.12.apk | 0.5000 | `tr` | 1 | — |
| FCBH/Turkish_Halk_Dilinde-1.0.8.apk | 0.5000 | `tr` | 1 | — |
| FCBH/Turkmen-1.0.3.apk | 0.5000 | `tk` | 1 | — |
| FCBH/Turkmen-1.0.9.apk | 0.5000 | `tk` | 1 | — |
| FCBH/Ukrainian_Bible-1.0.3.apk | 0.5000 | `uk` | 1 | — |
| FCBH/Urdu_Bible_(PBS)-1.0.1.apk | 0.5000 | `ur` | 1 | — |
| FCBH/Urdu_Geo_Version-1.0.1.apk | 0.5000 | `ur` | 1 | — |
| FCBH/Urdu_KJV-1.0.1.apk | 0.5000 | `ur` | 1 | — |
| FCBH/Urdu_NT_India-1.0.18.apk | 0.5000 | `ur` | 1 | — |
| FCBH/Vietnamese_Contemporary_Bible-1.0.apk | 0.5000 | `vi` | 1 | — |
| VoxPopuli accented English | 0.5000 | `en` | 1 | — |
| Kannada Emotional Speech | 0.5229 | `kn` | 1 | — |
| MSA-Moroccan | 0.6633 | `ary` | 1 | — |
| Konkani Bible Audio | 0.7167 | `gom` | 1 | — |
| KUET/KBES | 0.7500 | `bn` | 1 | — |
| CaFE | 1.1552 | `fr` | 1 | — |
| MrlolDev/voxtral-emotion-speech | 1.1588 | `en` | 1 | — |
| BANSpEmo | 1.3830 | `bn` | 1 | — |
| JL Corpus | 1.4000 | `en` | 1 | — |
| BanglaSER | 1.4260 | `bn` | 1 | — |
| SOREVA | 1.5000 | `af`, `pcm`, `sw` | 3 | SOREVA Afrikaans test set; SOREVA Kiswahili test set; SOREVA Pidgin test set |
| RAVDESS speech | 1.7080 | `en` | 1 | — |
| Casablanca Morocco | 2.0077 | `ary` | 1 | — |
| MDER-MA | 2.0750 | `ary` | 1 | — |
| yfish/WESR-Bench | 2.2878 | `en`, `zh` | 2 | yfish/WESR-Bench en; yfish/WESR-Bench zh |
| EMNS | 2.3157 | `en` | 1 | — |
| Thorsten Emotional | 2.9167 | `de` | 1 | — |
| nEMO | 3.0000 | `pl` | 1 | — |
| ASED | 3.0700 | `am` | 1 | — |
| ShEMO | 3.4167 | `fa` | 1 | — |
| synthbot/pony-singing | 3.4613 | `en` | 1 | — |
| RESD Annotated | 3.5000 | `ru` | 1 | — |
| IAMCB/elise-clone | 3.5870 | `en` | 1 | — |
| projecte-aina/LaFrescat | 3.7500 | `ca` | 1 | — |
| laion/synthetic_vocal_burts_dramabox | 4.1644 | `en` | 1 | — |
| EMOVIE | 4.1800 | `zh` | 1 | — |
| Magic Data scripted Malay | 4.7069 | `ms` | 1 | — |
| Magic Data Egyptian conversational | 5.1596 | `arz` | 1 | — |
| NAC-v1.0 / UD Naija Spoken Corpus derivative | 6.0225 | `pcm` | 1 | — |
| ESCorpus-PE | 6.2508 | `es` | 1 | — |
| EmoV-DB | 7.0000 | `en` | 1 | — |
| Crimean Tatar TTS | 7.7509 | `crh` | 1 | — |
| PMEmo2019 | 8.3800 | `en` | 1 | — |
| MikhailT/hifi-tts clean | 10.0000 | `en` | 1 | — |
| SyntAct | 11.1437 | `de` | 1 | — |
| ASVP-ESD | 12.2124 | `en`, `fr`, `ru`, `zh` | 4 | ASVP-ESD en; ASVP-ESD fr; ASVP-ESD ru; ASVP-ESD zh |
| MELD | 13.0000 | `en` | 1 | — |
| MLEnd Spoken Numerals | 14.3089 | `en` | 1 | — |
| DragonLine/ksponspeech_04 | 14.6315 | `ko` | 1 | — |
| Phonetico | 14.7000 | `ti` | 1 | — |
| Sh1man/elevenlabs | 14.9316 | `ru` | 1 | — |
| Yoyo ASR | 17.4024 | `azb` | 1 | — |
| Crimean Tatar Audiobooks | 18.3667 | `crh` | 1 | — |
| laion/more-synthetic-vocalbursts-raw | 18.3914 | `en` | 1 | — |
| CMU Haitian | 18.5000 | `ht` | 1 | — |
| ALFFA | 20.0000 | `am` | 1 | — |
| jp1924/KoreaSpeech | 20.0000 | `ko` | 1 | — |
| NPSC | 20.0000 | `no` | 1 | — |
| joujiboi/japanese-anime-speech-v2 | 20.6667 | `ja` | 1 | — |
| FCBH Hakka Bible New Testament | 23.9766 | `hak` | 1 | — |
| LJSpeech | 24.0000 | `en` | 1 | — |
| FLORAS | 26.8500 | `ar`, `az`, `bg`, `bn`, `ca`, `cs`, `cy`, `da`, `de`, `el`, `eo`, `es`, `et`, `eu`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `ka`, `ku`, `ky`, `la`, `lt`, `mi`, `ms`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sr`, `sv`, `ta`, `th`, `tr`, `uk`, `ur`, `uz`, `vi`, `zh` | 46 | FLORAS `ar`; FLORAS `az`; FLORAS `bg`; FLORAS `bn`; FLORAS `ca`; FLORAS `cs`; FLORAS `cy`; FLORAS `da`; FLORAS `de`; FLORAS `el`; FLORAS `eo`; FLORAS `es`; FLORAS `et`; FLORAS `eu`; FLORAS `fi`; FLORAS `fr`; FLORAS `hi`; FLORAS `hr`; FLORAS `hu`; FLORAS `id`; FLORAS `it`; FLORAS `ja`; FLORAS `ka`; FLORAS `ku`; FLORAS `ky`; FLORAS `la`; FLORAS `lt`; FLORAS `mi`; FLORAS `ms`; FLORAS `nl`; FLORAS `pl`; FLORAS `pt`; FLORAS `ro`; FLORAS `ru`; FLORAS `sk`; FLORAS `sl`; FLORAS `sr`; FLORAS `sv`; FLORAS `ta`; FLORAS `th`; FLORAS `tr`; FLORAS `uk`; FLORAS `ur`; FLORAS `uz`; FLORAS `vi`; FLORAS `zh` |
| FCBH Burmese MSB New Testament | 27.3061 | `my` | 1 | — |
| ShoukanLabs/AniSpeech | 28.6495 | `en` | 1 | — |
| ESD | 29.0000 | `en`, `zh` | 2 | ESD en; ESD zh |
| SASPEECH | 30.0000 | `he` | 1 | — |
| THAI-SER | 30.0000 | `th` | 1 | — |
| laion/vocal-burst-db | 30.8330 | `en` | 1 | — |
| alexandrainst/ftspeech | 32.4427 | `da` | 1 | — |
| FCBH Shan TBS New Testament | 39.8381 | `shn` | 1 | — |
| YODAS | 39.8800 | `af`, `am`, `ar`, `as`, `az`, `ba`, `be`, `bg`, `bn`, `bs`, `ca`, `cs`, `cy`, `da`, `de`, `el`, `en`, `eo`, `es`, `et`, `eu`, `fa`, `fi`, `fo`, `fr`, `ga`, `gn`, `gu`, `hi`, `hr`, `ht`, `hu`, `hy`, `ia`, `id`, `is`, `it`, `ja`, `ka`, `kk`, `kn`, `ko`, `ku`, `ky`, `la`, `lb`, `lt`, `lv`, `mi`, `mk`, `ml`, `mn`, `mr`, `ms`, `my`, `ne`, `nl`, `no`, `om`, `or`, `pa`, `pl`, `ps`, `pt`, `qu`, `ro`, `ru`, `sd`, `si`, `sk`, `sl`, `sq`, `sr`, `sv`, `sw`, `ta`, `te`, `th`, `ti`, `tk`, `tn`, `tr`, `tt`, `ug`, `uk`, `ur`, `uz`, `vi`, `zh` | 89 | YODAS `af`; YODAS `am`; YODAS `ar`; YODAS `as`; YODAS `az`; YODAS `ba`; YODAS `be`; YODAS `bg`; YODAS `bn`; YODAS `bs`; YODAS `ca`; YODAS `cs`; YODAS `cy`; YODAS `da`; YODAS `de`; YODAS `el`; YODAS `en`; YODAS `eo`; YODAS `es`; YODAS `et`; YODAS `eu`; YODAS `fa`; YODAS `fi`; YODAS `fo`; YODAS `fr`; YODAS `ga`; YODAS `gn`; YODAS `gu`; YODAS `hi`; YODAS `hr`; YODAS `ht`; YODAS `hu`; YODAS `hy`; YODAS `ia`; YODAS `id`; YODAS `is`; YODAS `it`; YODAS `ja`; YODAS `ka`; YODAS `kk`; YODAS `kn`; YODAS `ko`; YODAS `ku`; YODAS `ky`; YODAS `la`; YODAS `lb`; YODAS `lt`; YODAS `lv`; YODAS `mi`; YODAS `mk`; YODAS `ml`; YODAS `mn`; YODAS `mr`; YODAS `ms`; YODAS `my`; YODAS `ne`; YODAS `nl`; YODAS `no`; YODAS `om`; YODAS `or`; YODAS `pa`; YODAS `pl`; YODAS `ps`; YODAS `pt`; YODAS `qu`; YODAS `ro`; YODAS `ru`; YODAS `sd`; YODAS `si`; YODAS `sk`; YODAS `sl`; YODAS `sq`; YODAS `sr`; YODAS `sv`; YODAS `sw`; YODAS `ta`; YODAS `te`; YODAS `th`; YODAS `ti`; YODAS `tk`; YODAS `tn`; YODAS `tr`; YODAS `tt`; YODAS `ug`; YODAS `uk`; YODAS `ur`; YODAS `uz`; YODAS `vi`; YODAS `zh` |
| AISHELL-3 | 40.0000 | `zh` | 1 | — |
| facebook/multilingual_librispeech | 40.0000 | `de`, `en`, `es`, `fr`, `it`, `nl`, `pl`, `pt` | 8 | facebook/multilingual_librispeech de; facebook/multilingual_librispeech en; facebook/multilingual_librispeech es; facebook/multilingual_librispeech fr; facebook/multilingual_librispeech it; facebook/multilingual_librispeech nl; facebook/multilingual_librispeech pl; facebook/multilingual_librispeech pt |
| simon3000/starrail-voice | 40.0000 | `en`, `ja`, `ko`, `zh` | 4 | simon3000/starrail-voice en; simon3000/starrail-voice ja; simon3000/starrail-voice ko; simon3000/starrail-voice zh |
| MWA Western Armenian | 42.0000 | `hyw` | 1 | — |
| VCTK | 44.0000 | `en` | 1 | — |

## Group 2 — 949.2358 h

| Dataset / download source | Hours to get | Languages | Table rows | Included labels/configurations |
|---|---:|---:|---:|---|
| Karakalpak Speech Corpus | 50.0000 | `kaa` | 1 | — |
| LibriSpeech train-clean-360 | 50.0000 | `en` | 1 | — |
| sleeping-ai/11Labs | 50.0000 | `en` | 1 | — |
| TED-LIUM | 50.0000 | `en` | 1 | — |
| TurkmenSpeech | 54.4204 | `tk` | 1 | — |
| ParlaSpeech-HR | 54.5525 | `hr` | 1 | — |
| KSC | 55.3027 | `kk` | 1 | — |
| OpenSLR 52 Sinhala ASR | 55.5157 | `si` | 1 | — |
| FCBH South Azerbaijani Bible | 56.0727 | `azb` | 1 | — |
| LocalDoc ASR | 57.0797 | `az` | 1 | — |
| Althingi parliamentary corpus | 58.0208 | `is` | 1 | — |
| LOD_Claude synthetic Luxembourgish | 61.4300 | `lb` | 1 | — |
| Vox Classica | 73.0000 | `la` | 1 | — |
| facebook/voxpopuli | 74.0213 | `cs`, `de`, `en`, `es`, `et`, `fi`, `fr`, `hr`, `hu`, `it`, `lt`, `nl`, `pl`, `ro`, `sk`, `sl` | 16 | facebook/voxpopuli cs; facebook/voxpopuli de; facebook/voxpopuli en; facebook/voxpopuli es; facebook/voxpopuli et; facebook/voxpopuli fi; facebook/voxpopuli fr; facebook/voxpopuli hr; facebook/voxpopuli hu; facebook/voxpopuli it; facebook/voxpopuli lt; facebook/voxpopuli nl; facebook/voxpopuli pl; facebook/voxpopuli ro; facebook/voxpopuli sk; facebook/voxpopuli sl |
| OpenSLR 125 | 74.8200 | `fo` | 1 | — |
| DMC-ykfx33/nsfw_tts_dataset_30speakers | 75.0000 | `en` | 30 | DMC-ykfx33/nsfw_tts_dataset_30speakers/Caspian; DMC-ykfx33/nsfw_tts_dataset_30speakers/Darius; DMC-ykfx33/nsfw_tts_dataset_30speakers/Declan; DMC-ykfx33/nsfw_tts_dataset_30speakers/Elara; DMC-ykfx33/nsfw_tts_dataset_30speakers/Elias; DMC-ykfx33/nsfw_tts_dataset_30speakers/Esme; DMC-ykfx33/nsfw_tts_dataset_30speakers/Ezra; DMC-ykfx33/nsfw_tts_dataset_30speakers/Freya; DMC-ykfx33/nsfw_tts_dataset_30speakers/Genevieve; DMC-ykfx33/nsfw_tts_dataset_30speakers/Gideon; DMC-ykfx33/nsfw_tts_dataset_30speakers/Isolde; DMC-ykfx33/nsfw_tts_dataset_30speakers/Jasper; DMC-ykfx33/nsfw_tts_dataset_30speakers/Jensen; DMC-ykfx33/nsfw_tts_dataset_30speakers/Jocelyn; DMC-ykfx33/nsfw_tts_dataset_30speakers/Kael; DMC-ykfx33/nsfw_tts_dataset_30speakers/Kaia; DMC-ykfx33/nsfw_tts_dataset_30speakers/Killian; DMC-ykfx33/nsfw_tts_dataset_30speakers/Liora; DMC-ykfx33/nsfw_tts_dataset_30speakers/Maddox; DMC-ykfx33/nsfw_tts_dataset_30speakers/Persephone; DMC-ykfx33/nsfw_tts_dataset_30speakers/Rowan; DMC-ykfx33/nsfw_tts_dataset_30speakers/Ryder; DMC-ykfx33/nsfw_tts_dataset_30speakers/Seraphina; DMC-ykfx33/nsfw_tts_dataset_30speakers/Silas; DMC-ykfx33/nsfw_tts_dataset_30speakers/Sloane; DMC-ykfx33/nsfw_tts_dataset_30speakers/Soren; DMC-ykfx33/nsfw_tts_dataset_30speakers/Thorne; DMC-ykfx33/nsfw_tts_dataset_30speakers/Ursula; DMC-ykfx33/nsfw_tts_dataset_30speakers/Vaughn; DMC-ykfx33/nsfw_tts_dataset_30speakers/Zane |

## Group 3 — 681.7995 h

| Dataset / download source | Hours to get | Languages | Table rows | Included labels/configurations |
|---|---:|---:|---:|---|
| fixie-ai/soda-audio | 75.0000 | `en` | 1 | — |
| simon3000/genshin-voice | 80.0000 | `en`, `ja`, `ko`, `zh` | 4 | simon3000/genshin-voice en; simon3000/genshin-voice ja; simon3000/genshin-voice ko; simon3000/genshin-voice zh |
| amphion/Emilia | 90.0000 | `de`, `en`, `fr`, `ja`, `ko`, `zh` | 6 | amphion/Emilia de; amphion/Emilia en; amphion/Emilia fr; amphion/Emilia ja; amphion/Emilia ko; amphion/Emilia zh |
| Meta Omnilingual ASR | 99.9188 | `ary`, `arz`, `gom`, `haw`, `pnb`, `ps`, `si`, `sq` | 8 | Meta Omnilingual ASR `aln_Latn`; Meta Omnilingual ASR `ary_Arab`; Meta Omnilingual ASR `arz_Arab`; Meta Omnilingual ASR `gom_Deva`; Meta Omnilingual ASR `haw_Latn`; Meta Omnilingual ASR `pbu_Arab,pbt_Arab`; Meta Omnilingual ASR `pnb_Arab`; Meta Omnilingual ASR `sin_Sinh` |
| NCHLT | 103.2052 | `af`, `tn` | 2 | — |
| ylacombe/cml-tts | 110.0000 | `de`, `es`, `fr`, `it`, `nl`, `pl`, `pt` | 7 | ylacombe/cml-tts de; ylacombe/cml-tts es; ylacombe/cml-tts fr; ylacombe/cml-tts it; ylacombe/cml-tts nl; ylacombe/cml-tts pl; ylacombe/cml-tts pt |
| WAXAL | 123.6755 | `am`, `om`, `pcm`, `sw`, `ti` | 5 | WAXAL `amh_asr`; WAXAL `orm_asr`; WAXAL `pcm_tts`; WAXAL `swa_tts`; WAXAL `tir_asr` |

## Group 4 — 907.6893 h

| Dataset / download source | Hours to get | Languages | Table rows | Included labels/configurations |
|---|---:|---:|---:|---|
| OpenBibleTTS | 445.4073 | `ar`, `as`, `bn`, `gu`, `hi`, `ht`, `kn`, `mi`, `ml`, `mr`, `ne`, `om`, `pa`, `pl`, `sw`, `ta`, `te`, `tr`, `uk`, `ur`, `vi` | 21 | OpenBibleTTS `Arabic Standard`; OpenBibleTTS `Assamese`; OpenBibleTTS `Bengali`; OpenBibleTTS `Gujarati`; OpenBibleTTS `Haitian Creole`; OpenBibleTTS `Hindi`; OpenBibleTTS `Kannada`; OpenBibleTTS `Malayalam`; OpenBibleTTS `Maori`; OpenBibleTTS `Marathi`; OpenBibleTTS `Nepali`; OpenBibleTTS `Oromo`; OpenBibleTTS `Polish`; OpenBibleTTS `Punjabi`; OpenBibleTTS `Swahili`; OpenBibleTTS `Tamil`; OpenBibleTTS `Telugu`; OpenBibleTTS `Turkish`; OpenBibleTTS `Ukrainian`; OpenBibleTTS `Urdu`; OpenBibleTTS `Vietnamese` |
| espnet/mms_ulab_v2 | 462.2820 | `af`, `am`, `an`, `ar`, `ary`, `arz`, `as`, `az`, `azb`, `be`, `bg`, `bn`, `bs`, `ca`, `crh`, `cs`, `cy`, `da`, `el`, `es`, `fa`, `fi`, `ga`, `gom`, `gu`, `haw`, `he`, `hi`, `hr`, `hu`, `hy`, `hyw`, `id`, `is`, `it`, `ka`, `kaa`, `kk`, `kn`, `ko`, `ku`, `ky`, `lb`, `lt`, `mi`, `mk`, `ml`, `mr`, `ms`, `mt`, `my`, `nl`, `om`, `pa`, `pl`, `pnb`, `ps`, `pt`, `qu`, `ro`, `ru`, `sd`, `shn`, `si`, `sk`, `skr`, `sl`, `sq`, `sr`, `sv`, `te`, `ti`, `tk`, `tn`, `tt`, `ug`, `ur`, `vi` | 78 | espnet/mms_ulab_v2 `afr`; espnet/mms_ulab_v2 `als,aln`; espnet/mms_ulab_v2 `amh`; espnet/mms_ulab_v2 `arb`; espnet/mms_ulab_v2 `arg`; espnet/mms_ulab_v2 `ary`; espnet/mms_ulab_v2 `arz`; espnet/mms_ulab_v2 `asm`; espnet/mms_ulab_v2 `azb`; espnet/mms_ulab_v2 `azj`; espnet/mms_ulab_v2 `bel`; espnet/mms_ulab_v2 `ben`; espnet/mms_ulab_v2 `bos`; espnet/mms_ulab_v2 `bul`; espnet/mms_ulab_v2 `cat`; espnet/mms_ulab_v2 `ces`; espnet/mms_ulab_v2 `crh`; espnet/mms_ulab_v2 `cym`; espnet/mms_ulab_v2 `dan`; espnet/mms_ulab_v2 `ell`; espnet/mms_ulab_v2 `fin`; espnet/mms_ulab_v2 `gaz`; espnet/mms_ulab_v2 `gle`; espnet/mms_ulab_v2 `gom`; espnet/mms_ulab_v2 `guj`; espnet/mms_ulab_v2 `haw`; espnet/mms_ulab_v2 `heb`; espnet/mms_ulab_v2 `hin`; espnet/mms_ulab_v2 `hrv`; espnet/mms_ulab_v2 `hun`; espnet/mms_ulab_v2 `hye`; espnet/mms_ulab_v2 `hyw`; espnet/mms_ulab_v2 `ind`; espnet/mms_ulab_v2 `isl`; espnet/mms_ulab_v2 `ita`; espnet/mms_ulab_v2 `kaa`; espnet/mms_ulab_v2 `kan`; espnet/mms_ulab_v2 `kat`; espnet/mms_ulab_v2 `kaz`; espnet/mms_ulab_v2 `kir`; espnet/mms_ulab_v2 `kmr`; espnet/mms_ulab_v2 `kor`; espnet/mms_ulab_v2 `lit`; espnet/mms_ulab_v2 `ltz`; espnet/mms_ulab_v2 `mal`; espnet/mms_ulab_v2 `mar`; espnet/mms_ulab_v2 `mkd`; espnet/mms_ulab_v2 `mlt`; espnet/mms_ulab_v2 `mri`; espnet/mms_ulab_v2 `mya`; espnet/mms_ulab_v2 `nld`; espnet/mms_ulab_v2 `pan`; espnet/mms_ulab_v2 `pbu,pbt`; espnet/mms_ulab_v2 `pes`; espnet/mms_ulab_v2 `pnb`; espnet/mms_ulab_v2 `pol`; espnet/mms_ulab_v2 `por`; espnet/mms_ulab_v2 `qxp`; espnet/mms_ulab_v2 `ron`; espnet/mms_ulab_v2 `rus`; espnet/mms_ulab_v2 `shn`; espnet/mms_ulab_v2 `sin`; espnet/mms_ulab_v2 `skr`; espnet/mms_ulab_v2 `slk`; espnet/mms_ulab_v2 `slv`; espnet/mms_ulab_v2 `snd`; espnet/mms_ulab_v2 `spa`; espnet/mms_ulab_v2 `srp`; espnet/mms_ulab_v2 `swe`; espnet/mms_ulab_v2 `tat`; espnet/mms_ulab_v2 `tel`; espnet/mms_ulab_v2 `tir`; espnet/mms_ulab_v2 `tsn`; espnet/mms_ulab_v2 `tuk`; espnet/mms_ulab_v2 `uig`; espnet/mms_ulab_v2 `urd`; espnet/mms_ulab_v2 `vie`; espnet/mms_ulab_v2 `zsm` |

## Group 5 — 806.4034 h

| Dataset / download source | Hours to get | Languages | Table rows | Included labels/configurations |
|---|---:|---:|---:|---|
| Mozilla Common Voice — part 3/3 | 806.4034 | `as`, `az`, `be`, `bn`, `ca`, `en`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `ht`, `hu`, `ia`, `it`, `ja`, `ku`, `mk`, `mn`, `ms`, `no`, `pcm`, `ps`, `pt`, `skr`, `sw`, `tt`, `ug`, `uk` | 30 | CV26; CV26 Kurmanji (`kmr`) |

## Group 6 — 806.6597 h

| Dataset / download source | Hours to get | Languages | Table rows | Included labels/configurations |
|---|---:|---:|---:|---|
| Mozilla Common Voice — part 2/3 | 806.6597 | `af`, `ar`, `cv`, `da`, `de`, `eo`, `eu`, `id`, `is`, `ko`, `lt`, `lv`, `ml`, `mt`, `pa`, `pl`, `qu`, `ro`, `sd`, `sk`, `sl`, `sq`, `sv`, `ta`, `te`, `th`, `tn`, `ur`, `uz` | 30 | CV Spontaneous Gheg; CV26; CV26 Puno Quechua (`qxp`); CV26 Scripted |

## Group 7 — 806.8715 h

| Dataset / download source | Hours to get | Languages | Table rows | Included labels/configurations |
|---|---:|---:|---:|---|
| Mozilla Common Voice — part 1/3 | 806.8715 | `am`, `an`, `ba`, `bg`, `cs`, `cy`, `el`, `fa`, `ga`, `gn`, `he`, `hy`, `ka`, `kk`, `ky`, `ltg`, `mr`, `ne`, `nl`, `om`, `or`, `ru`, `sr`, `ti`, `tk`, `tr`, `vi`, `zh`, `zh-yue` | 29 | CV26 |

## Group 8 — 831.0394 h

| Dataset / download source | Hours to get | Languages | Table rows | Included labels/configurations |
|---|---:|---:|---:|---|
| OWSMv4 cleaned YODAS | 831.0394 | `af`, `am`, `ar`, `as`, `az`, `be`, `bg`, `bn`, `ca`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `eu`, `fa`, `fi`, `fr`, `gu`, `hi`, `hr`, `ht`, `hu`, `hy`, `id`, `is`, `it`, `ja`, `ka`, `kk`, `kn`, `ko`, `lb`, `lt`, `lv`, `mk`, `ml`, `mn`, `mr`, `ne`, `nl`, `pa`, `pl`, `ps`, `pt`, `ro`, `ru`, `sd`, `si`, `sk`, `sl`, `sr`, `sv`, `sw`, `ta`, `te`, `th`, `tk`, `tr`, `tt`, `uk`, `ur`, `uz`, `vi`, `zh` | 68 | OWSMv4 cleaned YODAS `afr`; OWSMv4 cleaned YODAS `amh`; OWSMv4 cleaned YODAS `ara`; OWSMv4 cleaned YODAS `asm`; OWSMv4 cleaned YODAS `aze`; OWSMv4 cleaned YODAS `bel`; OWSMv4 cleaned YODAS `ben`; OWSMv4 cleaned YODAS `bul`; OWSMv4 cleaned YODAS `cat`; OWSMv4 cleaned YODAS `ces`; OWSMv4 cleaned YODAS `cym`; OWSMv4 cleaned YODAS `dan`; OWSMv4 cleaned YODAS `deu`; OWSMv4 cleaned YODAS `ell`; OWSMv4 cleaned YODAS `eng`; OWSMv4 cleaned YODAS `est`; OWSMv4 cleaned YODAS `eus`; OWSMv4 cleaned YODAS `fas`; OWSMv4 cleaned YODAS `fin`; OWSMv4 cleaned YODAS `fra`; OWSMv4 cleaned YODAS `guj`; OWSMv4 cleaned YODAS `hat`; OWSMv4 cleaned YODAS `hin`; OWSMv4 cleaned YODAS `hrv`; OWSMv4 cleaned YODAS `hun`; OWSMv4 cleaned YODAS `hye`; OWSMv4 cleaned YODAS `ind`; OWSMv4 cleaned YODAS `isl`; OWSMv4 cleaned YODAS `ita`; OWSMv4 cleaned YODAS `jpn`; OWSMv4 cleaned YODAS `kan`; OWSMv4 cleaned YODAS `kat`; OWSMv4 cleaned YODAS `kaz`; OWSMv4 cleaned YODAS `kor`; OWSMv4 cleaned YODAS `lav`; OWSMv4 cleaned YODAS `lit`; OWSMv4 cleaned YODAS `ltz`; OWSMv4 cleaned YODAS `mal`; OWSMv4 cleaned YODAS `mar`; OWSMv4 cleaned YODAS `mkd`; OWSMv4 cleaned YODAS `mon`; OWSMv4 cleaned YODAS `nep`; OWSMv4 cleaned YODAS `nld`; OWSMv4 cleaned YODAS `pan`; OWSMv4 cleaned YODAS `pol`; OWSMv4 cleaned YODAS `por`; OWSMv4 cleaned YODAS `pus`; OWSMv4 cleaned YODAS `ron`; OWSMv4 cleaned YODAS `rus`; OWSMv4 cleaned YODAS `sin`; OWSMv4 cleaned YODAS `slk`; OWSMv4 cleaned YODAS `slv`; OWSMv4 cleaned YODAS `snd`; OWSMv4 cleaned YODAS `spa`; OWSMv4 cleaned YODAS `srp`; OWSMv4 cleaned YODAS `swa`; OWSMv4 cleaned YODAS `swe`; OWSMv4 cleaned YODAS `tam`; OWSMv4 cleaned YODAS `tat`; OWSMv4 cleaned YODAS `tel`; OWSMv4 cleaned YODAS `tha`; OWSMv4 cleaned YODAS `tuk`; OWSMv4 cleaned YODAS `tur`; OWSMv4 cleaned YODAS `ukr`; OWSMv4 cleaned YODAS `urd`; OWSMv4 cleaned YODAS `uzb`; OWSMv4 cleaned YODAS `vie`; OWSMv4 cleaned YODAS `zho` |

## Group 9 — 888.0000 h

| Dataset / download source | Hours to get | Languages | Table rows | Included labels/configurations |
|---|---:|---:|---:|---|
| FLEURS | 888.0000 | `af`, `am`, `ar`, `as`, `az`, `be`, `bg`, `bn`, `bs`, `ca`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `fa`, `fi`, `fr`, `ga`, `gu`, `he`, `hi`, `hr`, `hu`, `hy`, `id`, `is`, `it`, `ja`, `ka`, `kk`, `kn`, `ko`, `ky`, `lb`, `lt`, `lv`, `mi`, `mk`, `ml`, `mn`, `mr`, `ms`, `mt`, `ne`, `nl`, `no`, `om`, `or`, `pa`, `pl`, `ps`, `pt`, `ro`, `ru`, `sd`, `sk`, `sl`, `sr`, `sv`, `sw`, `ta`, `te`, `th`, `tr`, `uk`, `ur`, `uz`, `vi`, `zh`, `zh-yue` | 74 | — |

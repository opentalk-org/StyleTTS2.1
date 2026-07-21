for each importated data, you need to import/upload to backend:
- segments (if transcripts are per audio: create full audio segment, set src "dataset", score, accuracy None)
- alignment of words if exist to segments
- audio (24 khz, 24 bit, wav, mono)
- style prompt (if single keywords put them and use ", " as separator, if prompt place as it is)
- voice prompt (if single keywords put them and use ", " as separator, if prompt place as it is)
- mos as score (None if not, otherwise value)
- accuracy as some number if there is transcription accuracy metric (can be not normalized) otherwise none.
- for each dataset create dataset unique entry.
- speaker_id (create speaker and assign it to audio) "dataset_unique_name_{speaker_id}" (left none if none in labels, don't invent)
- if audio contain some effects (laught, cry, moan, etc.) then add them as <do prompt="cry"/> to transcript at valid place.
- language (shorcut like pl, en-us, etc.)
- import every possible other data and original format data as "metadata" of audio file.

Don't import more then 50h of audio per language per dataset. use tqdm for progress of upload do backend. run all imports inside tmux. use nix setup. prefer to download more than target hours, then remove data to have the biggest amount of speakers. avoid fully filling disk so balance it, but prefer as many voices as possible over quantity of single voice.

<do prompt=""/> is something that can be assigned a timing, things like style/voice are only hearable over some texts, the <do/> is real audio.
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
woo,ummm,ohhh,hhmmm, wwwmm

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

### Voice Prompt
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
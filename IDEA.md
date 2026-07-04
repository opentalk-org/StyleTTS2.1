- postgresql db with all data, each audio have only one row

bucket_file:
- id
- path
- size
- used_bytes

audio_file:
- id
- name
- bucket_file (always prefetched from db)
- byte_offset
- byte_length
- duration
- segments (jsonb) - (texts, voices, timestamps, etc.)
- metadata (jsonb)
- virtual

dataset:
- id
- audio_files

voice:
- id
- name

checkpoint:
- id
- name
- path
- type_ (str)
- metadata (jsonb)

extra_file:
- id
- name
- path
- type_ (str)
- metadata (jsonb)

config:
- id
- name
- type_
- metadata (jsonb)

initialization:
- id
- is_initialized

workflow:
- id
- data (jsonb)


load audio -> vad split audio -> audio diarization and split with sortformer -> deepfilternet -> 3 nodes (whisper, parakeet, canary) -> save audio


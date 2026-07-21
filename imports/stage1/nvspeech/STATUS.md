# NVSpeech status

Blocked on 2026-07-21. The official Hugging Face release
`amphion/Emilia-NV` is gated, and the token configured in `.env` receives
`GatedRepo: ... you are not in the authorized list` from the file resolver.
The repository tree and metadata are visible, but none of the audio shards can
be downloaded until the Hugging Face account accepts the dataset terms or is
approved.

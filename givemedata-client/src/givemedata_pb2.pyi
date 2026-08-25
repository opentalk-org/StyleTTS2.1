from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Split(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAINING: _ClassVar[Split]
    VALIDATION: _ClassVar[Split]
TRAINING: Split
VALIDATION: Split

class InitRequest(_message.Message):
    __slots__ = ("dataset_id", "validation_samples", "validation", "seed", "max_seconds", "num_workers", "device", "symbols", "max_text_tokens", "plbert_languages", "plbert_modality_id")
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    MAX_SECONDS_FIELD_NUMBER: _ClassVar[int]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    SYMBOLS_FIELD_NUMBER: _ClassVar[int]
    MAX_TEXT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PLBERT_LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    PLBERT_MODALITY_ID_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    validation_samples: int
    validation: bool
    seed: int
    max_seconds: float
    num_workers: int
    device: str
    symbols: _containers.RepeatedScalarFieldContainer[str]
    max_text_tokens: int
    plbert_languages: _containers.RepeatedScalarFieldContainer[str]
    plbert_modality_id: int
    def __init__(self, dataset_id: _Optional[str] = ..., validation_samples: _Optional[int] = ..., validation: bool = ..., seed: _Optional[int] = ..., max_seconds: _Optional[float] = ..., num_workers: _Optional[int] = ..., device: _Optional[str] = ..., symbols: _Optional[_Iterable[str]] = ..., max_text_tokens: _Optional[int] = ..., plbert_languages: _Optional[_Iterable[str]] = ..., plbert_modality_id: _Optional[int] = ...) -> None: ...

class InitResponse(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class DataRequest(_message.Message):
    __slots__ = ("session_id", "split")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SPLIT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    split: Split
    def __init__(self, session_id: _Optional[str] = ..., split: _Optional[_Union[Split, str]] = ...) -> None: ...

class Sample(_message.Message):
    __slots__ = ("wave", "duration", "speaker_id", "language_id", "text")
    WAVE_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_ID_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    wave: bytes
    duration: float
    speaker_id: int
    language_id: int
    text: bytes
    def __init__(self, wave: _Optional[bytes] = ..., duration: _Optional[float] = ..., speaker_id: _Optional[int] = ..., language_id: _Optional[int] = ..., text: _Optional[bytes] = ...) -> None: ...

class DataResponse(_message.Message):
    __slots__ = ("batch",)
    BATCH_FIELD_NUMBER: _ClassVar[int]
    batch: _containers.RepeatedCompositeFieldContainer[Sample]
    def __init__(self, batch: _Optional[_Iterable[_Union[Sample, _Mapping]]] = ...) -> None: ...

class EndRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class EndResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

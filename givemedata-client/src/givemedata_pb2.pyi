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
    __slots__ = ()
    def __init__(self) -> None: ...

class InitResponse(_message.Message):
    __slots__ = ("session_id", "train_config")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRAIN_CONFIG_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    train_config: str
    def __init__(self, session_id: _Optional[str] = ..., train_config: _Optional[str] = ...) -> None: ...

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

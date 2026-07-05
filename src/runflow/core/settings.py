from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")


def settings_defaults(settings_cls: type[BaseModel]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name, field in settings_cls.model_fields.items():
        if field.is_required():
            continue
        if field.default_factory is not None:
            defaults[name] = field.default_factory()
            continue
        defaults[name] = field.default

    return settings_cls.model_construct(**defaults).model_dump(mode="json")

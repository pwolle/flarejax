import functools
import json
from typing import Any, Generic, Self, TypeVar

from ._frozen import FrozenMappingHashable, FrozenSequence
from ._typecheck import typecheck

__all__ = [
    "is_valid_conifg_type",
    "ConfigMapping",
    "ConfigSequence",
    "coerce_config_type",
    "ConfigEncoder",
    "ConfigDecoder",
]


@typecheck
def is_valid_conifg_type(value) -> bool:
    if value is None:
        return True

    if isinstance(value, (bool, int, float, str)):
        return True

    if isinstance(value, (ConfigMapping, ConfigSequence)):
        return True

    return False


H = TypeVar("H")


@typecheck
class ConfigMapping(FrozenMappingHashable[str, H], Generic[H]):
    def __post_init__(self: Self) -> None:
        _data = dict(self._data)

        for k, v in _data.items():
            _data[k] = coerce_config_type(v)  # type: ignore

        object.__setattr__(self, "_data", _data)
        super().__post_init__()

        for k, v in self.items():
            if not isinstance(k, str):
                error = f"All keys must be strings, got {k}"
                raise TypeError(error)

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({dict(self._data)})"


@typecheck
class ConfigSequence(FrozenSequence[H], Generic[H]):
    def __post_init__(self) -> None:
        _data = list(self._data)

        for i, v in enumerate(_data):
            _data[i] = coerce_config_type(v)  # type: ignore

        object.__setattr__(self, "_data", _data)
        super().__post_init__()

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({list(self._data)})"

    # since hashing would be O(n) we cache the result
    @functools.cached_property
    def _hash(self: Self) -> int:
        return hash(self._data)

    def __hash__(self):
        return self._hash


@typecheck
def _ismapping(obj: Any) -> bool:
    try:
        _ = (lambda **kwargs: kwargs)(**obj)  # type: ignore
        return True
    except TypeError:
        return False


@typecheck
def _issequence(obj: Any) -> bool:
    try:
        _ = (lambda *args: args)(*obj)  # type: ignore
        return True
    except TypeError:
        return False


@typecheck
def coerce_config_type(
    data: Any,
) -> None | bool | int | float | str | ConfigMapping | ConfigSequence:
    if data is None:
        return data

    if isinstance(data, (bool, int, float, str)):
        return data

    if isinstance(data, (ConfigMapping, ConfigSequence)):
        return data

    if _ismapping(data):
        data_dict = {}

        for key, value in data.items():
            data_dict[key] = coerce_config_type(value)

        return ConfigMapping(data_dict)

    if _issequence(data):
        data_list = []

        for value in data:
            data_list.append(coerce_config_type(value))

        return ConfigSequence(data_list)

    error = f"Invalid value type: {data}"
    raise TypeError(error)


@typecheck
class ConfigEncoder(json.JSONEncoder):
    def default(self: Self, obj) -> dict[str, Any] | list[Any] | Any:
        if isinstance(obj, ConfigMapping):
            return dict(obj)

        if isinstance(obj, ConfigSequence):
            return list(obj)

        return super().default(obj)


@typecheck
class ConfigDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self: Self, obj) -> ConfigMapping | ConfigSequence | Any:
        if isinstance(obj, dict):
            return ConfigMapping(obj)

        if isinstance(obj, list):
            return ConfigSequence(obj)

        return obj

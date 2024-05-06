import functools
import json
from typing import Self, Any, TypeVar, Generic

from ._frozen import FrozenMapping, FrozenSequence

__all__ = [
    "is_valid_conifg_type",
    "ConfigMapping",
    "ConfigSequence",
    "to_immutable",
    "ConfigEncoder",
    "ConfigDecoder",
]


def is_valid_conifg_type(value):
    if value is None:
        return True

    if isinstance(value, (bool, int, float, str)):
        return True

    if isinstance(value, (ConfigMapping, ConfigSequence)):
        return True

    return False


C = TypeVar(
    "C",
    bound="None | bool | int | float | str | ConfigMapping | ConfigSequence",
)


class ConfigMapping(FrozenMapping[str, C], Generic[C]):
    def __post_init__(self: Self) -> None:
        _data = dict(self._data)

        for k, v in _data.items():
            _data[k] = to_immutable(v)  # type: ignore

        object.__setattr__(self, "_data", _data)
        super().__post_init__()

        for k, v in self.items():
            if not isinstance(k, str):
                error = f"All keys must be strings, got {k}"
                raise TypeError(error)

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({dict(self._data)})"

    # since hashing would be O(n) we cache the result
    @functools.cached_property
    def _hash(self: Self) -> int:
        unfolded = []

        for key in sorted(self, key=hash):
            unfolded.append((key, self[key]))

        return hash(tuple(unfolded))

    def __hash__(self: Self) -> int:
        return self._hash


class ConfigSequence(FrozenSequence):
    def __post_init__(self) -> None:
        _data = list(self._data)

        for i, v in enumerate(_data):
            _data[i] = to_immutable(v)

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


def _ismapping(obj: Any):
    try:
        _ = (lambda **kwargs: kwargs)(**obj)  # type: ignore
        return True
    except TypeError:
        return False


def _issequence(obj: Any):
    try:
        _ = (lambda *args: args)(*obj)  # type: ignore
        return True
    except TypeError:
        return False


def to_immutable(
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
            data_dict[key] = to_immutable(value)

        return ConfigMapping(data_dict)

    if _issequence(data):
        data_list = []

        for value in data:
            data_list.append(to_immutable(value))

        return ConfigSequence(data_list)

    error = f"Invalid value type: {data}"
    raise TypeError(error)


class ConfigEncoder(json.JSONEncoder):
    def default(self, obj) -> dict[str, Any] | list[Any] | Any:
        if isinstance(obj, ConfigMapping):
            return dict(obj)

        if isinstance(obj, ConfigSequence):
            return list(obj)

        return super().default(obj)


class ConfigDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj) -> ConfigMapping | ConfigSequence | Any:
        if isinstance(obj, dict):
            return ConfigMapping(obj)

        if isinstance(obj, list):
            return ConfigSequence(obj)

        return obj

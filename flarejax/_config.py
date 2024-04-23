import functools
import json
from typing import Hashable, Self, TypeVar

from ._frozen import FrozenMappingHashable, FrozenSequence

__all__ = [
    "valid_conifg_types",
    "ConfigMapping",
    "ConfigSequence",
    "to_immutable",
    "ConfigEncoder",
    "ConfigDecoder",
]


def valid_conifg_types(value):
    if value is None:
        return True

    if isinstance(value, (bool, int, float, str)):
        return True

    if isinstance(value, (ConfigMapping, ConfigSequence)):
        return True

    return False


H = TypeVar("H", bound=Hashable)


class ConfigMapping(FrozenMappingHashable[str, H]):
    def __post_init__(self: Self) -> None:
        super().__post_init__()

        if not all(isinstance(k, str) for k in self):
            raise TypeError("All keys must be strings")

        if not all(isinstance(v, Hashable) for v in self.values()):
            raise TypeError("All values must be hashable")

        if not all(valid_conifg_types(v) for v in self.values()):
            raise TypeError("Invalid value type")

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


class ConfigSequence(FrozenSequence[Hashable]):
    def __post_init__(self):
        super().__post_init__()

        if not all(isinstance(v, Hashable) for v in self):
            raise TypeError("All values must be hashable")

        if not all(valid_conifg_types(v) for v in self):
            raise TypeError("Invalid value type")

    def __repr__(self: Self) -> str:
        return f"{self.__class__.__name__}({list(self._data)})"

    # since hashing would be O(n) we cache the result
    @functools.cached_property
    def _hash(self: Self) -> int:
        return hash(self._data)

    def __hash__(self):
        return self._hash


def to_immutable(
    data,
) -> None | bool | int | float | str | ConfigMapping | ConfigSequence:
    if data is None:
        return data

    if isinstance(data, (bool, int, float, str)):
        return data

    if isinstance(data, (ConfigMapping, ConfigSequence)):
        return data

    if isinstance(data, dict):
        data_dict = {}

        for key, value in data.items():
            data_dict[key] = to_immutable(value)

        return ConfigMapping(data_dict)

    if isinstance(data, (list, tuple)):
        data_list = []

        for value in data:
            data_list.append(to_immutable(value))

        return ConfigSequence(data_list)

    error = f"Invalid value type: {data}"
    raise TypeError(error)


class ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ConfigMapping):
            return dict(obj)

        if isinstance(obj, ConfigSequence):
            return list(obj)

        return super().default(obj)


class ConfigDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj, dict):
            return ConfigMapping(obj)

        if isinstance(obj, list):
            return ConfigSequence(obj)

        return obj

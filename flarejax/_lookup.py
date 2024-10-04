import dataclasses
from typing import Hashable

__all__ = [
    "ItemLookup",
    "AttrLookup",
    "PathLookup",
    # "Lookup",
    # "LookupType",
]


@dataclasses.dataclass(frozen=True)
class ItemLookup:
    """
    Describes how to lookup an item in a dictionary or list or tuple.
    """

    key: Hashable | int

    def __repr__(self) -> str:
        return f"[{self.key}]"


@dataclasses.dataclass(frozen=True)
class AttrLookup:
    """
    Describes how to lookup an attribute in a class.
    """

    key: str

    def __repr__(self) -> str:
        return f".{self.key}"


@dataclasses.dataclass(frozen=True)
class PathLookup:
    """
    Describes how to lookup a value in a nested structure.
    """

    path: tuple[ItemLookup | AttrLookup, ...]

    def __repr__(self) -> str:
        return "obj" + "".join(map(str, self.path))

    def __add__(self, other):
        if isinstance(other, PathLookup):
            return PathLookup(self.path + other.path)

        # assert isinstance(other, (ItemLookup, AttrLookup))
        if not isinstance(other, (ItemLookup, AttrLookup)):
            raise TypeError(f"Cannot add {type(other)} to Lookup.")

        return PathLookup(self.path + (other,))

    def __lt__(self, other):
        return hash(self) < hash(other)


# @dataclasses.dataclass(frozen=True)
# class LookupType:
#     # """
#     # Create PathLookup objects by storing the lookup of this object.
#     # """

#     _path_lookup: tuple[ItemLookup | AttrLookup, ...] = ()

#     def __call__(self) -> PathLookup:
#         return PathLookup(self._path_lookup)

#     def __getattribute__(self, name: str):
#         if name == "_path_lookup":
#             return super().__getattribute__(name)

#         return LookupType(self._path_lookup + (AttrLookup(name),))

#     def __getitem__(self, key: Hashable):
#         return LookupType(self._path_lookup + (ItemLookup(key),))


# Lookup = LookupType(())

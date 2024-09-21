from typing import Any, Hashable, overload, Sequence, Never
from pprint import pprint

import dataclasses


@dataclasses.dataclass(frozen=True)
class ItemLookup:
    key: Hashable | int

    def __repr__(self) -> str:
        return f"[{self.key}]"


@dataclasses.dataclass(frozen=True)
class AttrLookup:
    key: str

    def __repr__(self) -> str:
        return f".{self.key}"


@dataclasses.dataclass(frozen=True)
class PathLookup:
    key: Sequence[ItemLookup | AttrLookup] = ()

    def __repr__(self) -> str:
        return "obj" + "".join(map(repr, self.key))


class Module:
    pass


def get_lookups(
    obj: Module | list | tuple | dict | Any,
    /,
) -> list[ItemLookup | AttrLookup]:
    if not isinstance(obj, (Module, list, tuple, dict)):
        return []

    if isinstance(obj, (list, tuple)):
        return [ItemLookup(i) for i in range(len(obj))]

    if isinstance(obj, dict):
        return [ItemLookup(k) for k in sorted(obj.keys(), key=hash)]

    assert isinstance(obj, Module)

    keys = []

    if hasattr(obj, "__dict__"):
        for key in sorted(obj.__dict__.keys()):
            keys.append(AttrLookup(key))

    if hasattr(obj, "__slots__"):
        for key in sorted(obj.__slots__):  # type: ignore
            keys.append(AttrLookup(key))

    return keys


def get_with_lookup(
    obj: Module | list | tuple | dict,
    /,
    lookup: ItemLookup | AttrLookup,
) -> Any:
    if isinstance(lookup, AttrLookup):
        assert isinstance(lookup.key, str)
        return getattr(obj, lookup.key)

    assert isinstance(lookup, ItemLookup)
    return obj[lookup.key]  # type: ignore


TYPEKEY = None


def deref(
    obj: Any,
    path: PathLookup = PathLookup(),
    refs: dict[int, PathLookup] | None = None,
) -> Any:
    refs = refs or {}
    lookups = get_lookups(obj)

    if id(obj) in refs:
        return refs[id(obj)]

    refs[id(obj)] = path

    if not lookups:
        return obj

    children: dict[ItemLookup | AttrLookup | None, Any]
    children = {TYPEKEY: type(obj)}

    for lookup in lookups:
        path_new = PathLookup((*path.key, lookup))

        child = get_with_lookup(obj, lookup)
        child = deref(child, path_new, refs)

        children[lookup] = child

    return children


def main():
    obj = [1, 2, [3, 4], 5, [6, 7]]
    obj.append(obj)
    obj[2].append(obj[4])
    obj[4].append(obj[2])

    derefed = deref(obj)
    # flatten = flatten_dict(derefed)
    # unflat = unflatten_dict(flatten)

    print(obj)
    print()

    pprint(derefed)
    print()

    # pprint(flatten)
    # print()

    # pprint(unflat)
    # print()


if __name__ == "__main__":
    main()

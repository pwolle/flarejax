import dataclasses
from pprint import pprint
from typing import Any, Hashable, Sequence


@dataclasses.dataclass(frozen=True)
class ItemLookup:
    key: Hashable | int
    src: type | None

    def __repr__(self) -> str:
        return f"[{self.key}]"


@dataclasses.dataclass(frozen=True)
class PathLookup:
    key: Sequence[ItemLookup] = ()

    def __repr__(self) -> str:
        return "obj" + "".join(map(repr, self.key))


def get_lookups(obj: Any, /) -> list[ItemLookup]:
    src = type(obj)

    if isinstance(obj, list):
        return [ItemLookup(i, src) for i in range(len(obj))]

    if isinstance(obj, dict):
        return [ItemLookup(k, src) for k in sorted(obj.keys(), key=hash)]

    return []


def get_with_lookup(obj: Any, lookup: ItemLookup) -> Any:
    return obj[lookup.key]


class Node(dict):
    def __repr__(self) -> str:
        return f"Node({super().__repr__()})"


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

    children = Node()

    for lookup in lookups:
        path_new = PathLookup((*path.key, lookup))

        child = get_with_lookup(obj, lookup)
        child = deref(child, path_new, refs)

        children[lookup] = child

    return children


def flatten_dict(
    nested_dict,
    path: PathLookup = PathLookup(),
):
    items = []

    for key, value in nested_dict.items():
        new_path = PathLookup((*path.key, key))

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_path).items())
            continue

        items.append((new_path, value))

    return dict(items)


def unflatten_dict(flat):
    nested = Node()

    for path, value in flat.items():
        current = nested

        for key in path.key[:-1]:
            current = current.setdefault(key, Node())

        current[path.key[-1]] = value

    return nested


def main():
    obj = [1, 2, [3, 4], 5, [6, 7]]
    obj.append(obj)
    obj[2].append(obj[4])
    obj[4].append(obj[2])

    derefed = deref(obj)
    flatten = flatten_dict(derefed)
    unflat = unflatten_dict(flatten)

    pprint(derefed)
    print()

    pprint(flatten)
    print()

    pprint(unflat)
    print()


if __name__ == "__main__":
    main()

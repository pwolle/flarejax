import dataclasses

from typing import (
    Any,
    Literal,
    Self,
    Protocol,
    runtime_checkable,
    TypeVar,
    Generic,
)

from ._typecheck import typecheck

EMPTY = object()


@runtime_checkable
class AtWrappableAttr(Protocol):
    def __frozen_set_attr__(self: Self, k: str, v: Any, /) -> Any: ...

    def __frozen_del_item__(self: Self, k: Any, /) -> Any: ...


@runtime_checkable
class AtWrappableItem(Protocol):
    def __frozen_set_item__(self: Self, k: Any, v: Any, /) -> Any: ...

    def __frozen_del_item__(self: Self, k: Any, /) -> Any: ...


T = TypeVar("T", bound=AtWrappableAttr | AtWrappableItem)


@typecheck
@dataclasses.dataclass(frozen=True)
class AtWrapper(Generic[T]):
    _at_module: T
    _at_adress: tuple[Any | str, ...] = ()
    _at_lookup: tuple[Literal["attr", "item"], ...] = ()

    def __post_init__(self: Self) -> None:
        if len(self._at_adress) != len(self._at_lookup):
            error = "Adress and lookup must have the same length."
            raise ValueError(error)

        if not all(lookup in ("attr", "item") for lookup in self._at_lookup):
            error = "Lookup must be either 'attr' or 'item'."
            raise ValueError(error)

    def __getattr__(self: Self, k: str) -> Self:
        return dataclasses.replace(
            self,
            _at_adress=self._at_adress + (k,),
            _at_lookup=self._at_lookup + ("attr",),
        )

    def __getitem__(self: Self, k: Any) -> Self:
        return dataclasses.replace(
            self,
            _at_adress=self._at_adress + (k,),
            _at_lookup=self._at_lookup + ("item",),
        )

    def set(self: Self, updated: Any) -> T:
        values = [self._at_module]

        for adress, lookup in zip(
            self._at_adress[:-1],
            self._at_lookup[:-1],
            strict=True,
        ):
            if lookup == "attr":
                value = getattr(values[-1], adress)
                values.append(value)
                continue

            if lookup == "item":
                value = values[-1][adress]  # type: ignore
                values.append(value)
                continue

            error = f"Invalid lookup '{lookup}'."
            raise ValueError(error)

        for adress, value, lookup in zip(
            reversed(self._at_adress),
            reversed(values),
            reversed(self._at_lookup),
            strict=True,
        ):
            if lookup == "attr":
                assert isinstance(value, AtWrappableAttr)
                updated = value.__frozen_set_attr__(adress, updated)
                continue

            if lookup == "item":
                if updated is EMPTY:
                    updated = value.__frozen_del_item__(adress)
                    continue

                assert isinstance(value, AtWrappableItem)
                updated = value.__frozen_set_item__(adress, updated)
                continue

            error = f"Invalid lookup '{lookup}'."
            raise ValueError(error)

        return updated

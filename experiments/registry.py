"""Small explicit registries used by every replaceable experiment strategy."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """Name-to-factory mapping with duplicate protection."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., T] | type[T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T] | type[T]], Callable[..., T] | type[T]]:
        key = name.strip().lower()
        if not key:
            raise ValueError(f"{self.kind} registry names cannot be empty")

        def decorator(factory: Callable[..., T] | type[T]) -> Callable[..., T] | type[T]:
            if key in self._items:
                raise KeyError(f"duplicate {self.kind} registration: {key}")
            self._items[key] = factory
            return factory

        return decorator

    def create(self, name: str, **kwargs: Any) -> T:
        key = name.strip().lower()
        try:
            factory = self._items[key]
        except KeyError as error:
            options = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"unknown {self.kind} '{name}'; available: {options}") from error
        return factory(**kwargs)

    def get(self, name: str) -> Callable[..., T] | type[T]:
        return self._items[name.strip().lower()]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


REGISTRIES: dict[str, Registry[Any]] = {
    name: Registry(name)
    for name in (
        "grouping",
        "parameter_block",
        "residual",
        "jacobian",
        "curvature",
        "solver",
        "aggregation",
        "acceptance",
    )
}

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass(frozen=True)
class Reference:
    authors: str
    title: str
    venue: str
    year: int
    doi_or_url: str | None = None


@dataclass(frozen=True)
class AlgorithmMetadata:
    name: str
    references: List[Reference]
    notes: str


_ALGORITHM_REGISTRY: dict[str, AlgorithmMetadata] = {}


def algorithm_metadata(name: str, references: List[Reference], notes: str = "") -> Callable:
    def decorator(func: Callable) -> Callable:
        meta = AlgorithmMetadata(name=name, references=references, notes=notes)
        setattr(func, "algorithm_metadata", meta)
        _ALGORITHM_REGISTRY[func.__name__] = meta
        return func
    return decorator


def get_algorithm_registry() -> dict[str, AlgorithmMetadata]:
    return dict(_ALGORITHM_REGISTRY)

from __future__ import annotations
from dataclasses import dataclass

from numpy.typing import DTypeLike


@dataclass
class Feature:
    shape: tuple[int, ...]
    dtype: DTypeLike

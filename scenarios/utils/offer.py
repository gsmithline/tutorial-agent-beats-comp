from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Offer:
	player: int
	offer: List[int]



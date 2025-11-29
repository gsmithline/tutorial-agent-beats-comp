from __future__ import annotations
from typing import List, Tuple
import random

from .base import BaseNegotiator


class SoftNegotiator(BaseNegotiator):
	def propose(self, quantities: Tuple[int, int, int], role: str, v_self: List[int], v_opp: List[int]) -> Tuple[List[int], List[int]]:
		# Starting agent proposes a random split across items
		a_self = [random.randint(0, q) for q in quantities]
		a_opp = [q - a_self[i] for i, q in enumerate(quantities)]
		return a_self, a_opp

	def accepts(self, offer_value: int, batna_value: int, counter_value: int) -> bool:
		# Always accept any offer on the table
		return True



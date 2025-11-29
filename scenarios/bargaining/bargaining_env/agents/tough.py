from __future__ import annotations
from typing import List, Tuple

from .base import BaseNegotiator


class ToughNegotiator(BaseNegotiator):
	def propose(self, quantities: Tuple[int, int, int], role: str, v_self: List[int], v_opp: List[int]) -> Tuple[List[int], List[int]]:
		# Offer exactly one unit of the least-valued item (by v_self), keep the rest
		idx = 0
		min_val = None
		for i, q in enumerate(quantities):
			if q <= 0:
				continue
			val = v_self[i] if i < len(v_self) else 0
			if min_val is None or val < min_val:
				min_val = val
				idx = i
		a_opp = [0, 0, 0]
		if sum(quantities) > 0:
			# give one of least-valued available item
			a_opp[idx] = 1
		a_self = [quantities[i] - a_opp[i] for i in range(len(quantities))]
		return a_self, a_opp

	def accepts(self, offer_value: int, batna_value: int, counter_value: int) -> bool:
		# Never accept; always counter with the least-valued item offer
		return False



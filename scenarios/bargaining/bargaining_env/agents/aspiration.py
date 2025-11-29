from __future__ import annotations
from typing import List, Tuple

from .base import BaseNegotiator


class AspirationNegotiator(BaseNegotiator):
	"""
	Lightweight aspiration-based negotiator:
	- As proposer: keeps enough items to reach ~85% of total self value, gives the rest
	- As responder: accepts if offer meets BATNA or is within 5% of a plausible counter
	"""

	def __init__(self, keep_fraction: float = 0.85, accept_slack: float = 0.05):
		self.keep_fraction = float(max(0.0, min(1.0, keep_fraction)))
		self.accept_slack = float(max(0.0, accept_slack))

	def propose(self, quantities: Tuple[int, int, int], role: str, v_self: List[int], v_opp: List[int]) -> Tuple[List[int], List[int]]:
		total_value = v_self[0] * quantities[0] + v_self[1] * quantities[1] + v_self[2] * quantities[2]
		target_value = self.keep_fraction * total_value

		# Greedy keep by value density
		idxs = sorted(range(3), key=lambda i: (-v_self[i], i))
		keep = [0, 0, 0]
		acc = 0.0
		for i in idxs:
			if quantities[i] <= 0 or v_self[i] <= 0:
				continue
			if acc >= target_value:
				break
			# keep as many as needed up to available
			need = int(max(0, (target_value - acc) // max(1, v_self[i])))
			need = min(need, quantities[i])
			if need == 0 and acc < target_value:
				need = min(1, quantities[i])
			keep[i] = need
			acc += need * v_self[i]

		a_self = keep
		a_opp = [quantities[i] - a_self[i] for i in range(3)]
		return a_self, a_opp

	def accepts(self, offer_value: int, batna_value: int, counter_value: int) -> bool:
		threshold = max(batna_value, int(counter_value * (1.0 - self.accept_slack)))
		return offer_value >= threshold



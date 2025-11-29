from __future__ import annotations
from typing import List, Tuple


class BaseNegotiator:
	"""
	Base interface for simplified bargaining negotiators used by the lightweight simulator.
	"""

	def propose(self, quantities: Tuple[int, int, int], role: str, v_self: List[int], v_opp: List[int]) -> Tuple[List[int], List[int]]:
		"""
		Return a proposed allocation (a_self, a_opp) such that element-wise sums equal quantities.
		role is 'row' or 'col' indicating proposer.
		"""
		raise NotImplementedError

	def accepts(self, offer_value: int, batna_value: int, counter_value: int) -> bool:
		"""
		Decide whether to accept an offer given:
		- offer_value: realized value from current offer for self
		- batna_value: BATNA value for self
		- counter_value: realized value the agent expects from its own counterproposal this round
		"""
		raise NotImplementedError



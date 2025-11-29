from __future__ import annotations
from typing import Dict, Tuple, Any


def build_negotiation_params(
	*,
	discount: float,
	max_rounds: int,
	num_items: int = 3,
	item_quantities: Tuple[int, int, int] = (7, 4, 1),
	min_value: int = 1,
	max_value: int = 100,
	max_quantity: int = 10,
) -> Dict[str, Any]:
	return {
		"enable_proposals": True,
		"enable_utterances": False,
		"num_items": num_items,
		"discount": discount,
		"min_value": min_value,
		"max_value": max_value,
		"max_rounds": max_rounds,
		"max_quantity": max_quantity,
		"item_quantities": f"{item_quantities[0]},{item_quantities[1]},{item_quantities[2]}",
	}


def try_load_pyspiel_game(params: Dict[str, Any]):
	try:
		import pyspiel  # type: ignore
	except Exception:
		return None
	try:
		return pyspiel.load_game("negotiation", params)
	except Exception:
		return None



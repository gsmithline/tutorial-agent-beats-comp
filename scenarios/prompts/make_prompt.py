from __future__ import annotations

import numpy as np  # noqa: F401

from ..utils.offer import Offer
from .prompt_texts.circle0 import make_prompt_circle_0
from .prompt_texts.circle1 import make_prompt_circle_1
from .prompt_texts.circle2 import make_prompt_circle_2
from .prompt_texts.circle3 import make_prompt_circle_3
from .prompt_texts.circle4 import make_prompt_circle_4
from .prompt_texts.circle5 import make_prompt_circle_5
from .prompt_texts.circle6 import make_prompt_circle_6


def make_prompt(
	T: int,
	quantities: list[int],
	V: int,
	values: list[float],
	W1: int,
	W2: int,
	w: int,
	R: int,
	g: float,
	r: int,
	history: dict,
	current_offer: Offer | None = None,
	player_num: int = 0,
	p1_outside_offer: list[int] | None = None,
	p2_outside_offer: list[int] | None = None,
	circle: int = 0,
	example_offer_less_than_outside_offer_self: list[int] | None = None,
) -> str:
	my_player_num = player_num + 1
	other_player_num = 2 if my_player_num == 1 else 1

	history_str = ""
	for round_num in range(len(history.get(0, [])) + len(history.get(1, []))):
		player = round_num % 2
		round_idx = round_num // 2
		offer = None
		if round_idx < len(history.get(player, [])):
			offer = history[player][round_idx]
		if isinstance(offer, Offer):
			history_str += f"\nRound {round_idx + 1}: Player {player + 1} offered {offer.offer}"
		elif offer is True:
			history_str += f"\nRound {round_idx + 1}: Player {player + 1} ACCEPTED"
		elif offer is False:
			history_str += f"\nRound {round_idx + 1}: Player {player + 1} WALKED away"

	current_offer_str = f"\nCurrent offer on the table (the amount of each item being offered to you): {current_offer.offer if isinstance(current_offer, Offer) else 'None'}"

	if r == 1 and my_player_num == 1:
		action_prompt = f"""
		What is your action? As the first player, your available actions are:
		- WALK to walk away
		- A list of numbers [n1, n2, ...] representing your initial offer (what you give to Player 2)"""
	elif current_offer is None:
		action_prompt = f"""
		What is your action? You can:
		- WALK to walk away
		- A list of numbers [n1, n2, ...] representing your offer (what you give to Player {other_player_num})"""
	else:
		action_prompt = f"""
	What is your action? You can:
	- ACCEPT to accept the current offer
	- WALK to walk away
	- A list of numbers [n1, n2, ...] representing your counteroffer (what you give to Player {other_player_num})"""

	if circle == 0:
		prompt = make_prompt_circle_0(T, quantities, V, values, W1, W2, w, R, g, r, history, current_offer, player_num, p1_outside_offer, p2_outside_offer, circle, other_player_num, my_player_num)
	elif circle == 1:
		prompt = make_prompt_circle_1(T, quantities, V, values, W1, W2, w, R, g, r, history, current_offer, player_num, p1_outside_offer, p2_outside_offer, circle, other_player_num, my_player_num)
	elif circle == 2:
		prompt = make_prompt_circle_2(T, quantities, V, values, W1, W2, w, R, g, r, history, current_offer, player_num, p1_outside_offer, p2_outside_offer, circle, other_player_num, my_player_num)
	elif circle == 3:
		prompt = make_prompt_circle_3(T, quantities, V, values, W1, W2, w, R, g, r, history, current_offer, player_num, p1_outside_offer, p2_outside_offer, circle, other_player_num, my_player_num)
	elif circle == 4:
		prompt = make_prompt_circle_4(T, quantities, V, values, W1, W2, w, R, g, r, history, current_offer, player_num, p1_outside_offer, p2_outside_offer, circle, other_player_num, my_player_num)
	elif circle == 5:
		prompt = make_prompt_circle_5(T, quantities, V, values, W1, W2, w, R, g, r, history, current_offer, player_num, p1_outside_offer, p2_outside_offer, circle, other_player_num, my_player_num, example_offer_less_than_outside_offer_self)
	elif circle == 6:
		prompt = make_prompt_circle_6(T, quantities, V, values, W1, W2, w, R, g, r, history, current_offer, player_num, p1_outside_offer, p2_outside_offer, circle, other_player_num, my_player_num, example_offer_less_than_outside_offer_self)
	else:
		raise ValueError(f"Circle {circle} not supported")

	return f"{prompt}\n Negotiation history:{history_str}\n{current_offer_str}\n{action_prompt}"



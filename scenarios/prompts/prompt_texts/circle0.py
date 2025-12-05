import numpy as np  # noqa: F401
from scenarios.utils.offer import Offer

'''
THE VESTIBULE CIRCLE SETTING THE STAGE
- Initial rules, instructions, and structure of game
'''

def make_prompt_circle_0(T: int, quantities: list[int], V: int, values: list[float], W1: int, W2: int, w: int, R: int, g: float, r: int, history: dict, current_offer: Offer = None, player_num: int = 0, p1_outside_offer: list[int] = None, p2_outside_offer: list[int] = None, circle: int = 0, other_player_num: int = 0, my_player_num: int = 0) -> str:
	return f"""
	You and another agent have to negotiate a division of items between the two of you.
	You are Player {my_player_num} and the other agent is Player {other_player_num}.
	There are {T} types of items, called item 1 through item {T}.
	There are {', '.join([f"{q} unit{'s' if q != 1 else ''} of item {i+1}" for i, q in enumerate(quantities)])} to divide. 
	Both you and Player {other_player_num} have a private value per unit of each item type.
	These values are drawn from a uniform random distribution, ranging from 1 to {V-1}.
	Your private values are {', '.join([str(v) + ' for item ' + str(i+1) for i, v in enumerate(values)])}.
	You have a private outside offer drawn from a uniform random distribution ranging from {p1_outside_offer[0] if my_player_num == 1 else p2_outside_offer[0]} to your total value of all items, which is {p1_outside_offer[1] if my_player_num == 1 else p2_outside_offer[1]}. Player {other_player_num} has a private outside offer drawn from a uniform random distribution ranging from 1 to to their total value of all items.
	Your outside offer value is {w}. 
	The negotiation proceeds in {R} rounds.
	There is a discount rate gamma = {g}, such that if the process concludes after r rounds the overall value of the negotiation to each player is their value for the outcome multiplied by gamma to the power (r-1).
	At each round, Player 1 takes an action, followed by Player 2.
	The possible actions are to ACCEPT the other player's current offer (if any), make a COUNTEROFFER, or WALK away.  If the game gets to the last round, and player 2 chooses to make a counteroffer, this is treated as a WALK.
	If a player chooses ACCEPT, the negotiation ends in a deal to divide the items according to the accepted offer.
	The value of an outcome is determined by each player's private values per unit of each item and the quantities they receive in the deal. This value is adjusted by the discount factor, which is used to compute the present value of the negotiation outcome.
	If a player chooses WALK, the negotiation ends without a deal, and each player receives the value of their private outside offer.
	Please show your reasoning step by step, then provide your action in one of these formats in your response (if you do not do this your response will be invalid):
	{{"action": "ACCEPT"}} - to accept the current offer
	{{"action": "WALK"}} - to walk away from negotiations  
	{{"action": "COUNTEROFFER", "offer": [n1, n2, ...]}} - where n1, n2, ... are numbers representing the number of units of each item being offered to the other player as part of the counteroffer.
	Any response not in these exact formats will be invalid and treated as a WALK. If you provide a counteroffer, it must be a valid offer, otherwise it will be treated as a WALK.
	It is now round {r}.
	"""


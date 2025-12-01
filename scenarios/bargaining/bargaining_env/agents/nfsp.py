from __future__ import annotations

import os
import traceback
from typing import List, Optional

import numpy as np


try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
except Exception as _e:
	torch = None  # type: ignore
	nn = None  # type: ignore
	F = None  # type: ignore


# Optional: OpenSpiel is only needed when actually running this agent
try:
	import pyspiel  # type: ignore
except Exception:
	pyspiel = None  # type: ignore


class PolicyModel(nn.Module):  # type: ignore
	def __init__(self, n_actions: int, hidden_size: int = 256):
		super().__init__()
		self.lin1 = nn.LazyLinear(hidden_size)
		self.lin2 = nn.Linear(hidden_size, hidden_size)
		# Match checkpoint naming "policy_head.*"
		self.policy_head = nn.Linear(hidden_size, n_actions)

	def forward(self, x, action_mask):  # type: ignore
		x = self.lin1(x.float())
		x = F.relu(x)
		out = self.lin2(x) + x
		out = F.relu(out)
		out = self.policy_head(out)
		out = torch.masked_fill(out, ~action_mask.bool(), -1e9)
		return out


class NFSPAgentWrapper:
	"""
	Average-policy NFSP agent for the OpenSpiel negotiation game.
	This wrapper is intended for inference-only usage with provided checkpoints.
	"""

	def __init__(
		self,
		game,  # pyspiel.Game
		player_id: int,
		*,
		checkpoint_path: Optional[str] = None,
		discount: float = 0.98,
		max_rounds: int = 3,
		debug: bool = False,
	) -> None:
		if torch is None:
			raise RuntimeError("PyTorch not available; NFSP requires torch to run.")
		if pyspiel is None:
			raise RuntimeError("OpenSpiel (pyspiel) not available; NFSP agent requires it.")

		self.player_id = int(player_id)
		self.discount = float(discount)
		self.max_rounds = int(max_rounds)
		self.debug = bool(debug)

		self.last_logits = None
		self.last_probs = None

		self._num_actions = int(game.num_distinct_actions())
		self._state_size = int(game.observation_tensor_size())
		self._num_items = int(game.get_parameters().get("num_items", 3))
		self._rng = np.random.RandomState(123)

		# Book-keeping decoded from observations (best-effort)
		self.items: Optional[List[int]] = None
		self.valuation_vector: Optional[List[int]] = None
		self.walk_away_value: Optional[float] = None
		self.current_counter_offer_give_format: Optional[List[int]] = None

		# Device + model
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if self.debug:
			print(f"[NFSP P{self.player_id}] device: {self.device}")

		self._policy_net = PolicyModel(self._num_actions, hidden_size=256).to(self.device)
		# Materialize LazyLinear BEFORE loading weights
		with torch.no_grad():
			dummy_x = torch.zeros(1, self._state_size, device=self.device)
			dummy_mask = torch.ones(1, self._num_actions, dtype=torch.bool, device=self.device)
			_ = self._policy_net(dummy_x, dummy_mask)

		# Load checkpoint if provided
		if checkpoint_path:
			if os.path.exists(checkpoint_path):
				try:
					checkpoint = torch.load(checkpoint_path, map_location=self.device)
					self._policy_net.load_state_dict(checkpoint)
					self._policy_net.eval()
					if self.debug:
						print(f"[NFSP P{self.player_id}] Loaded checkpoint {checkpoint_path}")
				except Exception as e:
					print(f"[NFSP ERROR] Failed to load checkpoint {checkpoint_path}: {e}")
					self._policy_net = None  # disable, will fallback to random
			else:
				raise FileNotFoundError(f"NFSP checkpoint not found: {checkpoint_path}")
		else:
			# Allow running without checkpoint (random policy over legal actions)
			if self.debug:
				print(f"[NFSP P{self.player_id}] No checkpoint provided; using random legal policy.")
			self._policy_net = None

	def _decode_action_to_proposal(self, action_id: int) -> Optional[np.ndarray]:
		# Decode base-(max_quantity+1) to get the kept quantities per item
		base = 10 + 1  # default OpenSpiel max_quantity is typically 10 for BGS configs we use
		max_id = base ** self._num_items - 1
		if action_id < 0 or action_id > max_id:
			return None
		proposal = np.zeros(self._num_items, dtype=int)
		tmp = action_id
		for i in range(self._num_items - 1, -1, -1):
			div = base ** i
			proposal[self._num_items - 1 - i] = tmp // div
			tmp %= div
		return proposal

	def _process_observation(self, observation: List[float]) -> None:
		# Heuristic decode matching typical OpenSpiel negotiation observation layout
		num_items = self._num_items
		is_response_state = observation[3] == 0.0 if len(observation) > 3 else False
		round_number = int(observation[6]) + 1 if len(observation) > 6 else 1
		discount_factor = observation[7] if len(observation) > 7 else self.discount
		item_quantities = observation[9 : 9 + num_items]
		self.items = list(map(int, item_quantities))
		pv_start = 9 + num_items
		pv_end = pv_start + num_items
		player_values = observation[pv_start:pv_end]
		self.valuation_vector = list(map(int, player_values))
		walk_away_value = observation[9 + 2 * num_items] if len(observation) > (9 + 2 * num_items) else 0.0
		self.walk_away_value = float(walk_away_value)
		total_utility = float(np.dot(player_values, item_quantities))
		if self.debug:
			print(f"[NFSP P{self.player_id}] round {round_number} items {item_quantities} values {player_values} W {walk_away_value} g={discount_factor} TU={total_utility}")

	def step(self, state, *, is_evaluation: bool = True) -> int:  # pyspiel.State
		cur = state.current_player()
		self.current_counter_offer_give_format = None
		self._process_observation(state.observation_tensor(self.player_id))
		if cur != self.player_id:
			return 0

		obs_raw = np.asarray(state.observation_tensor(cur), dtype=np.float32)
		obs = obs_raw[: self._state_size]

		try:
			legal = list(state.legal_actions(cur))
		except Exception:
			legal = []

		# Constrain to known action space
		legal = [a for a in legal if 0 <= int(a) < self._num_actions]
		if not legal:
			return 0

		# Random if no policy loaded
		if self._policy_net is None or torch is None:
			return int(self._rng.choice(legal))

		inp = torch.from_numpy(obs).unsqueeze(0).to(self.device)
		mask = torch.zeros(self._num_actions, dtype=torch.bool, device=self.device)
		for a in legal:
			mask[int(a)] = True
		with torch.no_grad():
			logits = self._policy_net(inp, mask.unsqueeze(0)).squeeze(0)
			probs = torch.softmax(logits, dim=-1)
			probs = probs.masked_fill(~mask, 0.0)
			mass = probs.sum()
			if mass.item() > 0:
				probs = probs / mass
				p_cpu = probs.detach().cpu().numpy()
				final_action = int(self._rng.choice(p_cpu.size, p=p_cpu))
			else:
				legal_idx_cpu = np.flatnonzero(mask.detach().cpu().numpy())
				final_action = int(self._rng.choice(legal_idx_cpu))

		# Attempt to decode current counter-offer (for external logging)
		try:
			proposal_keep = self._decode_action_to_proposal(final_action)
			if proposal_keep is not None and self.items is not None:
				give = (np.array(self.items) - proposal_keep).tolist()
				self.current_counter_offer_give_format = give
		except Exception:
			traceback.print_exc()

		return final_action




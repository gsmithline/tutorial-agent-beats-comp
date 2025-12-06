from __future__ import annotations

import importlib
import os
import pickle
import sys
import traceback
from typing import Dict, List, Optional

import numpy as np

try:
	import pyspiel  # type: ignore
except Exception:
	pyspiel = None  # type: ignore

# Attempt to import the original RNAD module; if missing, provide a fallback
# that still allows checkpoints to unpickle. Preferred path is the provided
# checkpoints repo: scenarios/bargaining/rl_agent_checkpoints/rnad/rnad.py
RNaDSolver = None  # type: ignore[var-annotated]
if RNaDSolver is None:
	try:
		_rnad_mod = importlib.import_module("scenarios.bargaining.rl_agent_checkpoints.rnad.rnad")
		sys.modules.setdefault("rnad", _rnad_mod)
		RNaDSolver = _rnad_mod.RNaDSolver  # type: ignore[attr-defined]
	except Exception:
		pass
if RNaDSolver is None:
	try:
		_rnad_mod = importlib.import_module("agents.rnad_working.rnad")
		sys.modules.setdefault("rnad", _rnad_mod)
		RNaDSolver = _rnad_mod.RNaDSolver  # type: ignore[attr-defined]
	except Exception:
		pass
if RNaDSolver is None:
	class RNaDSolver:  # type: ignore
		def __init__(self, *args, **kwargs):
			self._fallback_warned = False

		def __setstate__(self, state):
			self.__dict__.update(state)

		def action_probabilities(self, state) -> Dict[int, float]:
			try:
				legal = list(state.legal_actions())
			except Exception:
				legal = []
			if not legal:
				return {}
			if not self._fallback_warned:
				print("[RNAD fallback] rnad module missing; using uniform policy.")
				self._fallback_warned = True
			p = 1.0 / len(legal)
			return {int(a): p for a in legal}


class RNaDAgentWrapper:
	"""RNAD average‑policy agent using a pickled solver with action_probabilities(state)."""

	def __init__(
		self,
		game,  # pyspiel.Game
		player_id: int,
		*,
		checkpoint_path: str,
		debug: bool = True,
	) -> None:
		if pyspiel is None:
			raise RuntimeError("OpenSpiel (pyspiel) not available; RNAD agent requires it.")

		self.player_id = int(player_id)
		self.debug = bool(debug)

		self._num_actions = int(game.num_distinct_actions())
		self._state_size = int(game.observation_tensor_size())
		self._num_items = int(game.get_parameters().get("num_items", 3))
		self.max_quantity_for_encoding = 10

		# Public fields (parity with NFSP wrapper)
		self.action: str = "RNAD_ACTION"
		self.items: Optional[List[int]] = None
		self.valuation_vector: Optional[List[int]] = None
		self.walk_away_value: Optional[float] = None
		self.current_counter_offer_give_format: Optional[List[int]] = None
		self.prompt: Optional[str] = None
		self.last_probs: Optional[np.ndarray] = None
		self.player_num = self.player_id
		self._rng = np.random.RandomState(123)
		self.last_dist = None
		self.last_action = None
		self.last_prob = None

		if not os.path.exists(checkpoint_path):
			raise FileNotFoundError(f"RNAD checkpoint not found: {checkpoint_path}")
		if self.debug:
			print(f"[RNAD P{self.player_id}] loading: {checkpoint_path}")
		with open(checkpoint_path, "rb") as f:
			self._solver: RNaDSolver = pickle.load(f)  # type: ignore[assignment]
		if self.debug:
			cfg = getattr(self._solver, "config", None)
			if cfg is not None:
				game_name = getattr(cfg, "game_name", None)
				if game_name is not None:
					print(f"[RNAD P{self.player_id}] loaded config game_name: {game_name}")

	def _decode_action_to_proposal(self, action_id: int) -> Optional[np.ndarray]:
		base = self.max_quantity_for_encoding + 1
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
		# Minimal decode similar to NFSP wrapper
		num_items = self._num_items
		item_quantities = observation[9 : 9 + num_items]
		self.items = list(map(int, item_quantities))
		pv_start = 9 + num_items
		pv_end = pv_start + num_items
		player_values = observation[pv_start:pv_end]
		self.valuation_vector = list(map(int, player_values))
		walk_away_value = observation[9 + 2 * num_items] if len(observation) > (9 + 2 * num_items) else 0.0
		self.walk_away_value = float(walk_away_value)

	def step(self, state, *, is_evaluation: bool = True) -> int:  # pyspiel.State
		cur = state.current_player()
		self.current_counter_offer_give_format = None
		self._process_observation(state.observation_tensor(self.player_id))
		if cur != self.player_id:
			self.action = "ERROR_WRONG_PLAYER"
			return 0

		try:
			legal = list(state.legal_actions(cur))
		except Exception:
			legal = []
		legal = [a for a in legal if 0 <= int(a) < self._num_actions]
		if not legal:
			self.action = "ERROR_NO_LEGAL_ACTIONS"
			return 0

		try:
			probs_dict: Dict[int, float] = self._solver.action_probabilities(state)  # type: ignore[attr-defined]
		except Exception:
			traceback.print_exc()
			return int(self._rng.choice(legal))

		probs = np.zeros(self._num_actions, dtype=np.float32)
		for a, p in probs_dict.items():
			if 0 <= int(a) < self._num_actions:
				probs[int(a)] = float(p)

		q = np.zeros_like(probs, dtype=float)
		q[legal] = probs[legal]
		mass = q.sum()
		if mass > 0:
			q /= mass
			final_action = int(self._rng.choice(q.size, p=q))
		else:
			final_action = int(self._rng.choice(legal))

		self.last_dist = q.copy()
		self.last_action = final_action
		self.last_prob = float(q[final_action]) if q[final_action] > 0 else 0.0

		try:
			action_str = state.action_to_string(cur, final_action).lower()
			if "agreement" in action_str:
				self.action = "ACCEPT"
			elif "walk" in action_str:
				self.action = "WALK"
			else:
				self.action = "COUNTEROFFER"
				proposal_keep = self._decode_action_to_proposal(final_action)
				if proposal_keep is not None and self.items is not None:
					give = (np.array(self.items) - proposal_keep).tolist()
					self.current_counter_offer_give_format = give
		except Exception:
			traceback.print_exc()
			self.action = "UNKNOWN"

		if self.debug:
			print(f"[RNAD P{self.player_id}] chosen={final_action} prob={self.last_prob:.3f}")
		return final_action



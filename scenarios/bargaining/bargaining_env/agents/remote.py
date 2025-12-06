from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Sequence, Tuple

from agentbeats.tool_provider import ToolProvider
from scenarios.prompts.make_prompt import make_prompt
from scenarios.utils.offer import Offer

from .base import BaseNegotiator

logger = logging.getLogger(__name__)


class RemoteNegotiatorError(RuntimeError):
	def __init__(self, message: str):
		super().__init__(message)


class RemoteNegotiator(BaseNegotiator):
	"""
	Proxy negotiator that forwards decisions to a remote purple agent via ToolProvider.
	"""

	def __init__(
		self,
		*,
		label: str,
		endpoint: str,
		tool_provider: ToolProvider | None = None,
		max_retries: int = 2,
		prompt_circle: int | None = None,
	):
		self._label = label
		self._endpoint = endpoint
		self._tool_provider = tool_provider or ToolProvider()
		self._max_retries = max_retries
		self._conversation_started = False
		self._context: Dict[str, Any] = {}
		self._pending_offer: Dict[str, Any] | None = None
		self._prompt_circle = prompt_circle
		self._history: Dict[int, List[Any]] = {0: [], 1: []}
		self._current_offer: Offer | None = None

	def set_context(
		self,
		*,
		pair_key: str,
		game_index: int,
		role: str,
		valuations_self: Sequence[int],
		valuations_opp: Sequence[int],
		batna_self: int,
		batna_opp: int,
		discount: float,
		max_rounds: int,
		quantities: Sequence[int],
		value_cap: int | None = None,
	) -> None:
		self._history = {0: [], 1: []}
		player_idx = 0 if role == "row" else 1
		other_idx = 1 - player_idx
		q_list = list(map(int, quantities))
		v_player1 = list(valuations_self) if role == "row" else list(valuations_opp)
		v_player2 = list(valuations_opp) if role == "row" else list(valuations_self)
		total_p1 = int(sum(v_player1[i] * q_list[i] for i in range(len(q_list))))
		total_p2 = int(sum(v_player2[i] * q_list[i] for i in range(len(q_list))))
		example_offer = [max(0, q // 2) for q in q_list]
		self._context = {
			"pair": pair_key,
			"game_index": game_index,
			"role": role,
			"valuations_self": list(valuations_self),
			"valuations_opp": list(valuations_opp),
			"batna_self": int(batna_self),
			"batna_opp": int(batna_opp),
			"discount": float(discount),
			"max_rounds": int(max_rounds),
			"quantities": list(quantities),
			"round_index": 1,
			"player_index": player_idx,
			"my_player_num": player_idx + 1,
			"other_player_num": other_idx + 1,
			"p1_outside_offer": [1, max(1, total_p1)],
			"p2_outside_offer": [1, max(1, total_p2)],
			"value_cap": int(value_cap) if value_cap is not None else None,
			"example_offer": example_offer,
			"batna_player1": int(batna_self if role == "row" else batna_opp),
			"batna_player2": int(batna_opp if role == "row" else batna_self),
		}
		self._pending_offer = {}
		self._conversation_started = False
		self._current_offer = None

	def set_round(self, round_index: int) -> None:
		if self._context:
			self._context["round_index"] = int(round_index)

	def set_offer_context(self, **offer_data: Any) -> None:
		if not self._context:
			return
		if self._pending_offer is None:
			self._pending_offer = {}
		proposer_role = offer_data.get("proposer")
		proposer_idx = None
		if isinstance(proposer_role, str):
			proposer_idx = 0 if proposer_role.lower() == "row" else 1
		for key, value in offer_data.items():
			if value is None:
				continue
			if isinstance(value, (list, tuple)):
				self._pending_offer[key] = [int(v) for v in value]
			else:
				self._pending_offer[key] = value
		if "offer_allocation_self" in self._pending_offer:
			allocation = self._pending_offer["offer_allocation_self"]
			try:
				allocation_list = [int(v) for v in allocation]
			except Exception:
				allocation_list = allocation
			player_number = (proposer_idx + 1) if proposer_idx is not None else self._context.get("other_player_num", 0)
			offer_obj = Offer(player=player_number, offer=allocation_list)
			self._current_offer = offer_obj
			self._history.setdefault(player_number - 1, []).append(offer_obj)

	def propose(
		self,
		quantities: Tuple[int, int, int],
		role: str,
		v_self: List[int],
		v_opp: List[int],
	) -> Tuple[List[int], List[int]]:
		observation = self._build_observation(action="propose", quantities=quantities)
		options = self._enumerate_allocations(quantities)
		prompt = self._format_prompt(
			action="PROPOSE",
			observation=observation,
			instruction=(
				"Return ONLY JSON. Preferred: "
				'{"allocation_self":[...],"allocation_other":[...],"reason":"..."} '
				"or use the catalog: {\"choice_id\": <int>, \"reason\": \"...\"}.\n"
				"No extra text. Arrays must sum to quantities. If you omit allocation_other, it will be inferred as the complement."
			),
			options=options,
		)
		response = self._send(prompt)
		# Support action-based responses from circle prompts: COUNTEROFFER with "offer" field
		if "action" in response and isinstance(response.get("action"), str):
			act = response.get("action", "").strip().upper()
			if act in ("COUNTEROFFER", "COUNTER_OFFER", "OFFER"):
				offer = response.get("offer") or response.get("allocation_self")
				if offer is not None:
					response = dict(response)
					response.setdefault("allocation_self", offer)
			# ACCEPT/WALK here fall back to normal parsing; invalid allocation will be handled below.
		allocation_self, allocation_other = self._extract_allocation(response, quantities, options)
		return allocation_self, allocation_other

	def accepts(self, offer_value: int, batna_value: int, counter_value: int) -> bool:
		observation = self._build_observation(
			action="ACCEPT_OR_REJECT",
			extra={
				"offer_value": int(offer_value),
				"batna_value": int(batna_value),
				"counter_value": int(counter_value),
			},
		)
		prompt = self._format_prompt(
			action="ACCEPT_OR_REJECT",
			observation=observation,
			instruction=(
				"Return ONLY JSON: "
				'{"accept": true|false, "reason": "...", "plan_allocation": [..optional..]}.\n'
				'Actions are ACCEPT (accept=true) or COUNTER_OFFER/WALK (accept=false). No extra text.'
			),
		)
		response = self._send(prompt)
		# Support action-based responses (ACCEPT/COUNTEROFFER/WALK) from circle prompts
		if "accept" in response:
			decision = response.get("accept")
		elif "decision" in response:
			decision = response.get("decision")
		elif "action" in response:
			act = str(response.get("action", "")).strip().lower()
			if act == "accept":
				decision = True
			elif act in ("counteroffer", "counter_offer", "offer", "walk"):
				decision = False
			else:
				decision = None
		else:
			decision = None
		if decision is None:
			raise RemoteNegotiatorError(f"{self._label} response missing 'accept' field: {response}")
		return self._coerce_bool(decision)

	def _build_observation(
		self,
		*,
		action: str,
		quantities: Tuple[int, int, int] | None = None,
		extra: Dict[str, Any] | None = None,
	) -> Dict[str, Any]:
		if not self._context:
			raise RemoteNegotiatorError("Remote negotiator context not initialized.")
		data = dict(self._context)
		data["action"] = action
		if quantities is not None:
			data["quantities"] = list(quantities)
		if self._pending_offer:
			data["pending_offer"] = self._pending_offer
		if extra:
			data.update(extra)
		# Do not leak opponent private info to the remote agent
		for k in ["valuations_opp", "batna_opp", "p2_outside_offer"]:
			data.pop(k, None)
		return data

	def _format_prompt(
		self,
		*,
		action: str,
		observation: Dict[str, Any],
		instruction: str,
		options: List[Dict[str, Any]] | None = None,
	) -> str:
		circle_prompt = self._build_circle_prompt()
		# Do not leak opponent-private info in the rendered prompt
		obs_public = dict(observation)
		for k in ("valuations_opp", "batna_opp", "p2_outside_offer"):
			obs_public.pop(k, None)
		message = (
			f"You are participating in the AgentBeats bargaining meta-game as '{self._label}'.\n"
			f"Action: {action}.\n"
			f"{instruction}\n"
			"Always answer with valid JSON only.\n"
			"Observation:\n"
			f"```json\n{json.dumps(obs_public, indent=2)}\n```"
		)
		if options:
			message += (
				"\nAllocation catalog (use `choice_id` to reference an entry):\n"
				f"```json\n{json.dumps(options, indent=2)}\n```"
			)
		if circle_prompt:
			message = f"{circle_prompt}\n\n----\n{message}"
		return message

	def _build_circle_prompt(self) -> str | None:
		# Build a circle prompt without leaking opponent-private info.
		if self._prompt_circle is None or not self._context:
			return None
		try:
			quantities = list(self._context.get("quantities", []))
			values = list(self._context.get("valuations_self", []))
			value_cap = int(self._context.get("value_cap") or (max(quantities or [1]) + 1))
			prompt_text = make_prompt(
				T=len(quantities),
				quantities=quantities,
				V=value_cap,
				values=values,
				W1=int(self._context.get("batna_self", 0)), #doesnt matter
				# Do not expose opponent BATNA; provide a neutral placeholder
				W2=0, #doesnt matter
				w=int(self._context.get("batna_self", 0)),
				R=int(self._context.get("max_rounds", 2)),
				g=float(self._context.get("discount", 0.98)),
				r=int(self._context.get("round_index", 1)),
				history=self._history,
				current_offer=self._current_offer,
				player_num=int(self._context.get("player_index", 0)),
				p1_outside_offer=self._context.get("p1_outside_offer"),
				# Do not expose opponent outside offer; use a neutral placeholder
				p2_outside_offer=self._context.get("p2_outside_offer"),
				circle=int(self._prompt_circle),
				example_offer_less_than_outside_offer_self=self._context.get("example_offer"),
			)
			return prompt_text.strip()
		except Exception as exc:  # noqa: BLE001
			logger.warning("Remote negotiator %s could not build circle prompt: %s", self._label, exc)
			return None

	def _enumerate_allocations(self, quantities: Sequence[int]) -> List[Dict[str, Any]]:
		counts = list(quantities)
		options: List[Dict[str, Any]] = []
		idx = 0
		for q0 in range(counts[0] + 1):
			for q1 in range(counts[1] + 1):
				for q2 in range(counts[2] + 1):
					allocation_self = [q0, q1, q2]
					allocation_other = [counts[0] - q0, counts[1] - q1, counts[2] - q2]
					options.append(
						{
							"id": idx,
							"allocation_self": allocation_self,
							"allocation_other": allocation_other,
						}
					)
					idx += 1
		return options

	def _send(self, prompt: str) -> Dict[str, Any]:
		last_exc: Exception | None = None
		for attempt in range(self._max_retries):
			try:
				response_text = asyncio.run(
					self._tool_provider.talk_to_agent(
						prompt,
						self._endpoint,
						new_conversation=not self._conversation_started,
					)
				)
				self._conversation_started = True
				return self._parse_json(response_text)
			except Exception as exc:  # noqa: BLE001
				last_exc = exc
				self._conversation_started = False
				logger.warning("Remote negotiator %s attempt %d failed: %s", self._label, attempt + 1, exc)
		# Fallback: treat as WALK to avoid stalling meta-game when remote emits non-JSON
		logger.warning("Remote negotiator %s exhausted retries; defaulting to WALK.", self._label)
		return {"action": "WALK", "reason": f"defaulted after non-JSON: {last_exc}"}

	def _parse_json(self, payload: str) -> Dict[str, Any]:
		candidates = []
		blocks = re.findall(r"```(?:json)?\s*(.*?)```", payload, flags=re.IGNORECASE | re.DOTALL)
		if blocks:
			candidates.extend(blocks)
		candidates.append(payload)
		for candidate in candidates:
			text = candidate.strip()
			if not text:
				continue
			try:
				data = json.loads(text)
				if isinstance(data, dict):
					return data
			except json.JSONDecodeError:
				continue
		# Heuristic fallbacks for plain-text replies
		pl = payload.strip().lower()
		if pl in {"accept", "accepted", "yes"}:
			return {"accept": True, "reason": "parsed from plain text"}
		if any(k in pl for k in ["walk", "reject", "decline", "decision complete"]):
			return {"accept": False, "action": "WALK", "reason": "parsed from plain text"}
		# Heuristic: bare list counteroffers like "[5,2,1]" -> allocation_self
		if payload.strip().startswith("[") and payload.strip().endswith("]"):
			try:
				arr = json.loads(payload)
				if isinstance(arr, list):
					return {"allocation_self": arr, "reason": "parsed from bare list"}
			except Exception:
				pass
		raise RemoteNegotiatorError(f"{self._label} returned non-JSON response: {payload[:200]}")

	def _extract_allocation(
		self,
		response: Dict[str, Any],
		quantities: Sequence[int],
		options: List[Dict[str, Any]],
	) -> Tuple[List[int], List[int]]:
		opt_lookup = {int(opt["id"]): opt for opt in options}
		option = None
		for key in ("choice_id", "choice_idx", "choice"):
			if key in response:
				try:
					option = opt_lookup[int(response[key])]
					break
				except (ValueError, KeyError, TypeError):
					continue

		if option:
			allocation_self = option["allocation_self"]
			allocation_other = option["allocation_other"]
		else:
			allocation_self = response.get("allocation_self") or response.get("allocation") or response.get("a_self")
			if allocation_self is None:
				raise RemoteNegotiatorError(f"{self._label} must provide allocation_self or choice_id.")
			allocation_other = (
				response.get("allocation_other")
				or response.get("other_allocation")
				or response.get("a_opp")
				or [quantities[i] - int(allocation_self[i]) for i in range(len(quantities))]
			)

		alloc_self = self._coerce_allocation(allocation_self, quantities, "allocation_self")
		alloc_other = self._coerce_allocation(allocation_other, quantities, "allocation_other")
		for i, total in enumerate(quantities):
			if alloc_self[i] + alloc_other[i] != int(total):
				raise RemoteNegotiatorError(f"{self._label} produced invalid allocation that does not sum to quantities.")
		return alloc_self, alloc_other

	def _coerce_allocation(self, raw: Sequence[Any], quantities: Sequence[int], label: str) -> List[int]:
		values = [int(float(v)) for v in raw]
		if len(values) != len(quantities):
			raise RemoteNegotiatorError(f"{self._label} {label} must match item dimension {len(quantities)}.")
		for idx, (val, total) in enumerate(zip(values, quantities, strict=True)):
			if val < 0 or val > int(total):
				raise RemoteNegotiatorError(f"{self._label} {label}[{idx}] out of range: {val} vs {total}.")
		return values

	def _coerce_bool(self, raw: Any) -> bool:
		if isinstance(raw, bool):
			return raw
		if isinstance(raw, (int, float)):
			return raw != 0
		if isinstance(raw, str):
			return raw.strip().lower() in {"1", "true", "accept", "accepted", "yes", "y"}
		raise RemoteNegotiatorError(f"{self._label} returned non-boolean decision: {raw}")


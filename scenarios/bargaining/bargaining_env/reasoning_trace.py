import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ReasoningTracer:
	def __init__(self, base_dir: Optional[str] = None, file_prefix: str = "llm_reasoning"):
		self._base = Path(base_dir or "bargaining_llm_traces")
		self._base.mkdir(parents=True, exist_ok=True)
		self._file_prefix = file_prefix

	def _default_file(self) -> Path:
		ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
		return self._base / f"{self._file_prefix}_{ts}.jsonl"

	def _resolve_path(self, agent: Optional[str], pair: Optional[str]) -> Path:
		# Organize logs by agent (if provided), then by pair (if provided)
		if agent:
			dir_path = self._base / agent
			dir_path.mkdir(parents=True, exist_ok=True)
			if pair:
				return dir_path / f"{pair}.jsonl"
			return dir_path / "session.jsonl"
		return self._default_file()

	def log(
		self,
		*,
		agent: Optional[str],
		pair: Optional[str],
		game: Optional[int],
		round_index: Optional[int],
		role: Optional[str],
		prompt: str,
		options: Optional[list[str]],
		raw_response: str,
		decision: str,
		extra_meta: Optional[Dict[str, Any]] = None,
	) -> None:
		record: Dict[str, Any] = {
			"timestamp": datetime.utcnow().isoformat() + "Z",
			"agent": agent,
			"pair": pair,
			"game": game,
			"round": round_index,
			"role": role,
			"prompt": prompt,
			"options": options,
			"raw_model_response": raw_response,
			"decision": decision,
		}
		if extra_meta:
			record["meta"] = extra_meta
		out_path = self._resolve_path(agent, pair)
		with out_path.open("a") as f:
			f.write(json.dumps(record) + "\n")



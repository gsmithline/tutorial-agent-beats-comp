import argparse
import contextlib
import importlib.util
import json
import os
from typing import Any, Callable, Optional
from dataclasses import dataclass
from urllib.parse import urljoin

from dotenv import load_dotenv
load_dotenv()

from google import genai
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater, InMemoryTaskStore
from a2a.types import (
	TaskState,
	Part,
	TextPart,
	Task,
	UnsupportedOperationError,
	AgentCard,
	AgentCapabilities,
	AgentSkill,
)
from a2a.utils import (
	new_agent_text_message,
	new_task,
)
from a2a.utils.errors import ServerError
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
import uvicorn

from scenarios.bargaining.bargaining_env.reasoning_trace import ReasoningTracer

def minimal_agent_card(name: str, url: str) -> AgentCard:
	skill = AgentSkill(
		id="bargaining_llm",
		name="Bargaining negotiation",
		description="Negotiate divisions of items via text",
		tags=["bargaining", "negotiation"],
		examples=[],
	)
	return AgentCard(
		name=name,
		version="0.1.0",
		description="LLM Agent for bargaining decisions",
		url=url,
		preferred_transport="JSONRPC",
		protocol_version="0.3.0",
		default_input_modes=["text"],
		default_output_modes=["text"],
		capabilities=AgentCapabilities(streaming=True),
		skills=[skill],
	)


def load_custom_decider(module_path: Optional[str]) -> Optional[Callable[[str, Optional[list[str]]], str]]:
	if not module_path:
		return None
	spec = importlib.util.spec_from_file_location("custom_agent", module_path)
	if spec is None or spec.loader is None:
		return None
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)  # type: ignore
	fn = getattr(mod, "decide", None)
	if callable(fn):
		return fn  # type: ignore
	return None


class LLMAgent(AgentExecutor):
	def __init__(
		self,
		model: str,
		system_prompt: Optional[str],
		custom_decider: Optional[Callable[[str, Optional[list[str]]], str]] = None,
		trace_dir: Optional[str] = None,
		agent_name: str = "BargainingLLM",
		# Provider config
		provider: str = os.environ.get("LLM_AGENT_PROVIDER", "gemini"),
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		headers: Optional[dict[str, str]] = None,
		temperature: float = 0.0,
		top_p: Optional[float] = None,
		timeout: int = 60,
	):
		self._provider = (provider or "gemini").lower()
		self._model = model
		self._system_prompt = system_prompt
		self._custom_decider = custom_decider
		self._tracer = ReasoningTracer(base_dir=trace_dir)
		self._agent_name = agent_name
		self._api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GOOGLE_API_KEY")
		self._base_url = base_url
		self._headers = headers or {}
		self._temperature = float(temperature) if temperature is not None else None
		self._top_p = float(top_p) if top_p is not None else None
		self._timeout = int(timeout)
		# Lazy clients (created on first use)
		self._gemini_client = None
		self._openai_client = None
		self._anthropic_client = None

	async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
		msg = context.message
		if not msg:
			raise ServerError("Missing message")
		task = new_task(msg)
		await event_queue.enqueue_event(task)
		updater = TaskUpdater(event_queue, task.id, task.context_id)

		await updater.update_status(TaskState.working, new_agent_text_message("LLM agent received request.", context_id=context.context_id))
		request_text = context.get_user_input()

		# Expected input JSON:
		# { "prompt": "...", "options": ["...", "..."] }  # options optional
		try:
			body = json.loads(request_text)
		except Exception:
			body = {"prompt": request_text}

		prompt = str(body.get("prompt", ""))
		options = body.get("options", None)
		if options is not None and not isinstance(options, list):
			options = None
		meta = body.get("meta", {}) if isinstance(body.get("meta", {}), dict) else {}
		pair = meta.get("pair")
		game_idx = meta.get("game")
		round_idx = meta.get("round")
		role = meta.get("role")

		raw_text = ""
		if self._custom_decider is not None:
			try:
				choice = self._custom_decider(prompt, options)  # returns string (either freeform or chosen option)
			except Exception as e:
				choice = f"[custom_decider_error] {e}"
		else:
			if options:
				# Ask the model to pick one option index; capture raw model text
				choice, raw_text = self._choose_from_options(prompt, options)
			else:
				choice, raw_text = self._freeform(prompt)

		# Log reasoning trace (prompt, options, raw model output, final decision)
		try:
			self._tracer.log(
				agent=self._agent_name,
				pair=pair,
				game=game_idx,
				round_index=round_idx,
				role=role,
				prompt=prompt,
				options=options,
				raw_response=raw_text if raw_text else str(choice),
				decision=str(choice),
				extra_meta={k: v for k, v in meta.items() if k not in {"pair", "game", "round", "role"}},
			)
		except Exception:
			# Tracing should not interfere with decisions
			pass

		await updater.update_status(TaskState.completed, new_agent_text_message("Decision complete.", context_id=context.context_id))
		await updater.add_artifact(parts=[Part(root=TextPart(text=str(choice)))], name="decision")

	async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
		raise ServerError(error=UnsupportedOperationError())

	def _freeform(self, prompt: str) -> tuple[str, str]:
		provider = self._provider
		sys_inst = self._system_prompt or ""
		# Gemini
		if provider == "gemini":
			if self._gemini_client is None:
				self._gemini_client = genai.Client()
			config_kwargs = {
				"system_instruction": sys_inst,
				"response_mime_type": "text/plain",
			}
			if self._temperature is not None:
				config_kwargs["temperature"] = self._temperature
			resp = self._gemini_client.models.generate_content(
				model=self._model,
				config=genai.types.GenerateContentConfig(**config_kwargs),
				contents=prompt,
			)
			text = resp.text or ""
			return text, text
		# OpenAI API
		if provider == "openai":
			try:
				from openai import OpenAI  # type: ignore
			except Exception as e:
				raise RuntimeError(f"openai package not available: {e}")
			if self._openai_client is None:
				self._openai_client = OpenAI(api_key=self._api_key)
			is_o3 = self._model.lower().startswith("o3-")
			create_kwargs = {
				"model": self._model,
				"messages": (
					([{"role": "system", "content": sys_inst}] if sys_inst else []) +
					[{"role": "user", "content": prompt}]
				),
				"timeout": self._timeout,
			}
			if (not is_o3) and self._temperature is not None:
				create_kwargs["temperature"] = self._temperature
			if (not is_o3) and self._top_p is not None:
				create_kwargs["top_p"] = self._top_p
			resp = self._openai_client.chat.completions.create(**create_kwargs)
			text = (resp.choices[0].message.content or "").strip()
			return text, text
		# Anthropic
		if provider == "anthropic":
			try:
				import anthropic  # type: ignore
			except Exception as e:
				raise RuntimeError(f"anthropic package not available: {e}")
			if self._anthropic_client is None:
				self._anthropic_client = anthropic.Anthropic(api_key=self._api_key)
			msg = self._anthropic_client.messages.create(
				model=self._model,
				max_tokens=2048,
				temperature=self._temperature if self._temperature is not None else 0.0,
				system=sys_inst if sys_inst else None,
				messages=[{"role": "user", "content": prompt}],
			)
			try:
				text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"])
			except Exception:
				text = ""
			return text, text
		# OpenAI-compatible HTTP endpoint (incl. Azure OpenAI, vLLM, Ollama w/ compat)
		if provider in ("openai_compat", "http", "http_compat"):
			try:
				import requests  # type: ignore
			except Exception as e:
				raise RuntimeError(f"requests package not available for HTTP provider: {e}")
			if not self._base_url:
				raise RuntimeError("base_url is required for openai_compat/http provider")
			url = self._base_url
			if "chat/completions" not in url:
				url = urljoin(self._base_url.rstrip("/") + "/", "v1/chat/completions")
			headers = dict(self._headers)
			if self._api_key and "authorization" not in {k.lower(): v for k, v in headers.items()}:
				headers["Authorization"] = f"Bearer {self._api_key}"
			payload = {
				"model": self._model,
				"messages": (
					([{"role": "system", "content": sys_inst}] if sys_inst else []) +
					[{"role": "user", "content": prompt}]
				),
			}
			if self._temperature is not None:
				payload["temperature"] = self._temperature
			r = requests.post(url, headers=headers, json=payload, timeout=self._timeout)
			r.raise_for_status()
			data = r.json()
			try:
				text = (data["choices"][0]["message"]["content"] or "").strip()
			except Exception:
				text = json.dumps(data)
			return text, text
		raise RuntimeError(f"Unsupported provider: {provider}")

	def _choose_from_options(self, prompt: str, options: list[str]) -> tuple[str, str]:
		sys_inst = (self._system_prompt or "") + "\nSelect the best option index and respond ONLY with the integer index."
		opt_text = "\n".join(f"{i}: {opt}" for i, opt in enumerate(options))
		content = f"{prompt}\nOptions:\n{opt_text}\nAnswer index only."
		# Reuse _freeform generation with adjusted system content
		orig_prompt = self._system_prompt
		try:
			self._system_prompt = sys_inst
			text, raw = self._freeform(content)
			text = (text or "").strip()
			try:
				idx = int(text.split()[0])
				if 0 <= idx < len(options):
					return options[idx], raw
			except Exception:
				pass
			return (options[0] if options else ""), raw
		finally:
			self._system_prompt = orig_prompt


@dataclass
class LLMSpec:
	model: str
	prompt: str


def main():
	parser = argparse.ArgumentParser(description="Run an LLM Agent for bargaining.")
	parser.add_argument("--host", type=str, default="127.0.0.1")
	parser.add_argument("--port", type=int, default=9039)
	parser.add_argument("--card-url", type=str)
	parser.add_argument("--cloudflare-quick-tunnel", action="store_true")
	parser.add_argument("--model", type=str, default=os.environ.get("LLM_AGENT_MODEL", "gemini-2.5-flash"))
	parser.add_argument("--system-prompt-file", type=str)
	parser.add_argument("--prompt", type=str, help="Inline system prompt text for the LLM agent")
	parser.add_argument("--custom-decider", type=str, help="Path to a Python file exposing decide(prompt, options)->str")
	parser.add_argument("--trace-dir", type=str, help="Directory to write JSONL reasoning logs")
	parser.add_argument("--agent-name", type=str, default="BargainingLLM", help="Agent name used in trace logs")
	# Provider settings
	parser.add_argument("--provider", type=str, default=os.environ.get("LLM_AGENT_PROVIDER", "gemini"), help="gemini | openai | anthropic | openai_compat | http")
	parser.add_argument("--api-key", type=str, default=os.environ.get("LLM_API_KEY"))
	parser.add_argument("--base-url", type=str, help="Base URL for provider (for openai_compat/http)")
	parser.add_argument("--headers-json", type=str, help="Path to JSON file with extra HTTP headers")
	parser.add_argument("--temperature", type=float, default=None)
	parser.add_argument("--top-p", type=float, default=None)
	parser.add_argument("--timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "60")))
	args = parser.parse_args()

	system_prompt = None
	if args.system_prompt_file and os.path.exists(args.system_prompt_file):
		system_prompt = open(args.system_prompt_file, "r").read()
	if system_prompt is None and args.prompt:
		system_prompt = args.prompt
	if system_prompt is None:
		raise RuntimeError("LLM Agent requires a prompt. Provide --system-prompt-file or --prompt.")
	custom_decider = load_custom_decider(args.custom_decider)

	if args.cloudflare_quick_tunnel:
		from agentbeats.cloudflare import quick_tunnel
		agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
	else:
		agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

	headers: dict[str, str] | None = None
	if args.headers_json and os.path.exists(args.headers_json):
		try:
			with open(args.headers_json, "r") as hf:
				headers = json.load(hf)
		except Exception:
			headers = None

	async def _serve():
		async with agent_url_cm as agent_url:
			executor = LLMAgent(
				model=args.model,
				system_prompt=system_prompt,
				custom_decider=custom_decider,
				trace_dir=args.trace_dir,
				agent_name=args.agent_name,
				provider=args.provider,
				api_key=args.api_key,
				base_url=args.base_url,
				headers=headers,
				temperature=args.temperature,
				top_p=args.top_p,
				timeout=args.timeout,
			)
			card = minimal_agent_card("BargainingLLM", agent_url)
			request_handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
			server = A2AStarletteApplication(agent_card=card, http_handler=request_handler)
			uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
			uvicorn_server = uvicorn.Server(uvicorn_config)
			await uvicorn_server.serve()

	import asyncio
	asyncio.run(_serve())


if __name__ == "__main__":
	main()



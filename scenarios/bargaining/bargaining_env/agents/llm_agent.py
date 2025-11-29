import argparse
import contextlib
import importlib.util
import json
import os
from typing import Any, Callable, Optional
from dataclasses import dataclass

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

def minimal_agent_card(name: str, url: str) -> dict[str, Any]:
	return {
		"name": name,
		"version": "0.1.0",
		"description": "LLM Agent for bargaining decisions",
		"endpoints": [{"type": "http", "url": url}],
	}


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
	def __init__(self, model: str, system_prompt: Optional[str], custom_decider: Optional[Callable[[str, Optional[list[str]]], str]] = None, trace_dir: Optional[str] = None, agent_name: str = "BargainingLLM"):
		self._client = genai.Client()
		self._model = model
		self._system_prompt = system_prompt
		self._custom_decider = custom_decider
		self._tracer = ReasoningTracer(base_dir=trace_dir)
		self._agent_name = agent_name

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
		resp = self._client.models.generate_content(
			model=self._model,
			config=genai.types.GenerateContentConfig(
				system_instruction=self._system_prompt or "",
				response_mime_type="text/plain",
			),
			contents=prompt,
		)
		text = resp.text or ""
		return text, text

	def _choose_from_options(self, prompt: str, options: list[str]) -> tuple[str, str]:
		sys_inst = (self._system_prompt or "") + "\nSelect the best option index and respond ONLY with the integer index."
		opt_text = "\n".join(f"{i}: {opt}" for i, opt in enumerate(options))
		content = f"{prompt}\nOptions:\n{opt_text}\nAnswer index only."
		resp = self._client.models.generate_content(
			model=self._model,
			config=genai.types.GenerateContentConfig(
				system_instruction=sys_inst,
				response_mime_type="text/plain",
			),
			contents=content,
		)
		text = (resp.text or "").strip()
		try:
			idx = int(text.split()[0])
			if 0 <= idx < len(options):
				return options[idx], text
		except Exception:
			pass
		# Fallback to first option if parsing fails
		return (options[0] if options else ""), text


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

	async def _serve():
		async with agent_url_cm as agent_url:
			executor = LLMAgent(
				model=args.model,
				system_prompt=system_prompt,
				custom_decider=custom_decider,
				trace_dir=args.trace_dir,
				agent_name=args.agent_name,
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



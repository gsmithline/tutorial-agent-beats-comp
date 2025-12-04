import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# Ensure project root is on path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import contextlib

# Optional server dependencies; provide fallbacks for CLI 'once' mode
HAVE_A2A = True
try:
    from agentbeats.green_executor import GreenAgent, GreenExecutor  # type: ignore
    from agentbeats.models import EvalRequest, EvalResult  # type: ignore
    from a2a.types import TaskState, Part, TextPart  # type: ignore
    from a2a.server.tasks import TaskUpdater  # type: ignore
    from a2a.utils import new_agent_text_message  # type: ignore
    from a2a.server.apps import A2AStarletteApplication  # type: ignore
    from a2a.server.request_handlers import DefaultRequestHandler  # type: ignore
    from a2a.server.tasks import InMemoryTaskStore  # type: ignore
    import uvicorn  # type: ignore
except Exception:
    HAVE_A2A = False

    class GreenAgent:  # type: ignore
        pass

    class EvalRequest:  # type: ignore
        def __init__(self, participants: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
            self.participants = participants
            self.config = config or {}

    class EvalResult:  # type: ignore
        def __init__(self, winner: str, detail: Dict[str, Any]):
            self.winner = winner
            self.detail = detail

        def model_dump(self) -> Dict[str, Any]:
            return {"winner": self.winner, "detail": self.detail}

    class TaskState:  # type: ignore
        working = "working"
        completed = "completed"

    def new_agent_text_message(text: str, context_id: Optional[str] = None) -> Any:  # type: ignore
        class _Msg:
            pass
        m = _Msg()
        m.text = text
        return m

from .bargaining_env.run_entire_matrix import run_matrix_pipeline
from .bargaining_env.main import run_analysis

logger = logging.getLogger("bargaining_green")
logging.basicConfig(level=logging.INFO)


class BargainingGreenAgent(GreenAgent):
    """
    Green orchestrator for the bargaining meta-game framework.
    It simulates OpenSpiel negotiation games for a roster of agents,
    then runs the meta-game analysis to compute regrets/welfare metrics.
    """

    def __init__(self):
        # Require at least one challenger participant to compare against baselines.
        self._required_roles = ["challenger"]
        self._required_config_keys = []

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        participants = request.participants or {}
        missing_roles = set(self._required_roles) - set(participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        cfg = request.config or {}
        full_matrix = cfg.get("full_matrix", True)
        model = cfg.get("model")
        circle = cfg.get("circle")
        if not full_matrix and (model is None or circle is None):
            return False, "When full_matrix is false, provide both 'model' and 'circle' in config."
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: Any) -> None:
        cfg = req.config or {}
        participants = req.participants or {}
        await updater.update_status(
            TaskState.working, new_agent_text_message("Starting bargaining simulations...")
        )

        # Simulation parameters
        sim_kwargs: Dict[str, Any] = {
            "full_matrix": cfg.get("full_matrix", True),
            "matrix_id": cfg.get("matrix_id", 1),
            "model": cfg.get("model"),
            "circle": cfg.get("circle"),
            "date": cfg.get("date"),
            "max_rounds": cfg.get("max_rounds", 5),
            "games": cfg.get("games", 20),
            "total_games": cfg.get("total_games"),
            "parallel": cfg.get("parallel", True),
            "discount": cfg.get("discount", 0.98),
            "skip_existing": cfg.get("skip_existing", False),
            "force_new_dirs": cfg.get("force_new_dirs", False),
            "dry_run": cfg.get("dry_run", False),
            "use_openspiel": cfg.get("use_openspiel", True),
            "num_items": cfg.get("num_items", 3),
            "debug": cfg.get("debug", False),
        }

        challenger_label = cfg.get("challenger_label", "challenger")
        challenger_url = participants.get("challenger")
        remote_agents_cfg: Dict[str, str] = {}
        cfg_remote_agents = cfg.get("remote_agents")
        if isinstance(cfg_remote_agents, dict):
            remote_agents_cfg.update({str(k): str(v) for k, v in cfg_remote_agents.items()})
        if challenger_url:
            remote_agents_cfg.setdefault(str(challenger_label), str(challenger_url))
            sim_kwargs["challenger_url"] = str(challenger_url)
        sim_kwargs["challenger_label"] = str(challenger_label)
        if remote_agents_cfg:
            sim_kwargs["remote_agents"] = remote_agents_cfg
        # Optional prompt circle selection for remote entrants
        challenger_circle = cfg.get("challenger_circle")
        circle_map: Dict[str, int] = {}
        cfg_remote_circles = cfg.get("remote_agent_circles")
        if isinstance(cfg_remote_circles, dict):
            for label, circle_val in cfg_remote_circles.items():
                try:
                    circle_map[str(label)] = int(circle_val)
                except Exception:
                    continue
        if challenger_circle is not None:
            try:
                circle_map.setdefault(str(challenger_label), int(challenger_circle))
            except Exception:
                pass
        if circle_map:
            sim_kwargs["remote_agent_circles"] = circle_map

        model_circles = cfg.get("model_circles")
        model_shortnames = cfg.get("model_shortnames")

        def _run_matrix():
            return run_matrix_pipeline(
                model_circles=model_circles,
                model_shortnames=model_shortnames,
                **sim_kwargs,
				nfsp_checkpoint_path=cfg.get("nfsp_checkpoint_path"),
				rnad_checkpoint_path=cfg.get("rnad_checkpoint_path"),
            )

        sim_result = await asyncio.to_thread(_run_matrix)
        base_dir = sim_result.get("base_dir")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Simulation complete. Data in {base_dir}. Starting meta-game analysis..."),
        )

        output_dir = cfg.get("output_dir", "meta_game_analysis/results_bargaining")
        analysis_kwargs = {
            "input_dir": base_dir,
            "output_dir": output_dir,
            "discount_factor": cfg.get("discount", 0.98),
            "num_bootstrap": cfg.get("bootstrap", 100),
            "norm_constants": cfg.get("norm_constants", {}),
            "random_seed": cfg.get("random_seed", 42),
        }

        await asyncio.to_thread(run_analysis, **analysis_kwargs)

        result = EvalResult(
            winner="meta_game",
            detail={
                "simulation_output_dir": base_dir,
                "analysis_output_dir": output_dir,
                "experiments_ran": len(sim_result.get("experiments", [])),
            },
        )

        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(
                f"Meta-game analysis complete. Results saved to {output_dir}.", context_id=updater.context_id
            ),
        )
        if HAVE_A2A:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=json.dumps(result.model_dump())))],
                name="meta_game_result",
            )


def _run_once_from_cli(config_path: Optional[str]) -> None:
    cfg_raw: Dict[str, Any] = {}
    if config_path:
        with open(config_path, "r") as f:
            cfg_raw = json.load(f)
    cfg: Dict[str, Any] = dict(cfg_raw)
    participants = cfg.pop("participants", {})
    inline_challenger = cfg.pop("challenger_url", None)
    if inline_challenger and "challenger" not in participants:
        participants["challenger"] = inline_challenger

    dummy_req = EvalRequest(participants=participants, config=cfg)
    agent = BargainingGreenAgent()
    ok, msg = agent.validate_request(dummy_req)
    if not ok:
        raise ValueError(msg)

    class DummyUpdater:
        """Minimal stand-in for TaskUpdater when running via CLI."""

        def __init__(self):
            self.context_id = None

        async def update_status(self, status, message):
            logger.info(f"{status}: {getattr(message, 'text', message)}")

        async def add_artifact(self, parts, name):
            logger.info(f"Artifact ({name}): {parts}")

        async def complete(self):
            pass

    updater = DummyUpdater()
    asyncio.run(agent.run_eval(dummy_req, updater))  # type: ignore[arg-type]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the A2A bargaining green agent or a one-off CLI run.")
    sub = parser.add_subparsers(dest="mode", required=False)
    # server mode
    p_srv = sub.add_parser("serve", help="Start the bargaining green A2A server")
    p_srv.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    p_srv.add_argument("--port", type=int, default=9029, help="Port to bind the server")
    p_srv.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    p_srv.add_argument("--cloudflare-quick-tunnel", action="store_true", help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url")
    # one-off mode
    p_one = sub.add_parser("once", help="Run a single simulation + analysis from a JSON config")
    p_one.add_argument("--config", type=str, help="Path to JSON config matching EvalRequest.config")
    args = parser.parse_args()

    if args.mode == "once":
        _run_once_from_cli(getattr(args, "config", None))
    else:
        if not HAVE_A2A:
            raise ImportError("Server mode requires 'agentbeats' and 'a2a' packages. Use 'once' mode or install dependencies.")
        # default to server mode if no subcommand
        if getattr(args, "mode", None) is None:
            args.mode = "serve"
            args.host = "127.0.0.1"
            args.port = 9029
            args.card_url = None
            args.cloudflare_quick_tunnel = False

        try:
            from scenarios.debate.debate_judge_common import debate_judge_agent_card as _card  # reuse simple card builder
        except Exception:
            # Minimal agent card if debate module unavailable
            def _card(name: str, url: str) -> Dict[str, Any]:
                return {
                    "name": name,
                    "version": "0.1.0",
                    "description": "Bargaining Green Agent",
                    "endpoints": [{"type": "http", "url": url}],
                }

        if args.cloudflare_quick_tunnel:
            from agentbeats.cloudflare import quick_tunnel
            agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
        else:
            from a2a.utils import strip_trailing_slash
            base_url = f"http://{args.host}:{args.port}/"
            agent_url_cm = contextlib.nullcontext(args.card_url or base_url)

        async def _serve() -> None:
            async with agent_url_cm as agent_url:
                agent = BargainingGreenAgent()
                executor = GreenExecutor(agent)
                agent_card = _card("BargainingGreen", agent_url)
                request_handler = DefaultRequestHandler(
                    agent_executor=executor,
                    task_store=InMemoryTaskStore(),
                )
                server = A2AStarletteApplication(
                    agent_card=agent_card,
                    http_handler=request_handler,
                )
                uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
                uvicorn_server = uvicorn.Server(uvicorn_config)
                await uvicorn_server.serve()

        asyncio.run(_serve())

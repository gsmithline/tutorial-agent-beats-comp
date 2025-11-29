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

from agentbeats.green_executor import GreenAgent
from agentbeats.models import EvalRequest, EvalResult
from a2a.types import TaskState
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message

from bargaining_env.run_entire_matrix import run_matrix_pipeline
from bargaining_env.main import run_analysis

logger = logging.getLogger("bargaining_green")
logging.basicConfig(level=logging.INFO)


class BargainingGreenAgent(GreenAgent):
    """
    Green orchestrator for the bargaining meta-game framework.
    It simulates OpenSpiel negotiation games for a roster of agents,
    then runs the meta-game analysis to compute regrets/welfare metrics.
    """

    def __init__(self):
        # No fixed roles; participants can be empty.
        self._required_roles = []
        self._required_config_keys = []

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        cfg = request.config or {}
        full_matrix = cfg.get("full_matrix", True)
        model = cfg.get("model")
        circle = cfg.get("circle")
        if not full_matrix and (model is None or circle is None):
            return False, "When full_matrix is false, provide both 'model' and 'circle' in config."
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        cfg = req.config or {}
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
            "parallel": cfg.get("parallel", True),
            "discount": cfg.get("discount", 0.98),
            "skip_existing": cfg.get("skip_existing", False),
            "force_new_dirs": cfg.get("force_new_dirs", False),
            "dry_run": cfg.get("dry_run", False),
            "use_openspiel": cfg.get("use_openspiel", True),
            "num_items": cfg.get("num_items", 3),
            "debug": cfg.get("debug", False),
        }

        model_circles = cfg.get("model_circles")
        model_shortnames = cfg.get("model_shortnames")

        def _run_matrix():
            return run_matrix_pipeline(
                model_circles=model_circles,
                model_shortnames=model_shortnames,
                **sim_kwargs,
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
            "num_bootstrap": cfg.get("bootstrap", 100),
            "confidence": cfg.get("confidence", 0.95),
            "global_samples": cfg.get("global_samples", 100000),
            "use_raw_bootstrap": cfg.get("use_raw_bootstrap", True),
            "discount_factor": cfg.get("discount", 0.98),
            "track_br_evolution": cfg.get("track_br_evolution", False),
            "batch_size": cfg.get("batch_size", 100),
            "num_br_bootstraps": cfg.get("num_br_bootstraps", 100),
            "random_seed": cfg.get("random_seed", 42),
            "use_cell_based_bootstrap": cfg.get("use_cell_based_bootstrap", False),
            "run_temporal_analysis": cfg.get("run_temporal_analysis", False),
            "full_game_mix": cfg.get("full_game_mix", True),
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
        await updater.add_artifact(
            parts=[new_agent_text_message(json.dumps(result.model_dump()))],
            name="meta_game_result",
        )


def _run_once_from_cli(config_path: Optional[str]) -> None:
    cfg: Dict[str, Any] = {}
    if config_path:
        with open(config_path, "r") as f:
            cfg = json.load(f)

    dummy_req = EvalRequest(participants={}, config=cfg)
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
    parser = argparse.ArgumentParser(description="Run bargaining green agent once via CLI.")
    parser.add_argument("--config", type=str, help="Path to JSON config matching EvalRequest.config")
    args = parser.parse_args()
    _run_once_from_cli(args.config)

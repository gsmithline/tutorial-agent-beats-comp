"""
Minimal controller entrypoint for AgentBeats Cloud Run deployment.
This wraps the existing BargainingGreenAgent server implementation.
"""
import os
import asyncio
import logging
from typing import Dict, Any

# Ensure project root is on path for local imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scenarios.bargaining.bargaining_green import BargainingGreenAgent
from agentbeats.green_executor import GreenExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
import uvicorn

logger = logging.getLogger("bargaining_controller")
logging.basicConfig(level=logging.INFO)


def _create_agent_card(name: str, url: str) -> Dict[str, Any]:
    """Create a minimal agent card for the bargaining green agent."""
    return {
        "name": name,
        "version": "0.1.0",
        "description": "Bargaining Green Agent - Meta-game analysis controller",
        "endpoints": [{"type": "http", "url": url}],
    }


async def _serve(host: str = "0.0.0.0", port: int = 8080, card_url: str = None) -> None:
    """Start the A2A server for the bargaining green agent."""
    # Use PORT environment variable if set (Cloud Run convention)
    port = int(os.environ.get("PORT", port))
    
    # Determine card URL - use provided URL or construct from host/port
    if not card_url:
        # In Cloud Run, we need the public URL, but we'll use a placeholder
        # The actual URL will be set by Cloud Run's environment
        card_url = f"http://{host}:{port}/"
    
    agent = BargainingGreenAgent()
    executor = GreenExecutor(agent)
    agent_card = _create_agent_card("BargainingGreen", card_url)
    
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    uvicorn_config = uvicorn.Config(server.build(), host=host, port=port)
    uvicorn_server = uvicorn.Server(uvicorn_config)
    logger.info(f"Starting bargaining green agent server on {host}:{port}")
    await uvicorn_server.serve()


def main():
    """Main entrypoint for the controller."""
    # Cloud Run sets PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Card URL can be set via environment variable for Cloud Run
    card_url = os.environ.get("CARD_URL")
    
    asyncio.run(_serve(host=host, port=port, card_url=card_url))


if __name__ == "__main__":
    main()


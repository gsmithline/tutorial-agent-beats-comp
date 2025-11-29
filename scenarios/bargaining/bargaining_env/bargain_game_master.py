import argparse, asyncio, json
import pyspiel
from agentbeats.client import start_server, TaskUpdate, Artifact, AssessmentRequestError  # adjust imports to your SDK layout
import numpy as np
def build_game(params):
    return pyspiel.load_game("negotiation", params)

def classify_action(state, player, action_id):
    s = state.action_to_string(player, action_id).lower()
    if "agreement" in s:
        return "ACCEPT"
    if "walk away" in s or ("get" in s and "points" in s):
        return "WALK"
    return "COUNTEROFFER"

async def run_match(p1_endpoint, p2_endpoint, params, turn_timeout=20.0):
    game = build_game(params)
    state = game.new_initial_state()

    def log(msg): print(f"[GREEN] {msg}")

    async def ask_agent(endpoint, payload):
        # TODO: replace with your A2A client call; stubbed as async placeholder
        raise NotImplementedError("wire to your A2A send/receive here")

    while True:
        if state.is_chance_node():
            action, _ = state.chance_outcomes()[0]
            state.apply_action(action)
            continue

        if state.is_terminal():
            returns = state.returns()
            return {
                "returns": returns,
                "terminal_state": True,
                "history": state.history_str(),
                "pla"
            }

        current = state.current_player()
        endpoint = p1_endpoint if current == 0 else p2_endpoint
        obs = state.observation_tensor(current)
        legal_actions = state.legal_actions(current)
        legal_desc = {a: state.action_to_string(current, a) for a in legal_actions}

        # Send task update to agent
        payload = {
            "observation": obs,
            "legal_actions": legal_actions,
            "legal_descriptions": legal_desc,
            "player": current,
        }
        try:
            resp = await asyncio.wait_for(ask_agent(endpoint, payload), timeout=turn_timeout)
        except asyncio.TimeoutError:
            raise AssessmentRequestError(f"Player {current} timed out")

        action_id = int(resp.get("action", -1))
        if action_id not in legal_actions:
            raise AssessmentRequestError(f"Illegal action {action_id} by player {current}")

        state.apply_action(action_id)
        log(f"P{current+1} -> {classify_action(state, current, action_id)} ({legal_desc[action_id]})")

def make_artifact(result):
    returns = result["returns"]
    fairness_gap = abs(returns[0] - returns[1])
    winner = 0 if returns[0] > returns[1] else 1 if returns[1] > returns[0] else None
    data = {
        "returns": returns,
        "nash_welfare": np.sqrt(returns[0] * returns[1]),
        "utilitarian_welfare": returns[0] + returns[1],
        "fairness_gap": fairness_gap,
        "winner": winner,
        "terminal_state": result.get("terminal_state", False),
        "history": result.get("history"),
    }
    return Artifact(kind="bargaining_result", data=data)
def ef1_check_outcome
async def handle_assessment(request):
    # Extract endpoints/config from the assessment_request
    p1 = request.participants[0].endpoint
    p2 = request.participants[1].endpoint

    params = {
        "enable_proposals": True,
        "enable_utterances": False,
        "num_items": int(request.config.get("num_items", 3)),
        "discount": float(request.config.get("discount", 0.9)),
        "min_value": 1,
        "max_value": 100,
        "max_rounds": int(request.config.get("max_rounds", 3)),
        "max_quantity": int(request.config.get("max_quantity", 10)),
        "item_quantities": request.config.get("item_quantities", "7,4,1"),
    }

    result = await run_match(p1, p2, params)
    return make_artifact(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url")
    args = parser.parse_args()
    start_server(
        host=args.host,
        port=args.port,
        card_url=args.card_url,
        handle_assessment=handle_assessment,
        name="BargainingGreenAgent"
    )

if __name__ == "__main__":
    main()




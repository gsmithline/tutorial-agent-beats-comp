import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pyspiel_integration import try_load_pyspiel_game, build_negotiation_params

# Simple, dependency-light OpenSpiel runner for reference data dumps.
# Only engages when pyspiel is available and explicitly requested.


AGREE_TOKEN = "Agreement"
WALK_TOKEN = "walk away"
PROPOSAL_RE = re.compile(r"Proposal:\s*\[([^\]]+)\]")


def _parse_keep_vector(action_str: str) -> Optional[List[int]]:
    m = PROPOSAL_RE.search(action_str)
    if not m:
        return None
    try:
        inner = m.group(1)
        nums = [int(x.strip()) for x in inner.split(",")]
        return nums
    except Exception:
        return None


def _list_actions(state) -> List[Tuple[int, str]]:
    acts = []
    try:
        la = state.legal_actions()
        for a in la:
            try:
                s = state.action_to_string(state.current_player(), a)
            except Exception:
                s = str(a)
            acts.append((a, s))
    except Exception:
        pass
    return acts


def _find_accept_action(actions: List[Tuple[int, str]]) -> Optional[int]:
    for a, s in actions:
        if AGREE_TOKEN in s:
            return a
    return None


def _find_walk_action(actions: List[Tuple[int, str]]) -> Optional[int]:
    for a, s in actions:
        if WALK_TOKEN in s.lower():
            return a
    return None


def _non_terminal_actions(actions: List[Tuple[int, str]]) -> List[int]:
    res = []
    for a, s in actions:
        if AGREE_TOKEN in s:
            continue
        if WALK_TOKEN in s.lower():
            continue
        res.append(a)
    return res


@dataclass
class TurnRecord:
    round_index: int
    player: int
    action: int
    action_str: str


@dataclass
class GameRecord:
    pair: str
    game_index: int
    returns: List[float]
    turns: List[TurnRecord]


def _soft_step(state) -> int:
    actions = _list_actions(state)
    # If there's an offer on the table, Accept should be legal: always accept
    a_acc = _find_accept_action(actions)
    if a_acc is not None:
        return a_acc
    # Otherwise propose randomly among non-terminal actions
    choices = _non_terminal_actions(actions)
    if choices:
        return random.choice(choices)
    # Fallback to any legal action
    return actions[0][0] if actions else 0


def _tough_step(state, quantities: Tuple[int, int, int]) -> int:
    actions = _list_actions(state)
    # Never walk or accept; choose a proposal that gives exactly 1 of an item if possible
    candidates: List[Tuple[int, str, List[int]]] = []
    for a, s in actions:
        if AGREE_TOKEN in s or WALK_TOKEN in s.lower():
            continue
        keep = _parse_keep_vector(s)
        if keep is None or len(keep) != len(quantities):
            continue
        give = [quantities[i] - keep[i] for i in range(len(quantities))]
        if sum(give) == 1 and all(g >= 0 for g in give):
            candidates.append((a, s, give))
    if candidates:
        # Prefer giving the lowest-index item
        candidates.sort(key=lambda t: next(i for i, g in enumerate(t[2]) if g == 1))
        return candidates[0][0]
    # Otherwise any non-terminal proposal
    choices = _non_terminal_actions(actions)
    if choices:
        return random.choice(choices)
    return actions[0][0] if actions else 0


def run_pyspiel_pair(
    *,
    pair_key: str,
    agent_row: str,
    agent_col: str,
    discount: float,
    max_rounds: int,
    num_items: int,
    quantities: Tuple[int, int, int],
    games: int,
    out_dir: Path,
) -> Dict[str, Any]:
    params = build_negotiation_params(
        discount=discount,
        max_rounds=max_rounds,
        num_items=num_items,
        item_quantities=quantities,
        min_value=1,
        max_value=100,
        max_quantity=10,
    )
    game = try_load_pyspiel_game(params)
    if game is None:
        return {"status": "pyspiel_not_available"}

    pair_dir = out_dir / "pyspiel_traces"
    pair_dir.mkdir(parents=True, exist_ok=True)
    records: List[GameRecord] = []

    for gi in range(games):
        state = game.new_initial_state()
        turn_log: List[TurnRecord] = []
        round_idx = 1
        # Drive chance nodes and turns
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                if not outcomes:
                    break
                action, _ = outcomes[0]
                state.apply_action(action)
                continue
            cur = state.current_player()
            # Choose step fn
            if cur == 0:
                # row agent
                if "soft" in agent_row.lower():
                    a = _soft_step(state)
                elif "tough" in agent_row.lower():
                    a = _tough_step(state, quantities)
                else:
                    a = _soft_step(state)  # default simple
            else:
                if "soft" in agent_col.lower():
                    a = _soft_step(state)
                elif "tough" in agent_col.lower():
                    a = _tough_step(state, quantities)
                else:
                    a = _soft_step(state)
            try:
                a_str = state.action_to_string(cur, a)
            except Exception:
                a_str = str(a)
            turn_log.append(TurnRecord(round_index=round_idx, player=cur, action=a, action_str=a_str))
            state.apply_action(a)
            if cur == 1:
                round_idx += 1
            if round_idx > max_rounds:
                break
        try:
            rets = state.returns()
            rets = [float(x) for x in rets]
        except Exception:
            rets = []
        records.append(GameRecord(pair=pair_key, game_index=gi, returns=rets, turns=turn_log))

    # Save JSON
    dump = [asdict(gr) for gr in records]
    out_file = pair_dir / f"{pair_key}.json"
    out_file.write_text(json.dumps(dump, indent=2))
    return {"status": "ok", "file": str(out_file), "games": games}



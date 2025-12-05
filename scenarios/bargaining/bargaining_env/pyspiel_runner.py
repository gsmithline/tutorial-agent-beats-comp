import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pyspiel_integration import try_load_pyspiel_game, build_negotiation_params
from .agents.nfsp import NFSPAgentWrapper
from .agents.rnad import RNaDAgentWrapper
from .agents.remote import RemoteNegotiator

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


# -------------------- NFSP-enabled runner producing JSONL traces --------------------

def _decode_basic_from_obs(obs: List[float], num_items: int) -> Tuple[List[int], List[int], float]:
    # Returns (items, values, batna)
    items = list(map(int, obs[9: 9 + num_items]))
    pv_start = 9 + num_items
    pv_end = pv_start + num_items
    vals = list(map(int, obs[pv_start:pv_end]))
    w = float(obs[9 + 2 * num_items]) if len(obs) > (9 + 2 * num_items) else 0.0
    return items, vals, w


def _value(v: List[int], a: List[int]) -> int:
    return v[0] * a[0] + v[1] * a[1] + v[2] * a[2]


def _is_ef1(v: List[int], a_self: List[int], a_other: List[int]) -> bool:
    self_val = _value(v, a_self)
    other_val = _value(v, a_other)
    if other_val <= self_val:
        return True
    max_item = 0
    for k in range(3):
        if a_other[k] > 0:
            max_item = max(max_item, v[k])
    return (other_val - self_val) <= max_item


def _aspiration_step(state, quantities: Tuple[int, int, int], keep_fraction: float = 0.85) -> int:
    # Choose a non-terminal proposal that keeps ~keep_fraction of total value (greedy)
    actions = _list_actions(state)
    choices = []
    for a, s in actions:
        if AGREE_TOKEN in s or WALK_TOKEN in s.lower():
            continue
        keep = _parse_keep_vector(s)
        if keep is None or len(keep) != len(quantities):
            continue
        choices.append(a)
    if choices:
        # Simple heuristic fallback: random among non-terminals
        return random.choice(choices)
    return actions[0][0] if actions else 0


def run_pyspiel_pair_nfsp_with_traces(
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
    nfsp_checkpoint_path: Optional[str],
    rnad_checkpoint_path: Optional[str],
    remote_agents: Optional[Dict[str, str]] = None,
    remote_agent_circles: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Run OpenSpiel negotiation games where at least one agent is NFSP or RNAD.
    Writes JSONL traces compatible with the lightweight simulator format and
    returns aggregate payoffs for payoffs.json.
    """
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
        raise RuntimeError("OpenSpiel 'negotiation' game is unavailable; cannot run NFSP matches.")

    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_file = traces_dir / f"{pair_key}.jsonl"

    remote_map: Dict[str, str] = {str(k): str(v) for k, v in (remote_agents or {}).items()}
    remote_circles: Dict[str, int] = {}
    for k, v in (remote_agent_circles or {}).items():
        try:
            remote_circles[str(k)] = int(v)
        except Exception:
            continue

    def _allocations_from_keep(keep: List[int], proposer: str) -> Tuple[List[int], List[int]]:
        # keep is items proposer keeps; other gets remainder
        other = [quantities[i] - keep[i] for i in range(len(quantities))]
        if proposer == "row":
            return keep, other
        return other, keep

    def _find_action_for_keep(actions: List[Tuple[int, str]], keep_vec: List[int]) -> Optional[int]:
        for a, s in actions:
            parsed = _parse_keep_vector(s)
            if parsed is None:
                continue
            try:
                if all(int(parsed[i]) == int(keep_vec[i]) for i in range(len(keep_vec))):
                    return a
            except Exception:
                continue
        return None

    # Aggregates
    n_accept = 0
    n_ef1 = 0
    sum_row = 0.0
    sum_col = 0.0

    for gi in range(games):
        state = game.new_initial_state()
        round_idx = 1
        # Instantiate NFSP/RNAD wrappers if needed
        is_row_nfsp = "nfsp" in agent_row.lower()
        is_col_nfsp = "nfsp" in agent_col.lower()
        is_row_rnad = "rnad" in agent_row.lower()
        is_col_rnad = "rnad" in agent_col.lower()
        nfsp_row = NFSPAgentWrapper(game, 0, checkpoint_path=nfsp_checkpoint_path, discount=discount, max_rounds=max_rounds) if is_row_nfsp else None
        nfsp_col = NFSPAgentWrapper(game, 1, checkpoint_path=nfsp_checkpoint_path, discount=discount, max_rounds=max_rounds) if is_col_nfsp else None
        rnad_row = RNaDAgentWrapper(game, 0, checkpoint_path=rnad_checkpoint_path) if is_row_rnad else None
        rnad_col = RNaDAgentWrapper(game, 1, checkpoint_path=rnad_checkpoint_path) if is_col_rnad else None
        is_row_remote = agent_row in remote_map
        is_col_remote = agent_col in remote_map
        remote_row = RemoteNegotiator(label=agent_row, endpoint=remote_map[agent_row], prompt_circle=remote_circles.get(agent_row)) if is_row_remote else None
        remote_col = RemoteNegotiator(label=agent_col, endpoint=remote_map[agent_col], prompt_circle=remote_circles.get(agent_col)) if is_col_remote else None

        # Capture valuations/BATNAs from observations (best-effort) once at first decision node
        v1 = v2 = [0, 0, 0]
        b1 = b2 = 0.0
        captured_info = False

        # Track last proposed keep vectors to reconstruct accepted allocation
        last_keep_from_row: Optional[List[int]] = None
        last_keep_from_col: Optional[List[int]] = None
        accepted = False
        accepted_round = 1

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                if not outcomes:
                    break
                action, _ = outcomes[0]
                state.apply_action(action)
                continue

            # First time at a decision node, capture obs-derived params
            if not captured_info:
                try:
                    obs0 = state.observation_tensor(0)
                    obs1 = state.observation_tensor(1)
                    _, v1, b1 = _decode_basic_from_obs(obs0, num_items)
                    _, v2, b2 = _decode_basic_from_obs(obs1, num_items)
                except Exception:
                    pass
                captured_info = True
                # Initialize remote contexts once valuations are known
                if remote_row is not None:
                    remote_row.set_context(
                        pair_key=pair_key,
                        game_index=gi,
                        role="row",
                        valuations_self=v1,
                        valuations_opp=v2,
                        batna_self=b1,
                        batna_opp=b2,
                        discount=discount,
                        max_rounds=max_rounds,
                        quantities=quantities,
                        value_cap=100,
                    )
                if remote_col is not None:
                    remote_col.set_context(
                        pair_key=pair_key,
                        game_index=gi,
                        role="col",
                        valuations_self=v2,
                        valuations_opp=v1,
                        batna_self=b2,
                        batna_opp=b1,
                        discount=discount,
                        max_rounds=max_rounds,
                        quantities=quantities,
                        value_cap=100,
                    )

            cur = state.current_player()
            if cur == 0:
                if is_row_remote and remote_row is not None:
                    actions = _list_actions(state)
                    remote_row.set_round(round_idx)
                    walk_action = _find_walk_action(actions)
                    choose_walk = False
                    if last_keep_from_col:
                        alloc_self, alloc_other = _allocations_from_keep(last_keep_from_col, "col")
                        remote_row.set_offer_context(
                            proposer="col",
                            offer_allocation_self=alloc_self,
                            offer_allocation_other=alloc_other,
                            round_index=round_idx,
                        )
                    a_acc = _find_accept_action(actions)
                    chosen_accept = False
                    if a_acc is not None and last_keep_from_col is not None:
                        alloc_self, _ = _allocations_from_keep(last_keep_from_col, "col")
                        offer_value = _value(v1, alloc_self)
                        batna_value = b1
                        counter_value = batna_value
                        try:
                            if remote_row.accepts(offer_value, batna_value, counter_value):
                                a = a_acc
                                chosen_accept = True
                        except Exception:
                            chosen_accept = False
                    if not chosen_accept:
                        try:
                            alloc_self, alloc_other = remote_row.propose(quantities, "row", v1, v2)
                            keep_vec = [int(x) for x in alloc_self]
                            a_keep = _find_action_for_keep(actions, keep_vec)
                            if a_keep is None:
                                choices = _non_terminal_actions(actions)
                                a = choices[0] if choices else (actions[0][0] if actions else 0)
                            else:
                                a = a_keep
                        except Exception:
                            choose_walk = True
                            if walk_action is not None:
                                a = walk_action
                            else:
                                a = actions[0][0] if actions else 0
                    if chosen_accept and walk_action is not None and 'walk' in (a_str := state.action_to_string(cur, a) if 'a' in locals() else ''):
                        pass
                elif is_row_nfsp and nfsp_row is not None:
                    a = nfsp_row.step(state)
                elif is_row_rnad and rnad_row is not None:
                    a = rnad_row.step(state)
                else:
                    if "tough" in agent_row.lower():
                        a = _tough_step(state, quantities)
                    elif "aspire" in agent_row.lower() or "aspiration" in agent_row.lower():
                        a = _aspiration_step(state, quantities)
                    else:
                        a = _soft_step(state)
            else:
                if is_col_remote and remote_col is not None:
                    actions = _list_actions(state)
                    remote_col.set_round(round_idx)
                    walk_action = _find_walk_action(actions)
                    choose_walk = False
                    if last_keep_from_row:
                        alloc_self, alloc_other = _allocations_from_keep(last_keep_from_row, "row")
                        remote_col.set_offer_context(
                            proposer="row",
                            offer_allocation_self=alloc_self,
                            offer_allocation_other=alloc_other,
                            round_index=round_idx,
                        )
                    a_acc = _find_accept_action(actions)
                    chosen_accept = False
                    if a_acc is not None and last_keep_from_row is not None:
                        alloc_self, _ = _allocations_from_keep(last_keep_from_row, "row")
                        offer_value = _value(v2, alloc_self)
                        batna_value = b2
                        counter_value = batna_value
                        try:
                            if remote_col.accepts(offer_value, batna_value, counter_value):
                                a = a_acc
                                chosen_accept = True
                        except Exception:
                            chosen_accept = False
                    if not chosen_accept:
                        try:
                            alloc_self, alloc_other = remote_col.propose(quantities, "col", v2, v1)
                            keep_vec = [int(x) for x in alloc_self]
                            a_keep = _find_action_for_keep(actions, keep_vec)
                            if a_keep is None:
                                choices = _non_terminal_actions(actions)
                                a = choices[0] if choices else (actions[0][0] if actions else 0)
                            else:
                                a = a_keep
                        except Exception:
                            choose_walk = True
                            if walk_action is not None:
                                a = walk_action
                            else:
                                a = actions[0][0] if actions else 0
                elif is_col_nfsp and nfsp_col is not None:
                    a = nfsp_col.step(state)
                elif is_col_rnad and rnad_col is not None:
                    a = rnad_col.step(state)
                else:
                    if "tough" in agent_col.lower():
                        a = _tough_step(state, quantities)
                    elif "aspire" in agent_col.lower() or "aspiration" in agent_col.lower():
                        a = _aspiration_step(state, quantities)
                    else:
                        a = _soft_step(state)

            # Decode keep vector for proposals
            try:
                a_str = state.action_to_string(cur, a)
            except Exception:
                a_str = str(a)
            keep_vec = _parse_keep_vector(a_str)
            if keep_vec is not None and len(keep_vec) == len(quantities):
                if cur == 0:
                    last_keep_from_row = keep_vec
                else:
                    last_keep_from_col = keep_vec

            # Check if accepting
            if AGREE_TOKEN in a_str:
                accepted = True
                accepted_round = round_idx

            state.apply_action(a)
            if cur == 1:
                round_idx += 1
            if round_idx > max_rounds:
                break

        # Compute payoffs and record JSONL
        if accepted:
            # Accepted allocation is the proposal on the table; if the last mover was col (1), then accepted_round logic already set
            # Use the last proposal emitted before acceptance: if acceptance happened by row, then previous was col's keep; and vice-versa.
            # Approximation: prefer the last_keep_from_col if available at even rounds, else last_keep_from_row.
            if accepted_round % 2 == 0:
                # Column just moved this round
                keep_for_row = last_keep_from_col or [quantities[0] // 2, quantities[1] // 2, quantities[2] // 2]
            else:
                keep_for_row = last_keep_from_row or [quantities[0] // 2, quantities[1] // 2, quantities[2] // 2]
            a1 = keep_for_row
            a2 = [quantities[i] - a1[i] for i in range(len(quantities))]
            disc = discount ** (accepted_round - 1)
            p1 = float(_value(v1, a1)) * disc
            p2 = float(_value(v2, a2)) * disc
            ef1_ok = _is_ef1(v1, a1, a2) and _is_ef1(v2, a2, a1)
        else:
            # Walk: both get BATNAs discounted at the last round reached (cap at max_rounds)
            end_round = min(round_idx, max_rounds)
            disc = discount ** (end_round - 1)
            p1 = float(b1) * disc
            p2 = float(b2) * disc
            ef1_ok = None

        # Write record
        rec = {
            "pair": pair_key,
            "game": gi,
            "accepted": bool(accepted),
            "round": int(accepted_round if accepted else min(round_idx, max_rounds)),
            "q": list(quantities),
            "v1": list(map(int, v1)),
            "v2": list(map(int, v2)),
            "b1": float(b1),
            "b2": float(b2),
            "a1": a1 if accepted else [0, 0, 0],
            "a2": a2 if accepted else [0, 0, 0],
            "payoff1": p1,
            "payoff2": p2,
            "ef1": ef1_ok,
        }
        with (trace_file).open("a") as f:
            f.write(json.dumps(rec) + "\n")
        # Aggregates
        if accepted:
            n_accept += 1
            if isinstance(ef1_ok, bool) and ef1_ok:
                n_ef1 += 1
        sum_row += p1
        sum_col += p2

    return {
        "pair": pair_key,
        "trace_file": str(trace_file),
        "accept_rate": n_accept / max(1, games),
        "ef1_rate": n_ef1 / max(1, n_accept) if n_accept else 0.0,
        "row_mean_payoff": sum_row / max(1, games),
        "col_mean_payoff": sum_col / max(1, games),
    }



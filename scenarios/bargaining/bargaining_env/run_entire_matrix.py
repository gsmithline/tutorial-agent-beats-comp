import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import concurrent.futures
from urllib.parse import urlsplit, urlunsplit

from .agents.soft import SoftNegotiator
from .agents.tough import ToughNegotiator
from .agents.aspiration import AspirationNegotiator
from .agents.base import BaseNegotiator
from .agents.remote import RemoteNegotiator, RemoteNegotiatorError
from .pyspiel_integration import build_negotiation_params, try_load_pyspiel_game
from .pyspiel_runner import run_pyspiel_pair, run_pyspiel_pair_nfsp_with_traces

# BGS parameters (small game): fixed items
Q_BGS: Tuple[int, int, int] = (7, 4, 1)  # quantities per item type
V_MAX_DEFAULT: int = 100

logger = logging.getLogger(__name__)


@dataclass
class GameParams:
    q: Tuple[int, int, int]
    v_max: int
    gamma: float
    max_rounds: int


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_endpoint(url: str) -> str:
    try:
        parsed = urlsplit(str(url))
    except Exception:
        return str(url)
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{hostname}{port}" if hostname or port else parsed.netloc
    sanitized = (parsed.scheme or "", netloc, parsed.path, "", "")
    return urlunsplit(sanitized)


def _sample_instance(v_max: int) -> Tuple[List[int], List[int], int, int]:
    # Sample valuations v1, v2 ~ U({1..v_max}) per item type; BATNAs ~ U(1..v_i·q)
    v1 = [random.randint(1, v_max) for _ in range(3)]
    v2 = [random.randint(1, v_max) for _ in range(3)]
    return v1, v2, 0, 0  # batnas computed after dot with q


def _dot(v: List[int], q: Tuple[int, int, int]) -> int:
    return v[0] * q[0] + v[1] * q[1] + v[2] * q[2]


def _policy_kind(agent_name: str) -> str:
    n = agent_name.lower()
    if "walk" in n:
        return "walk"
    if "tough" in n or "boulware" in n or "-c-6" in n or "c-6" in n:
        return "tough"
    if "aspire" in n or "aspiration" in n or "conceder" in n:
        return "aspiration"
    if "soft" in n or "conceder" in n or "-c-0" in n or "c-0" in n or "-c-1" in n or "c-1" in n or "-c-2" in n or "c-2" in n:
        return "soft"
    return "balanced"

def _agent_impl(kind: str) -> BaseNegotiator | None:
    if kind == "soft":
        return SoftNegotiator()
    if kind == "tough":
        return ToughNegotiator()
    if kind == "aspiration":
        return AspirationNegotiator()
    return None


def _propose_allocation(policy: str, q: Tuple[int, int, int]) -> Tuple[List[int], List[int]]:
    # Returns (a1, a2) allocations
    if policy == "walk":
        return [0, 0, 0], [0, 0, 0]
    if policy == "soft":
        # even-ish split
        a1 = [q[0] // 2, q[1] // 2, q[2] // 2]
        a2 = [q[0] - a1[0], q[1] - a1[1], q[2] - a1[2]]
        return a1, a2
    if policy == "tough":
        # keep most items; leave at least 1 if available
        a1 = [max(q[0] - 1, 0), max(q[1] - 1, 0), max(q[2] - 1, 0)]
        a2 = [q[0] - a1[0], q[1] - a1[1], q[2] - a1[2]]
        return a1, a2
    # balanced
    take = [math.ceil(q[0] * 0.6), math.ceil(q[1] * 0.6), math.ceil(q[2] * 0.6)]
    take = [min(take[0], q[0]), min(take[1], q[1]), min(take[2], q[2])]
    a1 = take
    a2 = [q[0] - a1[0], q[1] - a1[1], q[2] - a1[2]]
    return a1, a2


def _value(v: List[int], a: List[int]) -> int:
    return v[0] * a[0] + v[1] * a[1] + v[2] * a[2]


def _accepts(policy: str, offer_value: int, batna: int, counter_value: int) -> bool:
    # Simple acceptance thresholds by policy
    if policy == "walk":
        return False
    if policy == "soft":
        return offer_value >= batna
    if policy == "tough":
        return offer_value >= max(batna, int(counter_value * 1.1))
    # balanced
    return offer_value >= max(batna, int(counter_value * 0.95))


def _is_ef1(v: List[int], a_self: List[int], a_other: List[int]) -> bool:
    # EF1 condition: v·a_other - v·a_self <= max_{k: a_other[k] > 0} v[k]
    self_val = _value(v, a_self)
    other_val = _value(v, a_other)
    if other_val <= self_val:
        return True
    max_item = 0
    for k in range(3):
        if a_other[k] > 0:
            max_item = max(max_item, v[k])
    return (other_val - self_val) <= max_item


def _simulate_pair(
    agent_row: str,
    agent_col: str,
    params: GameParams,
    games: int,
    base_dir: Path,
    pair_key: str,
    rng_seed: int | None = None,
    remote_agents: Dict[str, str] | None = None,
    remote_agent_circles: Dict[str, int] | None = None,
    debug: bool = False,
) -> Dict[str, Any]:
    random_gen = random.Random(rng_seed)
    out_path = _ensure_dir(base_dir / "traces")
    trace_file = out_path / f"{pair_key}.jsonl"

    row_policy = _policy_kind(agent_row)
    col_policy = _policy_kind(agent_col)
    remote_agents = remote_agents or {}
    remote_agent_circles = remote_agent_circles or {}
    row_is_remote = agent_row in remote_agents
    col_is_remote = agent_col in remote_agents

    row_impl: BaseNegotiator | None
    col_impl: BaseNegotiator | None
    if row_is_remote:
        row_impl = RemoteNegotiator(
            label=agent_row,
            endpoint=remote_agents[agent_row],
            prompt_circle=remote_agent_circles.get(agent_row),
        )
    else:
        row_impl = _agent_impl(row_policy)
    if col_is_remote:
        col_impl = RemoteNegotiator(
            label=agent_col,
            endpoint=remote_agents[agent_col],
            prompt_circle=remote_agent_circles.get(agent_col),
        )
    else:
        col_impl = _agent_impl(col_policy)

    n_accept = 0
    n_ef1 = 0
    sum_row = 0.0
    sum_col = 0.0

    with open(trace_file, "w") as f:
        for g in range(games):
            if row_is_remote and row_impl is None:
                row_impl = RemoteNegotiator(
                    label=agent_row,
                    endpoint=remote_agents[agent_row],
                    prompt_circle=remote_agent_circles.get(agent_row),
                )
            if col_is_remote and col_impl is None:
                col_impl = RemoteNegotiator(
                    label=agent_col,
                    endpoint=remote_agents[agent_col],
                    prompt_circle=remote_agent_circles.get(agent_col),
                )

            v1 = [random_gen.randint(1, params.v_max) for _ in range(3)]
            v2 = [random_gen.randint(1, params.v_max) for _ in range(3)]
            b1 = random_gen.randint(1, _dot(v1, params.q))
            b2 = random_gen.randint(1, _dot(v2, params.q))

            if row_impl is not None and hasattr(row_impl, "set_context"):
                row_impl.set_context(
                    pair_key=pair_key,
                    game_index=g,
                    role="row",
                    valuations_self=v1,
                    valuations_opp=v2,
                    batna_self=b1,
                    batna_opp=b2,
                    discount=params.gamma,
                    max_rounds=params.max_rounds,
                    quantities=list(params.q),
                    value_cap=params.v_max,
                )
            if col_impl is not None and hasattr(col_impl, "set_context"):
                col_impl.set_context(
                    pair_key=pair_key,
                    game_index=g,
                    role="col",
                    valuations_self=v2,
                    valuations_opp=v1,
                    batna_self=b2,
                    batna_opp=b1,
                    discount=params.gamma,
                    max_rounds=params.max_rounds,
                    quantities=list(params.q),
                    value_cap=params.v_max,
                )

            # Round 1: row proposes
            if row_impl is not None and hasattr(row_impl, "set_round"):
                row_impl.set_round(1)
            if row_impl is not None:
                try:
                    a1_prop, a2_prop = row_impl.propose(params.q, role="row", v_self=v1, v_opp=v2)
                except RemoteNegotiatorError as err:
                    logger.warning("Remote agent %s failed to propose as row: %s", agent_row, err)
                    if debug:
                        print(f"[DEBUG] remote row {agent_row} propose fallback: {err}")
                    row_impl = None
                    a1_prop, a2_prop = _propose_allocation(row_policy, params.q)
            else:
                a1_prop, a2_prop = _propose_allocation(row_policy, params.q)
            v2_offer = _value(v2, a2_prop)
            v1_offer = _value(v1, a1_prop)

            if col_impl is not None and hasattr(col_impl, "set_offer_context"):
                col_impl.set_offer_context(
                    proposer="row",
                    offer_allocation_self=a2_prop,
                    offer_allocation_opp=a1_prop,
                    offer_value=v2_offer,
                    round_index=1,
                )

            #counterfactual if column proposes
            if col_impl is not None and hasattr(col_impl, "set_round"):
                col_impl.set_round(2)
            if col_impl is not None:
                try:
                    a2_counter, a1_counter = col_impl.propose(params.q, role="col", v_self=v2, v_opp=v1)
                except RemoteNegotiatorError as err:
                    logger.warning("Remote agent %s failed to propose as column: %s", agent_col, err)
                    if debug:
                        print(f"[DEBUG] remote col {agent_col} propose fallback: {err}")
                    col_impl = None
                    a2_counter, a1_counter = _propose_allocation(col_policy, params.q)
            else:
                a2_counter, a1_counter = _propose_allocation(col_policy, params.q)
            v2_counter_val = _value(v2, a2_counter)
            v1_counter_val = _value(v1, a1_counter)
            if col_impl is not None and hasattr(col_impl, "set_offer_context"):
                col_impl.set_offer_context(
                    counter_allocation_self=a2_counter,
                    counter_allocation_opp=a1_counter,
                    counter_value=v2_counter_val,
                )

            accepted = False
            accepted_round = 1
            a1_final, a2_final = a1_prop, a2_prop
            if col_impl is not None:
                try:
                    col_accepts = col_impl.accepts(v2_offer, b2, v2_counter_val)
                except RemoteNegotiatorError as err:
                    logger.warning("Remote agent %s failed to decide acceptance: %s", agent_col, err)
                    if debug:
                        print(f"[DEBUG] remote col {agent_col} accept fallback: {err}")
                    col_impl = None
                    col_accepts = _accepts(col_policy, v2_offer, b2, v2_counter_val)
            else:
                col_accepts = _accepts(col_policy, v2_offer, b2, v2_counter_val)
            if col_accepts:
                accepted = True
                accepted_round = 1
            else:
                # Round 2: column proposes
                a2_prop2, a1_prop2 = a2_counter, a1_counter
                v1_offer2 = _value(v1, a1_prop2)
                v2_offer2 = _value(v2, a2_prop2)
                a1_final, a2_final = a1_prop2, a2_prop2
                if row_impl is not None and hasattr(row_impl, "set_offer_context"):
                    row_impl.set_offer_context(
                        proposer="col",
                        offer_allocation_self=a1_prop2,
                        offer_allocation_opp=a2_prop2,
                        offer_value=v1_offer2,
                        counter_allocation_self=a1_prop,
                        counter_allocation_opp=a2_prop,
                        counter_value=v1_offer,
                        round_index=2,
                    )
                if row_impl is not None and hasattr(row_impl, "set_round"):
                    row_impl.set_round(2)
                if row_impl is not None:
                    try:
                        row_accepts = row_impl.accepts(v1_offer2, b1, v1_offer)
                    except RemoteNegotiatorError as err:
                        logger.warning("Remote agent %s failed to decide acceptance: %s", agent_row, err)
                        if debug:
                            print(f"[DEBUG] remote row {agent_row} accept fallback: {err}")
                        row_impl = None
                        row_accepts = _accepts(row_policy, v1_offer2, b1, v1_offer)
                else:
                    row_accepts = _accepts(row_policy, v1_offer2, b1, v1_offer)
                if row_accepts:
                    accepted = True
                    accepted_round = 2

            if not accepted:
                # walk: both get BATNAs, discounted by gamma^(r-1)
                end_round = 2 if params.max_rounds >= 2 else max(1, params.max_rounds)
                disc = params.gamma ** (end_round - 1)
                payoff1 = b1 * disc
                payoff2 = b2 * disc
                record = {
                    "pair": pair_key,
                    "game": g,
                    "accepted": False,
                    "terminal": "walk",
                    "round": end_round,
                    "q": params.q,
                    "v1": v1,
                    "v2": v2,
                    "b1": b1,
                    "b2": b2,
                    "a1": [0, 0, 0],
                    "a2": [0, 0, 0],
                    "payoff1": payoff1,
                    "payoff2": payoff2,
                    "ef1": None,
                }
                f.write(json.dumps(record) + "\n")
                sum_row += payoff1
                sum_col += payoff2
                continue

            # accepted allocation
            r_idx = accepted_round - 1
            disc = params.gamma ** r_idx
            v1_realized = _value(v1, a1_final)
            v2_realized = _value(v2, a2_final)
            payoff1 = v1_realized * disc
            payoff2 = v2_realized * disc
            ef1_ok = _is_ef1(v1, a1_final, a2_final) and _is_ef1(v2, a2_final, a1_final)

            record = {
                "pair": pair_key,
                "game": g,
                "accepted": True,
                "terminal": "accept",
                "round": accepted_round,
                "q": params.q,
                "v1": v1,
                "v2": v2,
                "b1": b1,
                "b2": b2,
                "a1": a1_final,
                "a2": a2_final,
                "payoff1": payoff1,
                "payoff2": payoff2,
                "ef1": ef1_ok,
            }
            f.write(json.dumps(record) + "\n")
            n_accept += 1
            n_ef1 += 1 if ef1_ok else 0
            sum_row += payoff1
            sum_col += payoff2

    return {
        "pair": pair_key,
        "trace_file": str(trace_file),
        "accept_rate": n_accept / max(1, games),
        "ef1_rate": n_ef1 / max(1, n_accept) if n_accept else 0.0,
        "row_mean_payoff": sum_row / max(1, games),
        "col_mean_payoff": sum_col / max(1, games),
    }


def run_matrix_pipeline(
    *,
    model_circles: List[str] | None,
    model_shortnames: Dict[str, str] | None,
    full_matrix: bool = True,
    matrix_id: int = 1,
    model: str | None = None,
    circle: int | str | None = None,
    date: str | None = None,
    max_rounds: int = 3,
    games: int = 50,
    total_games: int | None = None,
    parallel: int | bool = False,  # number of worker threads; False/0 => sequential
    discount: float = 0.98,
    skip_existing: bool = False,  # not used
    force_new_dirs: bool = False,  # not used
    dry_run: bool = False,
    use_openspiel: bool = True,  # must remain True
    num_items: int = 3,
    debug: bool = False,
    pyspiel_dump_games: int = 0,
    nfsp_checkpoint_path: str | None = None,
    rnad_checkpoint_path: str | None = None,
    challenger_label: str | None = None,
    challenger_url: str | None = None,
    remote_agents: Dict[str, str] | None = None,
    challenger_circle: int | None = None,
    remote_agent_circles: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Simulate bargaining for a roster of 'agents' on OpenSpiel and save traces and payoffs."""
    if not use_openspiel:
        raise ValueError("OpenSpiel negotiation is required for all matchups.")
    assert num_items == 3, "This lightweight pipeline only supports BGS (3 items)."
    tag = date or _now_tag()
    base_dir = _ensure_dir(Path("bargaining_runs") / f"BGS_matrix_{matrix_id}_{tag}")

    remote_agent_map: Dict[str, str] = {str(k): str(v) for k, v in (remote_agents or {}).items()}
    if challenger_label and challenger_url:
        remote_agent_map.setdefault(str(challenger_label), str(challenger_url))

    remote_circle_map_raw: Dict[str, int] = {}
    for k, v in (remote_agent_circles or {}).items():
        try:
            remote_circle_map_raw[str(k)] = int(v)
        except Exception:
            continue
    if challenger_label and challenger_circle is not None:
        try:
            remote_circle_map_raw.setdefault(str(challenger_label), int(challenger_circle))
        except Exception:
            pass

    if full_matrix:
        if not model_circles:
            agents = ["soft", "tough", "aspiration", "walk"]
        else:
            agents = [str(a) for a in model_circles]
    else:
        if model is None or circle is None:
            raise ValueError("When full_matrix is false, both 'model' and 'circle' must be provided.")
        agents = [f"{model}-c-{circle}"]

    if remote_agent_map:
        for label in remote_agent_map.keys():
            if label not in agents:
                agents.append(label)

    # Always include a 'soft' baseline for comparison
    if not any(str(a).lower() == "soft" for a in agents):
        agents.append("soft")
    # Always include a 'tough' baseline for comparison
    if not any(str(a).lower() == "tough" for a in agents):
        agents.append("tough")
    # Always include an 'aspiration' baseline for comparison
    if not any("aspire" in str(a).lower() or "aspiration" in str(a).lower() for a in agents):
        agents.append("aspiration")

    # Optional shortnames mapping
    short_map = {a: (model_shortnames[a] if model_shortnames and a in model_shortnames else a) for a in agents}
    agent_ids = [short_map[a] for a in agents]

    remote_agent_urls: Dict[str, str] = {}
    remote_agent_meta: Dict[str, Dict[str, str]] = {}
    remote_circle_map: Dict[str, int] = {}
    if remote_agent_map:
        for orig_label, url in remote_agent_map.items():
            resolved = short_map.get(orig_label, orig_label)
            remote_agent_urls[resolved] = url
            remote_agent_meta[resolved] = {
                "original_label": orig_label,
                "endpoint_hint": _sanitize_endpoint(url),
            }
    if remote_circle_map_raw:
        for orig_label, circle_val in remote_circle_map_raw.items():
            resolved = short_map.get(orig_label, orig_label)
            remote_circle_map[resolved] = int(circle_val)

    meta = {
        "agents": agent_ids,
        "original_agents": agents,
        "params": {
            "q": Q_BGS,
            "v_max": V_MAX_DEFAULT,
            "gamma": discount,
            "max_rounds": max_rounds,
            **(
                {"games_per_pair": games}
                if not total_games
                else {"total_games": int(total_games)}
            ),
        },
    }
    # Attach OpenSpiel negotiation config; must load successfully
    neg_params = build_negotiation_params(
        discount=discount,
        max_rounds=max_rounds,
        num_items=num_items,
        item_quantities=Q_BGS,
        min_value=1,
        max_value=V_MAX_DEFAULT,
        max_quantity=10,
    )
    game = try_load_pyspiel_game(neg_params)
    if game is None:
        raise RuntimeError("Failed to load OpenSpiel negotiation game; required for all matchups.")
    pyspiel_loaded = True
    meta["pyspiel"] = {
        "enabled": bool(use_openspiel),
        "loaded": bool(pyspiel_loaded),
        "negotiation_params": neg_params,
    }
    if remote_agent_meta:
        meta["remote_agents"] = remote_agent_meta
    (base_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if dry_run:
        return {"base_dir": str(base_dir), "experiments": []}

    params = GameParams(q=Q_BGS, v_max=V_MAX_DEFAULT, gamma=discount, max_rounds=max_rounds)
    results: Dict[str, Any] = {}
    experiments: List[str] = []

    # Determine role-balanced allocation
    # Build work items (pair_key, agent_row, agent_col, num_games)
    work: List[Tuple[str, str, str, int]] = []

    if total_games and total_games > 0:
        # Use unordered pairs (i <= j), split games evenly, and balance roles per pair
        unordered_pairs: List[Tuple[int, int]] = []
        n_agents = len(agent_ids)
        for i in range(n_agents):
            for j in range(i, n_agents):
                unordered_pairs.append((i, j))
        num_pairs = len(unordered_pairs) if unordered_pairs else 1
        base = total_games // num_pairs
        remainder = total_games % num_pairs

        for k, (i, j) in enumerate(unordered_pairs):
            ai, aj = agent_ids[i], agent_ids[j]
            per_pair_total = base + (1 if k < remainder else 0)
            g_row = per_pair_total // 2
            g_col = per_pair_total - g_row
            if g_row > 0:
                work.append((f"{ai}__vs__{aj}", ai, aj, g_row))
            if g_col > 0:
                work.append((f"{aj}__vs__{ai}", aj, ai, g_col))
    else:
        # Per-pair behavior: for each unordered matchup, run `games` total, split evenly across roles
        unordered_pairs: List[Tuple[int, int]] = []
        n_agents = len(agent_ids)
        for i in range(n_agents):
            for j in range(i, n_agents):
                unordered_pairs.append((i, j))

        for (i, j) in unordered_pairs:
            ai, aj = agent_ids[i], agent_ids[j]
            if i == j:
                # Self-play: run all as ai vs ai once
                work.append((f"{ai}__vs__{aj}", ai, aj, games))
                continue

            g_row = games // 2
            g_col = games - g_row
            if g_row > 0:
                work.append((f"{ai}__vs__{aj}", ai, aj, g_row))
            if g_col > 0:
                work.append((f"{aj}__vs__{ai}", aj, ai, g_col))

    # Decide concurrency
    if parallel:
        max_workers = parallel if isinstance(parallel, int) and parallel > 0 else None
    else:
        max_workers = 1

    def _run_pair(item: Tuple[str, str, str, int]) -> Tuple[str, Dict[str, Any]]:
        pair_key, row_agent, col_agent, g = item
        sim = run_pyspiel_pair_nfsp_with_traces(
            pair_key=pair_key,
            agent_row=row_agent,
            agent_col=col_agent,
            discount=discount,
            max_rounds=max_rounds,
            num_items=num_items,
            quantities=Q_BGS,
            games=g,
            out_dir=base_dir,
            nfsp_checkpoint_path=nfsp_checkpoint_path,
            rnad_checkpoint_path=rnad_checkpoint_path,
            remote_agents=remote_agent_urls,
            remote_agent_circles=remote_circle_map,
        )
        return pair_key, sim

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_pair, item) for item in work]
        for fut in concurrent.futures.as_completed(futures):
            pair_key, sim = fut.result()
            results[pair_key] = sim
            experiments.append(pair_key)
            if debug:
                print(f"Simulated {pair_key}: {sim['row_mean_payoff']:.1f} / {sim['col_mean_payoff']:.1f}")

    (base_dir / "payoffs.json").write_text(json.dumps(results, indent=2))
    return {"base_dir": str(base_dir), "experiments": experiments}



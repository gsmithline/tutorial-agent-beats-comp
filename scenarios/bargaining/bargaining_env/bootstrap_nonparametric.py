"""
Non-parametric bootstrapping implementation for game theoretic analysis.
"""

import os
import re
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
import warnings
from tqdm import tqdm
import random


plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nash_equilibrium.nash_solver import milp_max_sym_ent_2p, replicator_dynamics_nash

def run_value_bootstrap_agents(
    all_results: list,
    baseline_agent_substr: str = "o3",
    *,
    n_boot: int = 300,
    alpha: float = 0.05,
    rng=None,
):
    """Bootstrap replicate equilibrium-weighted per-agent metrics using a *cell-based* resampling
    scheme (by agent-pair and source_matrix), matching the pattern in
    `nonparametric_bootstrap_from_raw_data`.

    For each bootstrap replicate:
      1) Build a performance matrix (mean proposer utilities) by resampling *within each cell*
         (agent-pair × source_matrix) with replacement; average within source, then across sources.
      2) Impute missing entries via column mean, else row mean, else 0; solve max-entropy NE
         on the imputed performance matrix to get the mixed strategy π.
      3) Using the *same resampled games* (chosen indices), construct, for each metric of interest,
         an N×N directional matrix X where X[a,b] is the mean of that metric for offers from a→b
         (or the symmetric game-level mean duplicated into both directions).
      4) Impute each X and compute per-agent expectations vs the opponent mix:  X·π  (a row-wise
         expectation). Store those as the equilibrium-weighted per-agent numbers.
      5) For distributional EMDs vs baseline, use the offer lists pooled across the replicate
         and weight with agent supports when available.

    Notes:
      • We keep the full agent set (no masking), consistent with your 12×12 requirement.
      • Games involving excluded agent substrings should already be filtered *before* arriving here.
    """
    import numpy as np
    import numpy.random as npr
    from collections import defaultdict
    from tqdm import tqdm
    from scipy.stats import wasserstein_distance
    # Import project-specific functions
    from nash_equilibrium.nash_solver import milp_max_sym_ent_2p

    def _extract_bootstrap_cells(all_results):
        # Returns: Dict[(A,B), Dict[source_matrix, List[game_idx]]] (ordered pairs)
        cell_data = defaultdict(lambda: defaultdict(list))
        for i, rec in enumerate(all_results):
            a1 = rec.get("agent1")
            a2 = rec.get("agent2")
            s = rec.get("source_matrix", "unknown")
            if not a1 or not a2:
                continue
            cell_data[(a1, a2)][s].append(i)
        return cell_data

    rng = rng or npr.default_rng()
    cell_data = _extract_bootstrap_cells(all_results)

    # Fixed agent universe
    agents = sorted({r.get("agent1") for r in all_results if r.get("agent1")} |
                    {r.get("agent2") for r in all_results if r.get("agent2")})
    agent2idx = {a: i for i, a in enumerate(agents)}
    N = len(agents)

    # Stats accumulator across bootstrap replicates
    stats = defaultdict(list)

    # Baseline for EMDs (by substring)
    baseline_agents = [a for a in agents if baseline_agent_substr.lower() in (a or "").lower()]
    if not baseline_agents:
        raise ValueError(f"No agents contain substring '{baseline_agent_substr}' – cannot compute EMD baseline.")

    def _impute_square(mat: np.ndarray) -> np.ndarray:
        """Impute NaNs by column mean where possible, else row mean, else 0."""
        M = mat.astype(float).copy()
        # Column pass
        for j in range(M.shape[1]):
            col = M[:, j]
            col_mean = np.nanmean(col)
            if not np.isnan(col_mean):
                mask = np.isnan(col)
                if np.any(mask):
                    col[mask] = col_mean
                    M[:, j] = col
        # Row pass + final 0
        for i in range(M.shape[0]):
            row = M[i, :]
            mask = np.isnan(row)
            if np.any(mask):
                row_mean = np.nanmean(row)
                fill = 0.0 if np.isnan(row_mean) else row_mean
                row[mask] = fill
                M[i, :] = row
        return M

    for _ in tqdm(range(n_boot), desc="Bootstrap"):
        # ------------------------------------------------------------
        # 1) Build performance matrix by cell-based resampling
        # ------------------------------------------------------------
        perf_mat = np.full((N, N), np.nan)
        chosen_indices = []

        for (A, B), src_map in cell_data.items():
            # Aggregate via within-source bootstrap means, then average across sources
            src_means_A = []
            src_means_B = []
            for idx_list in src_map.values():
                if not idx_list:
                    continue
                picks = rng.choice(idx_list, size=len(idx_list), replace=True)
                chosen_indices.extend(picks)

                pa_vals, pb_vals = [], []
                for i_idx in picks:
                    rec = all_results[i_idx]
                    # Orient to (A as proposer side) for this ordered cell key
                    if rec.get("agent1") == A:
                        va = rec.get("agent1_value")
                        vb = rec.get("agent2_value")
                    else:
                        va = rec.get("agent2_value")
                        vb = rec.get("agent1_value")
                    if va is not None and not np.isnan(va):
                        pa_vals.append(float(va))
                    if vb is not None and not np.isnan(vb):
                        pb_vals.append(float(vb))

                if pa_vals:
                    src_means_A.append(float(np.mean(pa_vals)))
                if pb_vals:
                    src_means_B.append(float(np.mean(pb_vals)))

            if not (src_means_A or src_means_B):
                continue
            i, j = agent2idx[A], agent2idx[B]
            perf_mat[i, j] = float(np.mean(src_means_A)) if src_means_A else np.nan
            perf_mat[j, i] = float(np.mean(src_means_B)) if src_means_B else np.nan

        perf_filled = _impute_square(perf_mat)

        # ------------------------------------------------------------
        # 2) Solve NE (max-entropy) with robust fallback
        # ------------------------------------------------------------
        try:
            mix = milp_max_sym_ent_2p(perf_filled, 2000)
            if not np.all(np.isfinite(mix)) or mix.shape != (N,):
                raise RuntimeError("Invalid mix returned from solver.")
        except Exception:
            # Uniform fallback to keep the replicate
            mix = np.ones(N, dtype=float) / float(N)

        # Keep track of support for optional debugging
        for idx_a, ag in enumerate(agents):
            stats.setdefault(f"mix_{ag}", []).append(float(mix[idx_a]))

        # ------------------------------------------------------------
        # 3) Collect *pair-directional* lists for each metric using the
        #    SAME resampled games (chosen_indices) for this replicate
        # ------------------------------------------------------------
        offer_bank = defaultdict(list)  # for EMDs & baseline
        # Directional pair-banks: key = (proposer, opponent)
        offer_pair_bank = defaultdict(list)
        surplus_pair_bank = defaultdict(list)
        recv_pair_bank = defaultdict(list)
        tqcs_pair_bank = defaultdict(list)
        tqgs_pair_bank = defaultdict(list)
        acc_pair_bank = defaultdict(list)
        # Game-level (we duplicate to both directions when building matrices)
        util_pair_bank = defaultdict(list)  # scaled 0..1
        nash_pair_bank = defaultdict(list)  # scaled 0..1
        ef1_pair_bank = defaultdict(list)   # 0/1 flags

        for i_idx in chosen_indices:
            rec = all_results[i_idx]
            a1 = rec.get("agent1")
            a2 = rec.get("agent2")
            if not a1 or not a2:
                continue

            # Offer-level, proposer-valued "value_given"
            if rec.get("offers_agent1"):
                offer_bank[a1].extend(rec.get("offers_agent1", []))
                offer_pair_bank[(a1, a2)].extend(rec.get("offers_agent1", []))
            if rec.get("offers_agent2"):
                offer_bank[a2].extend(rec.get("offers_agent2", []))
                offer_pair_bank[(a2, a1)].extend(rec.get("offers_agent2", []))

            # Opponent surplus (proposer units)
            if rec.get("surplus_agent1"):
                surplus_pair_bank[(a1, a2)].extend(rec.get("surplus_agent1", []))
            if rec.get("surplus_agent2"):
                surplus_pair_bank[(a2, a1)].extend(rec.get("surplus_agent2", []))

            # Receiver-valued "value_given"
            if rec.get("recv_agent1"):
                recv_pair_bank[(a1, a2)].extend(rec.get("recv_agent1", []))
            if rec.get("recv_agent2"):
                recv_pair_bank[(a2, a1)].extend(rec.get("recv_agent2", []))

            # Top-q capture / grant shares
            if rec.get("tqcs_agent1"):
                tqcs_pair_bank[(a1, a2)].extend(rec.get("tqcs_agent1", []))
            if rec.get("tqcs_agent2"):
                tqcs_pair_bank[(a2, a1)].extend(rec.get("tqcs_agent2", []))
            if rec.get("tqgs_agent1"):
                tqgs_pair_bank[(a1, a2)].extend(rec.get("tqgs_agent1", []))
            if rec.get("tqgs_agent2"):
                tqgs_pair_bank[(a2, a1)].extend(rec.get("tqgs_agent2", []))

            # Accepted value_given
            if rec.get("acc_vg_agent1"):
                acc_pair_bank[(a1, a2)].extend(rec.get("acc_vg_agent1", []))
            if rec.get("acc_vg_agent2"):
                acc_pair_bank[(a2, a1)].extend(rec.get("acc_vg_agent2", []))

            # Game-level (duplicate to both directions)
            if rec.get("util_welfare_scaled") is not None:
                u = float(rec["util_welfare_scaled"])
                util_pair_bank[(a1, a2)].append(u)
                util_pair_bank[(a2, a1)].append(u)
            if rec.get("nash_welfare_scaled") is not None:
                nw = float(rec["nash_welfare_scaled"])
                nash_pair_bank[(a1, a2)].append(nw)
                nash_pair_bank[(a2, a1)].append(nw)
            if rec.get("ef1_flag") is not None:
                ef = float(rec["ef1_flag"])
                ef1_pair_bank[(a1, a2)].append(ef)
                ef1_pair_bank[(a2, a1)].append(ef)
            

        # ------------------------------------------------------------
        # 4) Build matrices X for each metric from the pair-banks,
        #    then impute and compute per-agent expectations vs π.
        # ------------------------------------------------------------
        def _mat_from_pair_bank(bank):
            M = np.full((N, N), np.nan, dtype=float)
            for (p, o), lst in bank.items():
                if not lst:
                    continue
                ip = agent2idx.get(p)
                io = agent2idx.get(o)
                if ip is None or io is None:
                    continue
                M[ip, io] = float(np.mean(lst))
            return M

        matrices = {
            "mean": _mat_from_pair_bank(offer_pair_bank),          # proposer-valued value_given
            "surplus": _mat_from_pair_bank(surplus_pair_bank),     # proposer-units, opponent surplus
            "recv": _mat_from_pair_bank(recv_pair_bank),           # receiver-valued
            "tqcs": _mat_from_pair_bank(tqcs_pair_bank),           # 0..1
            "tqgs": _mat_from_pair_bank(tqgs_pair_bank),           # 0..1
            "accvg": _mat_from_pair_bank(acc_pair_bank),           # proposer-valued accepted offers
            "util": _mat_from_pair_bank(util_pair_bank),           # 0..1 (already scaled)
            "nash": _mat_from_pair_bank(nash_pair_bank),           # 0..1 (already scaled)
            "ef1": _mat_from_pair_bank(ef1_pair_bank),             # 0/1
        }

        # Compute X·π for each metric and store per-agent values
        for key, M in matrices.items():
            M_imp = _impute_square(M)
            row_exp = M_imp @ mix  # shape (N,)
            if key == "ef1":
                row_exp = row_exp * 100.0  # store EF1 as percentage to match pretty printer
            for idx_a, ag in enumerate(agents):
                stats.setdefault(f"{key}_{ag}", []).append(float(row_exp[idx_a]))

        # ------------------------------------------------------------
        # 5) EMD vs baseline distribution of offers (distributional)
        # ------------------------------------------------------------
        baseline_vals, baseline_wts = [], []
        for ag in baseline_agents:
            vals = offer_bank.get(ag, [])
            if vals and mix[agent2idx[ag]] > 0:
                w = mix[agent2idx[ag]] / len(vals)
                baseline_vals.extend(vals)
                baseline_wts.extend([w] * len(vals))

        # If we have a baseline distribution, compute EMDs
        if baseline_vals and np.sum(baseline_wts) > 0:
            for ag in agents:
                vals = offer_bank.get(ag, [])
                if not vals:
                    continue
                p_ag = float(mix[agent2idx[ag]])
                try:
                    u_w = [(p_ag / len(vals))] * len(vals) if p_ag > 0 else None
                    if u_w:
                        emd = wasserstein_distance(vals, baseline_vals, u_weights=u_w, v_weights=baseline_wts)
                    else:
                        emd = wasserstein_distance(vals, baseline_vals)
                except Exception:
                    emd = wasserstein_distance(vals, baseline_vals)
                stats.setdefault(f"emd_{ag}", []).append(float(emd))
        # Else: silently skip EMD this replicate

    # ----------------------------------------------------------------
    # Summarise across bootstrap replicates for each metric/agent
    # ----------------------------------------------------------------
    summary = {}
    for k, lst in stats.items():
        arr = np.asarray(lst, dtype=float)
        if arr.size == 0:
            continue
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        summary[k] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=1)),
            "ci": (float(lo), float(hi)),
            "n": int(np.sum(np.isfinite(arr))),
        }
    return summary

def nonparametric_bootstrap_from_raw_data(all_results, num_bootstrap=1000, confidence=0.95,
                                         # Add arguments to accept global max values
                                         global_max_nash_welfare=None, 
                                         global_max_nash_welfare_adv=None,
                                         global_max_util_welfare=None):
    """
    Proper non-parametric bootstrapping directly from raw game results using cell-based approach.
    Bootstraps each cell in the performance matrix independently rather than resampling whole games.
    
    Args:
        all_results: List of dictionaries, each representing a processed game outcome from data_processing.py
        num_bootstrap: Number of bootstrap replicas
        confidence: Confidence level for intervals
        global_max_nash_welfare: Global maximum value for Nash welfare (optional)
        global_max_util_welfare: Global maximum value for utilitarian welfare (optional)
        
    Returns:
        Dictionary of bootstrap results with statistics and confidence intervals
    """
    print(f"Performing non-parametric cell-based bootstrapping with {num_bootstrap} samples...")
    
  
    all_agents = set()
    for game in all_results:
        if game['agent1'] is not None:
            all_agents.add(game['agent1'])
        if game['agent2'] is not None:
            all_agents.add(game['agent2'])
    
    all_agents = sorted(list(all_agents))
    num_agents = len(all_agents)
    print(f"Identified {num_agents} unique agents")
    
    # Initialize results
    bootstrap_results = {
        'ne_regret': [],               
        'ne_strategy': [],
        'rd_regret': [],              
        'rd_strategy': [],             
        'agent_expected_utility': [],
        'agent_max_utility': [],
        'nash_value': [],              
        'rd_nash_value': [],          
        'bootstrapped_matrices': [],
        'ne_nbs': [],                   
        'agent_expected_nash_welfare': [],
        'agent_expected_nash_welfare_adv': [],
        'agent_expected_util_welfare': [],
        'agent_expected_ef1_freq': [],
        'bootstrapped_nash_welfare_matrices': [],
        'bootstrapped_nash_welfare_adv_matrices': [],
        'bootstrapped_util_welfare_matrices': [],
        'bootstrapped_ef1_freq_matrices': [],
        'agent_expected_normalized_nash_welfare': [],
        'agent_expected_normalized_nash_welfare_adv': [],
        'agent_expected_percent_max_util_welfare': [],
        'agent_avg_normalized_nash_welfare': [],
        'agent_avg_percent_max_util_welfare': [],
        'agent_avg_ef1_freq': [],
        'equilibrium_normalized_nash_welfare': [],
        'equilibrium_normalized_nash_welfare_adv': [],
        'equilibrium_percent_max_util_welfare': [],
        'equilibrium_ef1_freq': []
    }
    
    # cell_data = defaultdict(list) # Old structure
    cell_data_by_source = defaultdict(lambda: defaultdict(list)) # New: agent_pair -> source_matrix -> list_of_outcomes
    
    for game in all_results:
        agent1 = game.get('agent1')
        agent2 = game.get('agent2')
        source_matrix = game.get('source_matrix', 'unknown_source') # Get the source_matrix identifier
        
        if not agent1 or not agent2:
            continue
            
        p1_value = game.get('agent1_value')
        p2_value = game.get('agent2_value')
        nash_welfare = game.get('nash_welfare') 
        nash_welfare_adv = game.get('nash_welfare_adv')
        util_welfare = game.get('utilitarian_welfare') 
        is_ef1 = game.get('is_ef1') # Will be True, False, or None

        if p1_value is not None and p2_value is not None:
            key = tuple(sorted((agent1, agent2)))
            #key = (agent1, agent2)
            outcome_tuple = (p1_value, p2_value, nash_welfare, nash_welfare_adv, util_welfare, is_ef1)
            cell_data_by_source[key][source_matrix].append(outcome_tuple)

    total_interactions = 0
    total_agent_pairs = len(cell_data_by_source)
    total_sources_references = 0
    for pair_key, sources in cell_data_by_source.items():
        for source_key, outcomes in sources.items():
            total_interactions += len(outcomes)
            total_sources_references +=1
    print(f"Collected {total_interactions} total game interactions across {total_agent_pairs} unique agent pairs, from {total_sources_references} source references.")
    
    def safe_mean_skip_none(data):
        valid_data = [x for x in data if x is not None and not np.isnan(x)]
        return np.mean(valid_data) if valid_data else np.nan

    def mean_ef1_freq(data):
        count_true = 0
        count_false = 0
        for t in data:
            if t[5] == True:
                count_true +=1
            elif t[5] == False: 
                count_false +=1
        if count_true == 0:
            return 0
        checker = count_true / (count_true + count_false)
        return checker * 100

    for b in tqdm(range(num_bootstrap), desc="Bootstrap Samples"):
        performance_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        nash_welfare_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        util_welfare_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        nash_welfare_adv_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        ef1_freq_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        
    
        # for agent_pair_key, game_outcomes in cell_data.items(): # Old loop
        for agent_pair_key, sources_for_pair in cell_data_by_source.items():
            agent1_key_component, agent2_key_component = agent_pair_key 

            # Metrics for this cell, averaged across bootstrapped sources
            p1_source_means = []
            p2_source_means = []
            normalized_nash_welfare_source_means = []
            normalized_nash_welfare_adv_source_means = []
            percent_max_util_welfare_source_means = []
            ef1_freq_source_means = []

            if not sources_for_pair: 
                continue

            for source_id, game_outcomes_for_source in sources_for_pair.items():
                if not game_outcomes_for_source:
                    continue

                resample_indices = np.random.choice(range(len(game_outcomes_for_source)), size=len(game_outcomes_for_source), replace=True)
                resampled_tuples_from_source = [game_outcomes_for_source[i] for i in resample_indices]

                avg_perf_p1_for_source = safe_mean_skip_none([t[0] for t in resampled_tuples_from_source]) 
                avg_perf_p2_for_source = safe_mean_skip_none([t[1] for t in resampled_tuples_from_source]) 
                
                normalized_nash_welfares_for_source = []
                if global_max_nash_welfare is not None and global_max_nash_welfare > 1e-9:
                    normalized_nash_welfares_for_source = [ 
                        t[2] / global_max_nash_welfare 
                        for t in resampled_tuples_from_source 
                        if t[2] is not None and not np.isnan(t[2])
                    ]
                avg_normalized_nash_welfare_for_source = np.mean(normalized_nash_welfares_for_source) if normalized_nash_welfares_for_source else np.nan

                normalized_nash_welfares_adv_for_source = []

                if global_max_nash_welfare_adv is not None and global_max_nash_welfare_adv > 1e-9:
                    normalized_nash_welfares_adv_for_source = [ 
                        t[3] / global_max_nash_welfare_adv 
                        for t in resampled_tuples_from_source 
                        if t[3] is not None and not np.isnan(t[3])
                    ]

                avg_normalized_nash_welfare_adv_for_source = np.mean(normalized_nash_welfares_adv_for_source) if normalized_nash_welfares_adv_for_source else np.nan
                percent_max_util_welfares_for_source = []
                if global_max_util_welfare is not None and global_max_util_welfare > 1e-9:
                    percent_max_util_welfares_for_source = [
                        (t[4] / global_max_util_welfare) * 100 
                        for t in resampled_tuples_from_source 
                        if t[4] is not None and not np.isnan(t[4])
                    ]
                avg_percent_max_util_welfare_for_source = np.mean(percent_max_util_welfares_for_source) if percent_max_util_welfares_for_source else np.nan
                
                avg_ef1_freq_for_source = mean_ef1_freq(resampled_tuples_from_source)

                if not np.isnan(avg_perf_p1_for_source): p1_source_means.append(avg_perf_p1_for_source)
                if not np.isnan(avg_perf_p2_for_source): p2_source_means.append(avg_perf_p2_for_source)
                if not np.isnan(avg_normalized_nash_welfare_adv_for_source): normalized_nash_welfare_adv_source_means.append(avg_normalized_nash_welfare_adv_for_source)
                if not np.isnan(avg_normalized_nash_welfare_for_source): normalized_nash_welfare_source_means.append(avg_normalized_nash_welfare_for_source)
                if not np.isnan(avg_percent_max_util_welfare_for_source): percent_max_util_welfare_source_means.append(avg_percent_max_util_welfare_for_source)
                if not np.isnan(avg_ef1_freq_for_source): ef1_freq_source_means.append(avg_ef1_freq_for_source)
            
            final_avg_perf_p1 = np.mean(p1_source_means) if p1_source_means else np.nan
            final_avg_perf_p2 = np.mean(p2_source_means) if p2_source_means else np.nan
            final_avg_normalized_nash_welfare = np.mean(normalized_nash_welfare_source_means) if normalized_nash_welfare_source_means else np.nan
            final_avg_normalized_nash_welfare_adv = np.mean(normalized_nash_welfare_adv_source_means) if normalized_nash_welfare_adv_source_means else np.nan
            final_avg_percent_max_util_welfare = np.mean(percent_max_util_welfare_source_means) if percent_max_util_welfare_source_means else np.nan
            final_avg_ef1_freq = np.mean(ef1_freq_source_means) if ef1_freq_source_means else np.nan

            if agent1_key_component == all_agents[0] and agent2_key_component == all_agents[1] and b < 2: # Print for first pair, first 2 iterations
                 print(f"  DEBUG Cell ({agent1_key_component}, {agent2_key_component}), Iter {b} (after source averaging):")
                 print(f"    Num sources for pair: {len(sources_for_pair)}")
                 print(f"    P1 source means: {p1_source_means}")
                 print(f"    Final Avg P1 Perf: {final_avg_perf_p1}")
                 print(f"    Norm Nash Welfare source means: {normalized_nash_welfare_source_means}")
                 print(f"    Norm Nash Welfare Adv source means: {normalized_nash_welfare_adv_source_means}")
                 print(f"    Final Avg Norm Nash Welfare: {final_avg_normalized_nash_welfare}")

            if agent1_key_component == agent_pair_key[0]: # agent1_key_component was p1 in the original game storage relative to key
                 performance_matrix.loc[agent1_key_component, agent2_key_component] = final_avg_perf_p1
                 performance_matrix.loc[agent2_key_component, agent1_key_component] = final_avg_perf_p2
            else: 
                 performance_matrix.loc[agent2_key_component, agent1_key_component] = final_avg_perf_p1 
                 performance_matrix.loc[agent1_key_component, agent2_key_component] = final_avg_perf_p2

            nash_welfare_matrix.loc[agent1_key_component, agent2_key_component] = final_avg_normalized_nash_welfare 
            nash_welfare_matrix.loc[agent2_key_component, agent1_key_component] = final_avg_normalized_nash_welfare 

            nash_welfare_adv_matrix.loc[agent1_key_component, agent2_key_component] = final_avg_normalized_nash_welfare_adv
            nash_welfare_adv_matrix.loc[agent2_key_component, agent1_key_component] = final_avg_normalized_nash_welfare_adv


            util_welfare_matrix.loc[agent1_key_component, agent2_key_component] = final_avg_percent_max_util_welfare 
            util_welfare_matrix.loc[agent2_key_component, agent1_key_component] = final_avg_percent_max_util_welfare 

            ef1_freq_matrix.loc[agent1_key_component, agent2_key_component] = final_avg_ef1_freq
            ef1_freq_matrix.loc[agent2_key_component, agent1_key_component] = final_avg_ef1_freq 

        bootstrap_results['bootstrapped_matrices'].append(performance_matrix.copy())
        
        try: #TODO remove imputation. 
            game_matrix_np = performance_matrix.to_numpy(dtype=float) # Ensure float
            
            for i in range(game_matrix_np.shape[0]):
                for j in range(game_matrix_np.shape[1]):
                    if np.isnan(game_matrix_np[i, j]):
                        col_mean = np.nanmean(game_matrix_np[:, j])
                        if not np.isnan(col_mean):
                            game_matrix_np[i, j] = col_mean
                        else:
                            row_mean = np.nanmean(game_matrix_np[i, :])
                            game_matrix_np[i, j] = row_mean if not np.isnan(row_mean) else 0
            
            nash_strategy = milp_max_sym_ent_2p(game_matrix_np, 2000)
            
            nash_welfare_matrix_np = nash_welfare_matrix.to_numpy(dtype=float)
            nash_welfare_adv_matrix_np = nash_welfare_adv_matrix.to_numpy(dtype=float)
            util_welfare_matrix_np = util_welfare_matrix.to_numpy(dtype=float)
            ef1_freq_matrix_np = ef1_freq_matrix.to_numpy(dtype=float) 
            #TODO repetive prob should move to utils 
            for mat in [nash_welfare_matrix_np, util_welfare_matrix_np, ef1_freq_matrix_np, nash_welfare_adv_matrix_np]:
                 for i in range(mat.shape[0]):
                     for j in range(mat.shape[1]):
                         if np.isnan(mat[i, j]):
                             col_mean = np.nanmean(mat[:, j])
                             if not np.isnan(col_mean):
                                 mat[i, j] = col_mean
                             else:
                                 row_mean = np.nanmean(mat[i, :])
                                 mat[i, j] = row_mean if not np.isnan(row_mean) else 0

    

            if global_max_nash_welfare is not None and global_max_nash_welfare > 1e-6:
                 #nash_welfare_matrix_np = nash_welfare_matrix_np / global_max_nash_welfare
                 print(f"  DEBUG (Bootstrap iter {b}): Normalized Nash welfare matrix by {global_max_nash_welfare:.2f}")
            else:
                 print(f"  DEBUG (Bootstrap iter {b}): Skipping Nash welfare normalization (max value: {global_max_nash_welfare})")
                 
            if global_max_util_welfare is not None and global_max_util_welfare > 1e-6:
                 #util_welfare_matrix_np = (util_welfare_matrix_np / global_max_util_welfare) * 100 
                 print(f"  DEBUG (Bootstrap iter {b}): Normalized Utilitarian welfare matrix by {global_max_util_welfare:.2f} (to %)")
            else:
                 print(f"  DEBUG (Bootstrap iter {b}): Skipping Utilitarian welfare normalization (max value: {global_max_util_welfare})")
            
            if global_max_nash_welfare_adv is not None and global_max_nash_welfare_adv > 1e-6:
                 #nash_welfare_matrix_np = nash_welfare_matrix_np / global_max_nash_welfare
                 print(f"  DEBUG (Bootstrap iter {b}): Normalized Nash welfare adv matrix by {global_max_nash_welfare_adv:.2f}")
            else:
                 print(f"  DEBUG (Bootstrap iter {b}): Skipping Nash welfare adv normalization (max value: {global_max_nash_welfare_adv})")

            

            expected_normalized_nash_welfare = np.dot(nash_welfare_matrix_np, nash_strategy)

            expected_normalized_nash_welfare = expected_normalized_nash_welfare * 100

            expected_normalized_nash_welfare_adv = np.dot(nash_welfare_adv_matrix_np, nash_strategy)

            expected_normalized_nash_welfare_adv = expected_normalized_nash_welfare_adv * 100


            
            expected_percent_max_util_welfare = np.dot(util_welfare_matrix_np, nash_strategy)

            expected_percent_max_util_welfare = expected_percent_max_util_welfare
            
            expected_ef1_freq = np.dot(ef1_freq_matrix_np, nash_strategy) 
            eq_norm_nash_welfare = float(nash_strategy.reshape((1, -1)) @ nash_welfare_matrix_np @ nash_strategy.reshape((-1, 1)))
            eq_norm_nash_welfare = eq_norm_nash_welfare * 100.0  # match per-agent scaling
            eq_norm_nash_welfare_adv = float(nash_strategy.reshape((1, -1)) @ nash_welfare_adv_matrix_np @ nash_strategy.reshape((-1, 1)))
            eq_norm_nash_welfare_adv = eq_norm_nash_welfare_adv * 100.0


            eq_percent_max_util_welfare = float(nash_strategy.reshape((1, -1)) @ util_welfare_matrix_np @ nash_strategy.reshape((-1, 1)))
            eq_ef1_freq = float(nash_strategy.reshape((1, -1)) @ ef1_freq_matrix_np @ nash_strategy.reshape((-1, 1)))
            

            



            print(f"DEBUG (Bootstrap iter {b}): Expected Norm Nash Welfare range: [{np.nanmin(expected_normalized_nash_welfare):.4e}, {np.nanmax(expected_normalized_nash_welfare):.4e}]")
            print(f"DEBUG (Bootstrap iter {b}): Expected Norm Nash Welfare Adv range: [{np.nanmin(expected_normalized_nash_welfare_adv):.4e}, {np.nanmax(expected_normalized_nash_welfare_adv):.4e}]")

            print(f"DEBUG (Bootstrap iter {b}): Expected % Util Welfare range: [{np.nanmin(expected_percent_max_util_welfare):.4f}%, {np.nanmax(expected_percent_max_util_welfare):.4f}%]")

            # rd_strategy, _ = replicator_dynamics_nash(game_matrix_np, 1)
            expected_utils = np.dot(game_matrix_np, nash_strategy)
            # rd_expected_utils = np.dot(game_matrix_np, rd_strategy)
            nash_value = nash_strategy.reshape((1, -1)) @ game_matrix_np @ nash_strategy.reshape((-1, 1))
            nash_value = nash_value.item()
            # rd_nash_value = rd_strategy.reshape((1, -1)) @ game_matrix_np @ rd_strategy.reshape((-1, 1))
            # rd_nash_value = rd_nash_value.item()
            max_utils = np.max(game_matrix_np, axis=1) 
            nash_regrets = nash_value - expected_utils
            # rd_regrets = rd_nash_value - rd_expected_utils
            
            ne_nbs_scores = np.zeros(len(all_agents))
            for i in range(len(all_agents)):
                u_agent_vs_nash = expected_utils[i]
                u_nash_vs_agent = 0
                for j in range(len(all_agents)):
                    u_nash_vs_agent += nash_strategy[j] * game_matrix_np[j, i]
                ne_nbs_scores[i] = u_agent_vs_nash * u_nash_vs_agent
            
            bootstrap_results['ne_regret'].append(nash_regrets)
            bootstrap_results['ne_strategy'].append(nash_strategy)
            bootstrap_results['agent_expected_utility'].append(expected_utils) # Perf vs NE 
            bootstrap_results['agent_max_utility'].append(max_utils)
            bootstrap_results['nash_value'].append(nash_value)
            bootstrap_results['ne_nbs'].append(ne_nbs_scores)
            bootstrap_results['agent_expected_normalized_nash_welfare'].append(expected_normalized_nash_welfare)
            bootstrap_results['agent_expected_normalized_nash_welfare_adv'].append(expected_normalized_nash_welfare_adv)
            bootstrap_results['agent_expected_percent_max_util_welfare'].append(expected_percent_max_util_welfare)
            bootstrap_results['agent_expected_ef1_freq'].append(expected_ef1_freq)

            bootstrap_results['equilibrium_normalized_nash_welfare_adv'].append(eq_norm_nash_welfare_adv)
            bootstrap_results['equilibrium_normalized_nash_welfare_adv'].append(eq_norm_nash_welfare_adv)


            bootstrap_results['equilibrium_percent_max_util_welfare'].append(eq_percent_max_util_welfare)
            bootstrap_results['equilibrium_ef1_freq'].append(eq_ef1_freq)

            #add in equilibrium values

            if ('equilibrium_normalized_nash_welfare' in bootstrap_results and bootstrap_results['equilibrium_normalized_nash_welfare']) or \
            ('equilibrium_percent_max_util_welfare' in bootstrap_results and bootstrap_results['equilibrium_percent_max_util_welfare']) or \
            ('equilibrium_ef1_freq' in bootstrap_results and bootstrap_results['equilibrium_ef1_freq']):

                def _scalar_stats(arr_like):
                    arr = np.asarray(arr_like, dtype=float)
                    if arr.size == 0:
                        return np.nan, 0.0, np.nan, np.nan
                    mean = float(np.nanmean(arr))
                    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
                    lo = float(np.percentile(arr, lower_percentile))
                    hi = float(np.percentile(arr, upper_percentile))
                    return mean, std, lo, hi

                eq_row = {'Agent': 'NE_mix'}

                if bootstrap_results.get('equilibrium_normalized_nash_welfare'):
                    m, s, lo, hi = _scalar_stats(bootstrap_results['equilibrium_normalized_nash_welfare'])
                    eq_row['Mean Expected Norm Nash Welfare'] = m
                    eq_row['Std Expected Norm Nash Welfare'] = s
                    eq_row[f'Lower {confidence*100:.0f}% CI (Exp Norm Nash Welf)'] = lo
                    eq_row[f'Upper {confidence*100:.0f}% CI (Exp Norm Nash Welf)'] = hi
                
                if bootstrap_results.get('equilibrium_normalized_nash_welfare_adv'):
                    m, s, lo, hi = _scalar_stats(bootstrap_results['equilibrium_normalized_nash_welfare_adv'])
                    eq_row['Mean Expected Norm Nash Welfare Adv'] = m
                    eq_row['Std Expected Norm Nash Welfare Adv'] = s
                    eq_row[f'Lower {confidence*100:.0f}% CI (Exp Norm Nash Welf Adv)'] = lo
                    eq_row[f'Upper {confidence*100:.0f}% CI (Exp Norm Nash Welf Adv)'] = hi

                

                if bootstrap_results.get('equilibrium_percent_max_util_welfare'):
                    m, s, lo, hi = _scalar_stats(bootstrap_results['equilibrium_percent_max_util_welfare'])
                    eq_row['Mean Expected % Max Util Welfare'] = m
                    eq_row['Std Expected % Max Util Welfare'] = s
                    eq_row[f'Lower {confidence*100:.0f}% CI (Exp % Max Util Welf)'] = lo
                    eq_row[f'Upper {confidence*100:.0f}% CI (Exp % Max Util Welf)'] = hi

                if bootstrap_results.get('equilibrium_ef1_freq'):
                    m, s, lo, hi = _scalar_stats(bootstrap_results['equilibrium_ef1_freq'])
                    eq_row['Mean Expected EF1 Freq (%)'] = m
                    eq_row['Std Expected EF1 Freq (%)'] = s
                    eq_row[f'Lower {confidence*100:.0f}% CI (Exp EF1 Freq %)'] = lo
                    eq_row[f'Upper {confidence*100:.0f}% CI (Exp EF1 Freq %)'] = hi

                results = pd.concat([results, pd.DataFrame([eq_row])], ignore_index=True)

            
       

            if b < 1: 
                print(f"  DEBUG (Iter {b}) Imputed nash_welfare_matrix_np (sample, shape={nash_welfare_matrix_np.shape}):")
                # Print top-left corner, ensuring indices don't exceed bounds
                print_rows = min(5, nash_welfare_matrix_np.shape[0])
                print_cols = min(5, nash_welfare_matrix_np.shape[1])
                print(nash_welfare_matrix_np[:print_rows, :print_cols])
            # <<< End Debug Print >>>

        except Exception as e:
            print(f"Error in bootstrap sample {b}: {e}")
            continue
    
    bootstrap_stats = analyze_bootstrap_results(bootstrap_results, all_agents, confidence)
    bootstrap_results['statistics'] = bootstrap_stats
    
    return bootstrap_results, all_agents

def analyze_bootstrap_results(bootstrap_results, agent_names, confidence=0.95):
    """
    Analyze bootstrap results and compute confidence intervals for Nash equilibrium regrets
    
    Args:
        bootstrap_results: Dictionary with bootstrap samples
        agent_names: List of agent names
        confidence: Confidence level (default: 0.95)
        
    Returns:
        DataFrame with statistics and confidence intervals
    """
    if not bootstrap_results['ne_regret']:
        print("No bootstrap results to analyze. Check for errors in the bootstrap process.")
        return pd.DataFrame({
            'Agent': agent_names,
            'Mean NE Regret': [np.nan] * len(agent_names),
            'Std NE Regret': [np.nan] * len(agent_names),
            f'Lower {confidence*100:.0f}% CI (NE Regret)': [np.nan] * len(agent_names),
            f'Upper {confidence*100:.0f}% CI (NE Regret)': [np.nan] * len(agent_names),
            'Mean Expected Utility': [np.nan] * len(agent_names),
            'Std Expected Utility': [np.nan] * len(agent_names)
        })
    
    try:
        ne_regrets = np.stack(bootstrap_results['ne_regret'])
        expected_utils = np.stack(bootstrap_results['agent_expected_utility'])
        
        has_ne_nbs = 'ne_nbs' in bootstrap_results and bootstrap_results['ne_nbs']
        if has_ne_nbs:
            ne_nbs_scores = np.stack(bootstrap_results['ne_nbs'])
        
        has_me_normal_regrets = 'me_normal_regret' in bootstrap_results and bootstrap_results['me_normal_regret']
        has_rd_normal_regrets = 'rd_normal_regret' in bootstrap_results and bootstrap_results['rd_normal_regret']
        
        if has_me_normal_regrets:
            me_normal_regrets = np.stack(bootstrap_results['me_normal_regret'])
        
        if has_rd_normal_regrets:
            rd_normal_regrets = np.stack(bootstrap_results['rd_normal_regret'])

        # --- Add extraction for new metrics (using AVG keys) ---
        has_nash_welfare = 'agent_avg_normalized_nash_welfare' in bootstrap_results and bootstrap_results['agent_avg_normalized_nash_welfare']
        if has_nash_welfare:
            nash_welfares = np.stack(bootstrap_results['agent_avg_normalized_nash_welfare'])

        has_nash_welfare_adv = 'agent_avg_normalized_nash_welfare_adv' in bootstrap_results and bootstrap_results['agent_avg_normalized_nash_welfare_adv']
        if has_nash_welfare_adv:
            nash_welfares_adv = np.stack(bootstrap_results['agent_avg_normalized_nash_welfare_adv'])
        
        has_util_welfare = 'agent_avg_percent_max_util_welfare' in bootstrap_results and bootstrap_results['agent_avg_percent_max_util_welfare']
        if has_util_welfare:
            util_welfares = np.stack(bootstrap_results['agent_avg_percent_max_util_welfare'])
            
        has_ef1_freq = 'agent_avg_ef1_freq' in bootstrap_results and bootstrap_results['agent_avg_ef1_freq']
        if has_ef1_freq:
            ef1_freqs = np.stack(bootstrap_results['agent_avg_ef1_freq'])

        has_nash_welfare = 'agent_expected_normalized_nash_welfare' in bootstrap_results and bootstrap_results['agent_expected_normalized_nash_welfare']
        if has_nash_welfare:
            nash_welfares = np.stack(bootstrap_results['agent_expected_normalized_nash_welfare'])
        
        has_util_welfare = 'agent_expected_percent_max_util_welfare' in bootstrap_results and bootstrap_results['agent_expected_percent_max_util_welfare']
        if has_util_welfare:
            util_welfares = np.stack(bootstrap_results['agent_expected_percent_max_util_welfare'])
            
        has_ef1_freq = 'agent_expected_ef1_freq' in bootstrap_results and bootstrap_results['agent_expected_ef1_freq']
        if has_ef1_freq:
            ef1_freqs = np.stack(bootstrap_results['agent_expected_ef1_freq'])

    except ValueError:
        print("Warning: Bootstrap samples have inconsistent shapes. Using a more flexible approach.")
        
        first_regret = bootstrap_results['ne_regret'][0]
        first_util = bootstrap_results['agent_expected_utility'][0]
        
        num_samples = len(bootstrap_results['ne_regret'])
        ne_regrets = np.zeros((num_samples, len(first_regret)), dtype=np.float64)
        expected_utils = np.zeros((num_samples, len(first_util)), dtype=np.float64)
        
        for i in range(num_samples):
            if i < len(bootstrap_results['ne_regret']):
                ne_regrets[i] = bootstrap_results['ne_regret'][i]
            if i < len(bootstrap_results['agent_expected_utility']):
                expected_utils[i] = bootstrap_results['agent_expected_utility'][i]
        
        has_ne_nbs = 'ne_nbs' in bootstrap_results and bootstrap_results['ne_nbs']
        if has_ne_nbs:
            first_ne_nbs = bootstrap_results['ne_nbs'][0]
            ne_nbs_scores = np.zeros((num_samples, len(first_ne_nbs)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['ne_nbs']):
                    ne_nbs_scores[i] = bootstrap_results['ne_nbs'][i]
        
        has_me_normal_regrets = 'me_normal_regret' in bootstrap_results and bootstrap_results['me_normal_regret']
        if has_me_normal_regrets:
            first_me_normal = bootstrap_results['me_normal_regret'][0]
            me_normal_regrets = np.zeros((num_samples, len(first_me_normal)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['me_normal_regret']):
                    me_normal_regrets[i] = bootstrap_results['me_normal_regret'][i]
        
        if has_rd_normal_regrets:
            first_rd_normal = bootstrap_results['rd_normal_regret'][0]
            rd_normal_regrets = np.zeros((num_samples, len(first_rd_normal)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['rd_normal_regret']):
                    rd_normal_regrets[i] = bootstrap_results['rd_normal_regret'][i]
        
        has_nash_welfare = 'agent_avg_normalized_nash_welfare' in bootstrap_results and bootstrap_results['agent_avg_normalized_nash_welfare']
        if has_nash_welfare:
            first_nash_welfare = bootstrap_results['agent_avg_normalized_nash_welfare'][0]
            nash_welfares = np.zeros((num_samples, len(first_nash_welfare)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['agent_avg_normalized_nash_welfare']):
                    nash_welfares[i] = bootstrap_results['agent_avg_normalized_nash_welfare'][i]

        has_nash_welfare_adv = 'agent_avg_normalized_nash_welfare_adv' in bootstrap_results and bootstrap_results['agent_avg_normalized_nash_welfare_adv']
        if has_nash_welfare_adv:
            first_nash_welfare = bootstrap_results['agent_avg_normalized_nash_welfare_adv'][0]
            nash_welfares = np.zeros((num_samples, len(first_nash_welfare)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['agent_avg_normalized_nash_welfare_adv']):
                    nash_welfares[i] = bootstrap_results['agent_avg_normalized_nash_welfare_adv'][i]
        
        has_util_welfare = 'agent_avg_percent_max_util_welfare' in bootstrap_results and bootstrap_results['agent_avg_percent_max_util_welfare']
        if has_util_welfare:
            first_util_welfare = bootstrap_results['agent_avg_percent_max_util_welfare'][0]
            util_welfares = np.zeros((num_samples, len(first_util_welfare)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['agent_avg_percent_max_util_welfare']):
                    util_welfares[i] = bootstrap_results['agent_avg_percent_max_util_welfare'][i]

        has_ef1_freq = 'agent_avg_ef1_freq' in bootstrap_results and bootstrap_results['agent_avg_ef1_freq']
        if has_ef1_freq:
            first_ef1_freq = bootstrap_results['agent_avg_ef1_freq'][0]
            ef1_freqs = np.zeros((num_samples, len(first_ef1_freq)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['agent_avg_ef1_freq']):
                    ef1_freqs[i] = bootstrap_results['agent_avg_ef1_freq'][i]

        has_nash_welfare = 'agent_expected_normalized_nash_welfare' in bootstrap_results and bootstrap_results['agent_expected_normalized_nash_welfare']
        if has_nash_welfare:
            first_nash_welfare = bootstrap_results['agent_expected_normalized_nash_welfare'][0]
            nash_welfares = np.zeros((num_samples, len(first_nash_welfare)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['agent_expected_normalized_nash_welfare']):
                    nash_welfares[i] = bootstrap_results['agent_expected_normalized_nash_welfare'][i]


        has_nash_welfare_adv = 'agent_expected_normalized_nash_welfare_adv' in bootstrap_results and bootstrap_results['agent_expected_normalized_nash_welfare_adv']
        if has_nash_welfare:
            first_nash_welfare = bootstrap_results['agent_expected_normalized_nash_welfare_adv'][0]
            nash_welfares = np.zeros((num_samples, len(first_nash_welfare)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['agent_expected_normalized_nash_welfare_adv']):
                    nash_welfares_adv[i] = bootstrap_results['agent_expected_normalized_nash_welfare_adv'][i]
        
        has_util_welfare = 'agent_expected_percent_max_util_welfare' in bootstrap_results and bootstrap_results['agent_expected_percent_max_util_welfare']
        if has_util_welfare:
            first_util_welfare = bootstrap_results['agent_expected_percent_max_util_welfare'][0]
            util_welfares = np.zeros((num_samples, len(first_util_welfare)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['agent_expected_percent_max_util_welfare']):
                    util_welfares[i] = bootstrap_results['agent_expected_percent_max_util_welfare'][i]

        has_ef1_freq = 'agent_expected_ef1_freq' in bootstrap_results and bootstrap_results['agent_expected_ef1_freq']
        if has_ef1_freq:
            first_ef1_freq = bootstrap_results['agent_expected_ef1_freq'][0]
            ef1_freqs = np.zeros((num_samples, len(first_ef1_freq)), dtype=np.float64)
            for i in range(num_samples):
                if i < len(bootstrap_results['agent_expected_ef1_freq']):
                    ef1_freqs[i] = bootstrap_results['agent_expected_ef1_freq'][i]

    mean_ne_regrets = np.mean(ne_regrets, axis=0)
    mean_expected_utils = np.mean(expected_utils, axis=0)
    if has_ne_nbs:
        mean_ne_nbs_scores = np.mean(ne_nbs_scores, axis=0)
    if has_me_normal_regrets:
        mean_me_normal_regrets = np.mean(me_normal_regrets, axis=0)
    if has_rd_normal_regrets:
        mean_rd_normal_regrets = np.mean(rd_normal_regrets, axis=0)
    
    if has_nash_welfare:
        mean_nash_welfares = np.mean(nash_welfares, axis=0)
    if has_nash_welfare_adv:
        mean_nash_welfares_adv = np.mean(nash_welfares_adv, axis=0)
    if has_util_welfare:
        mean_util_welfares = np.mean(util_welfares, axis=0)
    if has_ef1_freq:
        mean_ef1_freqs = np.mean(ef1_freqs, axis=0)

    std_ne_regrets = np.power(np.mean((ne_regrets - mean_ne_regrets) ** 2, axis=0), 0.5)
    std_expected_utils = np.power(np.mean((expected_utils - mean_expected_utils) ** 2, axis=0), 0.5)
    if has_ne_nbs:
        std_ne_nbs_scores = np.power(np.mean((ne_nbs_scores - mean_ne_nbs_scores) ** 2, axis=0), 0.5)
    if has_me_normal_regrets:
        std_me_normal_regrets = np.power(np.mean((me_normal_regrets - mean_me_normal_regrets) ** 2, axis=0), 0.5)
    if has_rd_normal_regrets:
        std_rd_normal_regrets = np.power(np.mean((rd_normal_regrets - mean_rd_normal_regrets) ** 2, axis=0), 0.5)
    
    # --- Calculate stds for new metrics ---
    if has_nash_welfare:
        std_nash_welfares = np.power(np.mean((nash_welfares - mean_nash_welfares) ** 2, axis=0), 0.5)

    if has_nash_welfare_adv:
        std_nash_welfares_adv = np.power(np.mean((nash_welfares_adv - mean_nash_welfares_adv) ** 2, axis=0), 0.5)
    if has_util_welfare:
        std_util_welfares = np.power(np.mean((util_welfares - mean_util_welfares) ** 2, axis=0), 0.5)
    if has_ef1_freq:
        std_ef1_freqs = np.power(np.mean((ef1_freqs - mean_ef1_freqs) ** 2, axis=0), 0.5)
    # --- End stds ---

    epsilon = 1e-6  # More forgiving for numerical precision issues
    '''
    if np.any(mean_ne_regrets > epsilon):
        max_ne_regret = np.max(mean_ne_regrets)
        worst_idx_ne = np.argmax(mean_ne_regrets)
        worst_agent_ne = agent_names[worst_idx_ne]
        
        error_msg = [f"Max Entropy NE: {max_ne_regret:.10f} for agent {worst_agent_ne}"]
        
        if has_me_normal_regrets and np.any(mean_me_normal_regrets > epsilon):
            max_me_normal = np.max(mean_me_normal_regrets)
            worst_idx_me_normal = np.argmax(mean_me_normal_regrets)
            worst_agent_me_normal = agent_names[worst_idx_me_normal]
            error_msg.append(f"ME Normal: {max_me_normal:.10f} for agent {worst_agent_me_normal}")
        
        if has_rd_normal_regrets and np.any(mean_rd_normal_regrets > epsilon):
            max_rd_normal = np.max(mean_rd_normal_regrets)
            worst_idx_rd_normal = np.argmax(mean_rd_normal_regrets)
            worst_agent_rd_normal = agent_names[worst_idx_rd_normal]
            error_msg.append(f"RD Normal: {max_rd_normal:.10f} for agent {worst_agent_rd_normal}")
            
        print(f"NOTE: Large positive mean regret detected:\n{', '.join(error_msg)}")
        print("Continuing analysis with positive regrets. Results may indicate non-equilibrium strategies.")
    elif np.any(mean_ne_regrets > 0) or (has_me_normal_regrets and np.any(mean_me_normal_regrets > 0)) or \
         (has_rd_normal_regrets and np.any(mean_rd_normal_regrets > 0)):
        # For small positive regrets, just warn
        if np.any(mean_ne_regrets > 0):
            max_ne_regret = np.max(mean_ne_regrets)
            worst_idx_ne = np.argmax(mean_ne_regrets)
            worst_agent_ne = agent_names[worst_idx_ne]
            print(f"NOTE: Small positive mean ME Nash regret detected: {max_ne_regret:.10f} for agent {worst_agent_ne}")
        
        if has_me_normal_regrets and np.any(mean_me_normal_regrets > 0):
            max_me_normal = np.max(mean_me_normal_regrets)
            worst_idx_me_normal = np.argmax(mean_me_normal_regrets)
            worst_agent_me_normal = agent_names[worst_idx_me_normal]
            print(f"NOTE: Small positive mean ME normal regret detected: {max_me_normal:.10f} for agent {worst_agent_me_normal}")
            
        if has_rd_normal_regrets and np.any(mean_rd_normal_regrets > 0):
            max_rd_normal = np.max(mean_rd_normal_regrets)
            worst_idx_rd_normal = np.argmax(mean_rd_normal_regrets)
            worst_agent_rd_normal = agent_names[worst_idx_rd_normal]
            print(f"NOTE: Small positive mean RD normal regret detected: {max_rd_normal:.10f} for agent {worst_agent_rd_normal}")
    '''
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ne_regrets = np.percentile(ne_regrets, lower_percentile, axis=0)
    upper_ne_regrets = np.percentile(ne_regrets, upper_percentile, axis=0)
    
    if has_ne_nbs:
        lower_ne_nbs = np.percentile(ne_nbs_scores, lower_percentile, axis=0)
        upper_ne_nbs = np.percentile(ne_nbs_scores, upper_percentile, axis=0)
    
    if has_me_normal_regrets:
        lower_me_normal_regrets = np.percentile(me_normal_regrets, lower_percentile, axis=0)
        upper_me_normal_regrets = np.percentile(me_normal_regrets, upper_percentile, axis=0)
        
    if has_rd_normal_regrets:
        lower_rd_normal_regrets = np.percentile(rd_normal_regrets, lower_percentile, axis=0)
        upper_rd_normal_regrets = np.percentile(rd_normal_regrets, upper_percentile, axis=0)
    
    if has_nash_welfare:
        lower_nash_welfares = np.percentile(nash_welfares, lower_percentile, axis=0)
        upper_nash_welfares = np.percentile(nash_welfares, upper_percentile, axis=0)

    if has_nash_welfare_adv:
        lower_nash_welfares_adv = np.percentile(nash_welfares_adv, lower_percentile, axis=0)
        upper_nash_welfares_adv = np.percentile(nash_welfares_adv, upper_percentile, axis=0)
    if has_util_welfare:
        lower_util_welfares = np.percentile(util_welfares, lower_percentile, axis=0)
        upper_util_welfares = np.percentile(util_welfares, upper_percentile, axis=0)
    if has_ef1_freq:
        lower_ef1_freqs = np.percentile(ef1_freqs, lower_percentile, axis=0)
        upper_ef1_freqs = np.percentile(ef1_freqs, upper_percentile, axis=0)

    if np.any(upper_ne_regrets > epsilon):
        warnings.warn("Some confidence intervals for NE regrets include positive values. "
                     "This may indicate numerical instability in the Nash equilibrium calculation.")
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Agent': agent_names,
        'Mean NE Regret': mean_ne_regrets,
        'Std NE Regret': std_ne_regrets,
        f'Lower {confidence*100:.0f}% CI (NE Regret)': lower_ne_regrets,
        f'Upper {confidence*100:.0f}% CI (NE Regret)': upper_ne_regrets,
        'Mean Expected Utility': mean_expected_utils,
        'Std Expected Utility': std_expected_utils
    })
    
    if has_ne_nbs:
        results['Mean NE-NBS'] = mean_ne_nbs_scores
        results['Std NE-NBS'] = std_ne_nbs_scores
        results[f'Lower {confidence*100:.0f}% CI (NE-NBS)'] = lower_ne_nbs
        results[f'Upper {confidence*100:.0f}% CI (NE-NBS)'] = upper_ne_nbs
    
    if has_me_normal_regrets:
        results['Mean ME Normal Regret'] = mean_me_normal_regrets
        results['Std ME Normal Regret'] = std_me_normal_regrets
        results[f'Lower {confidence*100:.0f}% CI (ME Normal Regret)'] = lower_me_normal_regrets
        results[f'Upper {confidence*100:.0f}% CI (ME Normal Regret)'] = upper_me_normal_regrets
    
    if has_rd_normal_regrets:
        results['Mean RD Normal Regret'] = mean_rd_normal_regrets
        results['Std RD Normal Regret'] = std_rd_normal_regrets
        results[f'Lower {confidence*100:.0f}% CI (RD Normal Regret)'] = lower_rd_normal_regrets
        results[f'Upper {confidence*100:.0f}% CI (RD Normal Regret)'] = upper_rd_normal_regrets
    
    if has_nash_welfare:
        results['Mean Normalized Nash Welfare vs NE'] = mean_nash_welfares
        results['Std Normalized Nash Welfare vs NE'] = std_nash_welfares
        results[f'Lower {confidence*100:.0f}% CI (Norm. Nash Welfare vs NE)'] = lower_nash_welfares
        results[f'Upper {confidence*100:.0f}% CI (Norm. Nash Welfare vs NE)'] = upper_nash_welfares
    if has_nash_welfare_adv:
        results['Mean Normalized Nash Welfare Adv vs NE'] = mean_nash_welfares_adv
        results['Std Normalized Nash Welfare Adv vs NE'] = std_nash_welfares_adv
        results[f'Lower {confidence*100:.0f}% CI (Norm. Nash Welfare Adv vs NE)'] = lower_nash_welfares_adv
        results[f'Upper {confidence*100:.0f}% CI (Norm. Nash Welfare Adv vs NE)'] = upper_nash_welfares_adv
        
    if has_util_welfare:
        results['Mean % Max Util Welfare vs NE'] = mean_util_welfares
        results['Std % Max Util Welfare vs NE'] = std_util_welfares
        results[f'Lower {confidence*100:.0f}% CI (% Max Util Welfare vs NE)'] = lower_util_welfares
        results[f'Upper {confidence*100:.0f}% CI (% Max Util Welfare vs NE)'] = upper_util_welfares
        
    if has_ef1_freq:
        results['Mean EF1 Freq vs NE (%)'] = mean_ef1_freqs
        results['Std EF1 Freq vs NE (%)'] = std_ef1_freqs
        results[f'Lower {confidence*100:.0f}% CI (EF1 Freq vs NE %)'] = lower_ef1_freqs
        results[f'Upper {confidence*100:.0f}% CI (EF1 Freq vs NE %)'] = upper_ef1_freqs
    
    results = results.sort_values(by='Mean Expected Utility', ascending=False)
    
    print("\nSummary of Nash Equilibrium Analysis:")
    print(f"Average NE regret: {np.mean(mean_ne_regrets):.8f}")
    if has_ne_nbs:
        print(f"Average NE-NBS: {np.mean(mean_ne_nbs_scores):.8f}")
    if has_me_normal_regrets:
        print(f"Average ME normal regret: {np.mean(mean_me_normal_regrets):.8f}")
    if has_rd_normal_regrets:
        print(f"Average RD normal regret: {np.mean(mean_rd_normal_regrets):.8f}")
    
    # --- Add print summary for new metrics ---
    if has_nash_welfare:
        print(f"Average Normalized Nash Welfare vs NE: {np.mean(mean_nash_welfares):.8f}")
    if has_nash_welfare_adv:
        print(f"Average Normalized Nash Welfare Adv vs NE: {np.mean(mean_nash_welfares_adv):.8f}")
    if has_util_welfare:
        print(f"Average % Max Utilitarian Welfare vs NE: {np.mean(mean_util_welfares):.8f}")
    if has_ef1_freq:
        print(f"Average EF1 Frequency vs NE (%): {np.mean(mean_ef1_freqs):.8f}")
    # --- End print summary ---

    print(f"Maximum NE regret: {np.max(mean_ne_regrets):.8f}")
    if has_ne_nbs:
        print(f"Maximum NE-NBS: {np.max(mean_ne_nbs_scores):.8f}")
    if has_me_normal_regrets:
        print(f"Maximum ME normal regret: {np.max(mean_me_normal_regrets):.8f}")
    if has_rd_normal_regrets:
        print(f"Maximum RD normal regret: {np.max(mean_rd_normal_regrets):.8f}")
    
    # --- Add max print summary for new metrics ---
    if has_nash_welfare:
        print(f"Maximum Normalized Nash Welfare vs NE: {np.max(mean_nash_welfares):.8f}")

    if has_nash_welfare_adv:
        print(f"Maximum Normalized Nash Welfare vs NE: {np.max(mean_nash_welfares_adv):.8f}")
    if has_util_welfare:
        print(f"Maximum % Max Utilitarian Welfare vs NE: {np.max(mean_util_welfares):.8f}")
    if has_ef1_freq:
        print(f"Maximum EF1 Frequency vs NE (%): {np.max(mean_ef1_freqs):.8f}")
    # --- End max print summary ---

    print("\nTop 5 agents by Expected Utility:")
    print(results[['Agent', 'Mean Expected Utility', 'Mean NE Regret']].head(5))
    
    if has_ne_nbs:
        print("\nTop 5 agents by NE-NBS (higher is better):")
        print(results.sort_values('Mean NE-NBS', ascending=False)[['Agent', 'Mean NE-NBS', 'Mean NE Regret']].head(5))
    
    # --- Add new metric columns to DataFrame (using AVG keys/labels) ---
    if has_nash_welfare:
        results['Mean Avg Norm Nash Welfare'] = mean_nash_welfares # Updated label
        results['Std Avg Norm Nash Welfare'] = std_nash_welfares  # Updated label
        results[f'Lower {confidence*100:.0f}% CI (Avg Norm Nash Welf)'] = lower_nash_welfares # Updated label
        results[f'Upper {confidence*100:.0f}% CI (Avg Norm Nash Welf)'] = upper_nash_welfares # Updated label
    
    if has_nash_welfare_adv:
        results['Mean Avg Norm Nash Welfare Adv'] = mean_nash_welfares_adv # Updated label
        results['Std Avg Norm Nash Welfare Adv'] = std_nash_welfares_adv  # Updated label
        results[f'Lower {confidence*100:.0f}% CI (Avg Norm Nash Welf Adv)'] = lower_nash_welfares_adv # Updated label
        results[f'Upper {confidence*100:.0f}% CI (Avg Norm Nash Welf Adv)'] = upper_nash_welfares_adv # Updated label
        
    if has_util_welfare:
        results['Mean Avg % Max Util Welfare'] = mean_util_welfares # Updated label
        results['Std Avg % Max Util Welfare'] = std_util_welfares # Updated label
        results[f'Lower {confidence*100:.0f}% CI (Avg % Max Util Welf)'] = lower_util_welfares # Updated label
        results[f'Upper {confidence*100:.0f}% CI (Avg % Max Util Welf)'] = upper_util_welfares # Updated label
        
    if has_ef1_freq:
        results['Mean Avg EF1 Freq (%)'] = mean_ef1_freqs # Updated label
        results['Std Avg EF1 Freq (%)'] = std_ef1_freqs # Updated label
        results[f'Lower {confidence*100:.0f}% CI (Avg EF1 Freq %)'] = lower_ef1_freqs # Updated label
        results[f'Upper {confidence*100:.0f}% CI (Avg EF1 Freq %)'] = upper_ef1_freqs # Updated label
    # --- End new columns ---
    
    # --- Add print summary for new metrics (using AVG labels) ---
    if has_nash_welfare:
        print(f"Average (over agents) of Mean Avg Norm Nash Welfare: {np.mean(mean_nash_welfares):.8f}") # Updated label
    if has_util_welfare:
        print(f"Average (over agents) of Mean Avg % Max Util Welfare: {np.mean(mean_util_welfares):.8f}") # Updated label
    if has_ef1_freq:
        print(f"Average (over agents) of Mean Avg EF1 Frequency (%): {np.mean(mean_ef1_freqs):.8f}") # Updated label
    # --- End print summary ---
    if has_nash_welfare_adv:
        print(f"Maximum (over agents) of Mean Avg Norm Nash Welfare: {np.max(mean_nash_welfares_adv):.8f}")
    if has_util_welfare:
        print(f"Maximum (over agents) of Mean Avg % Max Util Welfare: {np.max(mean_util_welfares):.8f}")
    if has_ef1_freq:
        print(f"Maximum (over agents) of Mean Avg EF1 Frequency (%): {np.max(mean_ef1_freqs):.8f}")
    
    # --- Add new metric columns to DataFrame (using EXPECTED keys/labels) ---
    if has_nash_welfare:
        results['Mean Expected Norm Nash Welfare'] = mean_nash_welfares # Updated label
        results['Std Expected Norm Nash Welfare'] = std_nash_welfares  # Updated label
        results[f'Lower {confidence*100:.0f}% CI (Exp Norm Nash Welf)'] = lower_nash_welfares # Updated label
        results[f'Upper {confidence*100:.0f}% CI (Exp Norm Nash Welf)'] = upper_nash_welfares # Updated label
    if has_nash_welfare_adv:
        results['Mean Expected Norm Nash Welfare Adv'] = mean_nash_welfares_adv # Updated label
        results['Std Expected Norm Nash Welfare Adv'] = std_nash_welfares_adv # Updated label
        results[f'Lower {confidence*100:.0f}% CI (Exp Norm Nash Welf Adv)'] = lower_nash_welfares_adv # Updated label
        results[f'Upper {confidence*100:.0f}% CI (Exp Norm Nash Welf Adv)'] = upper_nash_welfares_adv # Updated label
        
    if has_util_welfare:
        results['Mean Expected % Max Util Welfare'] = mean_util_welfares # Updated label
        results['Std Expected % Max Util Welfare'] = std_util_welfares # Updated label
        results[f'Lower {confidence*100:.0f}% CI (Exp % Max Util Welf)'] = lower_util_welfares # Updated label
        results[f'Upper {confidence*100:.0f}% CI (Exp % Max Util Welf)'] = upper_util_welfares # Updated label
        
    if has_ef1_freq:
        results['Mean Expected EF1 Freq (%)'] = mean_ef1_freqs # Updated label
        results['Std Expected EF1 Freq (%)'] = std_ef1_freqs # Updated label
        results[f'Lower {confidence*100:.0f}% CI (Exp EF1 Freq %)'] = lower_ef1_freqs # Updated label
        results[f'Upper {confidence*100:.0f}% CI (Exp EF1 Freq %)'] = upper_ef1_freqs # Updated label
    
    if has_nash_welfare:
        print(f"Average (over agents) of Mean Expected Norm Nash Welfare: {np.mean(mean_nash_welfares):.8f}") # Updated label
    if has_util_welfare:
        print(f"Average (over agents) of Mean Expected % Max Util Welfare: {np.mean(mean_util_welfares):.8f}") # Updated label
    if has_ef1_freq:
        print(f"Average (over agents) of Mean Expected EF1 Frequency (%): {np.mean(mean_ef1_freqs):.8f}") # Updated label

    if has_nash_welfare:
        print(f"Maximum (over agents) of Mean Expected Norm Nash Welfare: {np.max(mean_nash_welfares):.8f}") # Updated label
    if has_util_welfare:
        print(f"Maximum (over agents) of Mean Expected % Max Util Welfare: {np.max(mean_util_welfares):.8f}") # Updated label
    if has_ef1_freq:
        print(f"Maximum (over agents) of Mean Expected EF1 Frequency (%): {np.max(mean_ef1_freqs):.8f}") # Updated label
   

    return results

def run_bootstrap_analysis(performance_matrix, num_bootstrap=1000, confidence=0.95):
    """
    Run bootstrap analysis on a performance matrix using non-parametric bootstrapping.
    
    Args:
        performance_matrix: DataFrame with agent performance data (mean performance for each pair)
        num_bootstrap: Number of bootstrap replicas to generate
        confidence: Confidence level for intervals
        
    Returns:
        List of bootstrap sample results
    """
    print(f"Running non-parametric bootstrap analysis with {num_bootstrap} samples...")
    
    agents = performance_matrix.index.tolist()
    num_agents = len(agents)
    
    # Convert the performance matrix to a numpy array
    game_matrix_np = performance_matrix.to_numpy()
    
    # Handle missing values
    for i in range(game_matrix_np.shape[0]):
        for j in range(game_matrix_np.shape[1]):
            if np.isnan(game_matrix_np[i, j]):
                # Try column mean first (more relevant for opponent-specific performance)
                col_mean = np.nanmean(game_matrix_np[:, j])
                if not np.isnan(col_mean):
                    game_matrix_np[i, j] = col_mean
                else:
                    # Fall back to row mean if column mean is not available
                    row_mean = np.nanmean(game_matrix_np[i, :])
                    game_matrix_np[i, j] = row_mean if not np.isnan(row_mean) else 0
    
    bootstrap_results = []
    
    # For each bootstrap iteration
    for b in range(num_bootstrap):
        if b % 100 == 0 and b > 0:
            print(f"Processed {b} bootstrap samples...")
        
        # Create a bootstrap sample by resampling with replacement
        bootstrap_indices = np.random.choice(
            range(game_matrix_np.shape[0]), 
            size=game_matrix_np.shape[0], 
            replace=True
        )
        
        # Create a bootstrap game matrix
        bootstrap_game_matrix = game_matrix_np[bootstrap_indices][:, bootstrap_indices]
        
        # Calculate ME Nash equilibrium
        try:
            nash_strategy = milp_max_sym_ent_2p(bootstrap_game_matrix, 2000)
            
            # Calculate Replicator Dynamics Nash Equilibrium
            # rd_strategy, _ = replicator_dynamics_nash(bootstrap_game_matrix, 2000)
            
            # Calculate expected utilities against the ME Nash mixture
            expected_utils = np.dot(bootstrap_game_matrix, nash_strategy)
            
            # Calculate expected utilities against the RD Nash mixture
            # rd_expected_utils = np.dot(bootstrap_game_matrix, rd_strategy)
            
            # Calculate ME Nash equilibrium value
            nash_value = nash_strategy.reshape((1, -1)) @ bootstrap_game_matrix @ nash_strategy.reshape((-1, 1))
            nash_value = nash_value.item()  # Convert to scalar
            
            # Calculate RD Nash equilibrium value
            # rd_nash_value = rd_strategy.reshape((1, -1)) @ bootstrap_game_matrix @ rd_strategy.reshape((-1, 1))
            # rd_nash_value = rd_nash_value.item()  # Convert to scalar
            
            # Calculate Nash equilibrium regret
            nash_regrets = nash_value - expected_utils
            
            # Calculate Nash equilibrium regret for RD NE
            # rd_regrets = rd_nash_value - rd_expected_utils
            
            # Store results for this bootstrap sample
            bootstrap_results.append({
                'ne_regrets': nash_regrets,
                'ne_strategy': nash_strategy,
                # 'rd_regrets': rd_regrets,
                # 'rd_strategy': rd_strategy,
                'expected_utils': expected_utils,
                'nash_value': nash_value,
                # 'rd_nash_value': rd_nash_value
            })
        except Exception as e:
            print(f"Error in bootstrap sample {b}: {e}")
    
    print(f"Successfully completed {len(bootstrap_results)} bootstrap samples.")
    
    return bootstrap_results

def plot_bootstrap_iteration(bootstrap_results, statistic_key, agent_names, output_dir='bootstrap_analysis'):
    """
    Create bootstrap iteration plots to visualize how statistics stabilize with more bootstrap samples.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        statistic_key: Key in bootstrap_results for the statistic to analyze ('ne_regret', 'rd_regret', etc.)
        agent_names: Names of agents for labeling
        output_dir: Directory to save output plots
        
    Returns:
        tuple: (running_means, running_stds, running_errors, running_ci_width)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data based on dictionary format
    if isinstance(bootstrap_results, dict) and statistic_key in bootstrap_results:
        data = bootstrap_results[statistic_key]
    else:
        print(f"Error: couldn't find {statistic_key} in bootstrap results")
        return None, None, None, None
    
    # Convert data to numpy array
    try:
        data_array = np.array(data, dtype=np.float64)
        # Handle single-dimensional data
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(-1, 1)
    except Exception as e:
        print(f"Error converting {statistic_key} data to array: {e}")
        # As a fallback, try creating an array of consistent shape
        n_agents = len(agent_names)
        n_samples = len(data)
        data_array = np.zeros((n_samples, n_agents), dtype=np.float64)
        for i, sample in enumerate(data):
            try:
                if isinstance(sample, (list, np.ndarray)) and len(sample) == n_agents:
                    data_array[i, :] = sample
                elif isinstance(sample, (int, float, np.number)):
                    data_array[i, 0] = float(sample)
                else:
                    # Skip invalid data
                    data_array[i, :] = np.nan
            except Exception as inner_e:
                print(f"Error processing sample {i}: {inner_e}")
                data_array[i, :] = np.nan
    
    n_samples = data_array.shape[0]
    if n_samples < 2:
        print(f"Not enough samples to create bootstrap iteration plot for {statistic_key}")
        return None, None, None, None
    
    n_agents = data_array.shape[1]
    agent_subset = agent_names[:n_agents]
    
    # Arrays to store running statistics
    running_means = np.zeros((n_samples-1, n_agents))
    running_stds = np.zeros((n_samples-1, n_agents))
    running_errors = np.zeros((n_samples-1, n_agents))
    running_ci_width = np.zeros((n_samples-1, n_agents))
    
    # Calculate running statistics
    # Start from i=1 since we need at least 2 samples for std
    for i in range(2, n_samples+1):
        running_means[i-2] = np.nanmean(data_array[:i], axis=0)
        
        # Calculate std with ddof=1 for unbiased estimation
        # For small samples, use try-except to handle potential issues
        try:
            std_values = np.nanstd(data_array[:i], axis=0, ddof=1)
            running_stds[i-2] = std_values
            
            # Calculate standard error (SE = std / sqrt(n))
            # Ensure proper handling of scalar vs array operations
            n_samples_i = np.sum(~np.isnan(data_array[:i]), axis=0)
            n_samples_i = np.maximum(n_samples_i, 1)  # Ensure at least 1 to avoid division by zero
            
            # Use np.sqrt directly on array elements
            sqrt_n = np.sqrt(n_samples_i)
            running_errors[i-2] = std_values / sqrt_n
            
            # Calculate CI width (assuming normal distribution)
            z_value = 1.96  # 95% confidence
            running_ci_width[i-2] = 2 * z_value * running_errors[i-2]
        except Exception as e:
            print(f"Warning: Error calculating statistics for sample {i}: {e}")
            running_stds[i-2] = np.nan
            running_errors[i-2] = np.nan
            running_ci_width[i-2] = np.nan
    
    # Create plots
    print(f"Creating bootstrap iteration plot for {statistic_key}")
    
    # Different plot types: means, std, errors, CI width
    plot_titles = {
        'means': 'Running Mean',
        'stds': 'Running Standard Deviation',
        'errors': 'Running Standard Error',
        'ci_width': 'Running CI Width'
    }
    
    for plot_type, plot_title_suffix in plot_titles.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get the data for this plot type
        if plot_type == 'means':
            plot_data = running_means
        elif plot_type == 'stds':
            plot_data = running_stds
        elif plot_type == 'errors':
            plot_data = running_errors
        else:  # ci_width
            plot_data = running_ci_width
        
        # Plot each agent
        for agent_idx, agent_name in enumerate(agent_subset):
            if agent_idx < plot_data.shape[1]:
                # Extract non-NaN values
                agent_data = plot_data[:, agent_idx]
                valid_mask = ~np.isnan(agent_data)
                if np.any(valid_mask):
                    x_vals = np.arange(2, n_samples+1)[valid_mask]
                    ax.plot(x_vals, agent_data[valid_mask], label=agent_name)
        
        ax.set_title(f"{statistic_key.replace('_', ' ').title()} - {plot_title_suffix}")
        ax.set_xlabel('Number of Bootstrap Samples')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"{statistic_key}_{plot_type.lower()}_iteration.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved {plot_type} iteration plot to {os.path.join(output_dir, filename)}")
    
    return running_means, running_stds, running_errors, running_ci_width

def plot_confidence_interval_stability(bootstrap_results, statistic_key, agent_names, output_dir='bootstrap_analysis'):
    """
    Plot confidence interval stability to determine if more bootstrap samples are needed.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        statistic_key: Key in bootstrap_results for the statistic to analyze ('ne_regret', 'rd_regret', etc.)
        agent_names: Names of agents for labeling
        output_dir: Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data based on dictionary format
    if isinstance(bootstrap_results, dict) and statistic_key in bootstrap_results:
        data = bootstrap_results[statistic_key]
    else:
        print(f"Error: couldn't find {statistic_key} in bootstrap results")
        return
        
    data_array = np.array(data)
    n_samples = len(data)
    n_agents = data_array.shape[1]
    
    print(f"Creating confidence interval stability plot for {statistic_key}")
    
    fig, axes = plt.subplots(n_agents, 1, figsize=(12, 4 * n_agents), sharex=True)
    
    # Handle case with single agent
    if n_agents == 1:
        axes = [axes]
    
    
    confidence_levels = [0.90, 0.95, 0.99]
    colors = ['blue', 'green', 'red']
    
    start_idx = min(10, n_samples - 1)
    
    for agent_idx in range(n_agents):
        ax = axes[agent_idx]
        agent_name = agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent {agent_idx}"
        
        for ci_idx, confidence in enumerate(confidence_levels):
            lower_percentile = (1 - confidence) / 2 * 100
            upper_percentile = (1 - (1 - confidence) / 2) * 100
            
            running_lower_ci = np.zeros(n_samples - start_idx)
            running_upper_ci = np.zeros(n_samples - start_idx)
            
            for i in range(start_idx, n_samples):
                running_lower_ci[i-start_idx] = np.percentile(data_array[:i+1, agent_idx], lower_percentile)
                running_upper_ci[i-start_idx] = np.percentile(data_array[:i+1, agent_idx], upper_percentile)
            
            # Plot the CIs
            x_vals = range(start_idx, n_samples)
            ax.plot(x_vals, running_lower_ci, '--', color=colors[ci_idx], alpha=0.7, 
                    label=f"{confidence*100:.0f}% CI Lower")
            ax.plot(x_vals, running_upper_ci, '-', color=colors[ci_idx], alpha=0.7, 
                    label=f"{confidence*100:.0f}% CI Upper")
            
            # Fill between the CIs
            ax.fill_between(x_vals, running_lower_ci, running_upper_ci, color=colors[ci_idx], alpha=0.1)
        
        # Add a line for the final mean
        final_mean = np.mean(data_array[:, agent_idx])
        ax.axhline(final_mean, color='black', linestyle='-', label="Mean")
        
        ax.set_title(f"{agent_name} - {statistic_key} Confidence Interval Stability")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        
        # Only add legend to the first subplot to save space
        if agent_idx == 0:
            ax.legend()
    
    axes[-1].set_xlabel("Number of Bootstrap Samples")
    plt.tight_layout()
    
    # Save the figure
    filename = f"{statistic_key}_ci_stability.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved confidence interval stability plot to {os.path.join(output_dir, filename)}")

def plot_ci_size_evolution(bootstrap_results, statistic_key, metric_label, agent_names, output_dir='bootstrap_analysis'):
    """
    Plot the evolution of confidence interval sizes during bootstrapping for all agents.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        statistic_key: Key in bootstrap_results for the statistic to analyze
        metric_label: Human-readable label for the metric
        agent_names: Names of agents for labeling
        output_dir: Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data based on dictionary format
    if isinstance(bootstrap_results, dict) and statistic_key in bootstrap_results:
        data = bootstrap_results[statistic_key]
    else:
        print(f"Error: couldn't find {statistic_key} in bootstrap results")
        return
        
    if not data:
        print(f"Error: No data found for statistic '{statistic_key}' in plot_ci_size_evolution.")
        return

    try:
        data_array = np.array(data, dtype=np.float64)
        # Handle single-dimensional data
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(-1, 1)
    except Exception as e:
        print(f"Error converting {statistic_key} data to array in plot_ci_size_evolution: {e}")
        return # Cannot proceed without valid data array

    n_samples = data_array.shape[0]
    if n_samples < 10: # Need enough samples to show evolution
        print(f"Not enough samples ({n_samples}) to plot CI size evolution for {statistic_key}")
        return
        
    n_agents = data_array.shape[1]
    
    print(f"Creating confidence interval size evolution plot for {statistic_key}")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Calculate CI sizes for each agent as sample size increases
    # Start after accumulating at least 10 samples
    start_idx = min(10, n_samples - 1)
    x_vals = range(start_idx, n_samples)
    
    # Use a colormap for distinct colors
    colors = plt.cm.viridis(np.linspace(0, 1, n_agents))
    
    for agent_idx in range(n_agents):
        agent_name = agent_names[agent_idx] if agent_idx < len(agent_names) else f"Agent {agent_idx}"
        
        # Calculate running CI sizes
        ci_sizes = np.zeros(n_samples - start_idx)
        for i in range(start_idx, n_samples):
            try:
                 # Extract data up to sample i+1
                current_data = data_array[:i+1, agent_idx]
                # Remove NaNs before calculating percentiles
                valid_data = current_data[~np.isnan(current_data)]
                if len(valid_data) >= 2: # Need at least 2 points for percentile
                    lower_ci = np.percentile(valid_data, 2.5)
                    upper_ci = np.percentile(valid_data, 97.5)
                    ci_sizes[i-start_idx] = abs(upper_ci - lower_ci)
                else:
                    ci_sizes[i-start_idx] = np.nan # Not enough data
            except IndexError:
                 ci_sizes[i-start_idx] = np.nan # Handle potential index errors
            except Exception as e:
                 print(f"Warning: Error calculating CI size for {agent_name} at sample {i}: {e}")
                 ci_sizes[i-start_idx] = np.nan
        
        # Plot the CI size evolution
        valid_mask = ~np.isnan(ci_sizes)
        if np.any(valid_mask):
            plt.plot(np.array(x_vals)[valid_mask], ci_sizes[valid_mask], color=colors[agent_idx], label=agent_name, linewidth=2)
    
    # Use metric_label in the title
    plt.title(f"Evolution of 95% Confidence Interval Sizes\nfor {metric_label}")
    plt.xlabel("Number of Bootstrap Samples")
    plt.ylabel("Confidence Interval Size")
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure using statistic_key in filename
    filename = f"{statistic_key}_ci_size_evolution.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confidence interval size evolution plot to {os.path.join(output_dir, filename)}")

def analyze_bootstrap_results_for_convergence(bootstrap_results, agent_names, output_dir='bootstrap_analysis'):
    """
    Analyze bootstrap results for convergence based on the bootstrap paper methods.
    This helps determine if you need more bootstrap samples or more simulator data.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        agent_names: Names of agents for labeling
        output_dir: Directory to save output plots
    
    Returns:
        Dictionary with convergence metrics and assessment
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\nAnalyzing bootstrap convergence using methods from the bootstrap paper...")
    
    # Define the statistics to analyze and their human-readable labels
    statistics_to_analyze = {
        'ne_regret': 'NE Regret',
        # 'rd_regret': 'RD Regret', # Removed as per user request
        'agent_expected_utility': 'Expected Utility vs NE',
        'agent_expected_nash_welfare': 'Nash Welfare vs NE',
        'agent_expected_util_welfare': 'Utilitarian Welfare vs NE',
        'agent_expected_ef1_freq': 'EF1 Frequency vs NE (%)',
        'agent_expected_nash_welfare_adv': 'Nash Welfare Adv vs NE',

    }
    
    # Filter statistics based on availability in bootstrap_results
    available_statistics = {}
    for key, label in statistics_to_analyze.items():
        if key in bootstrap_results and bootstrap_results[key]:
            available_statistics[key] = label
        else:
            print(f"Skipping convergence analysis for '{label}' (key: '{key}') - data not found or empty.")

    if not available_statistics:
        print("No valid statistics found in bootstrap results for convergence analysis.")
        return {}
    
    # Generate iteration plots and confidence interval stability plots for each available statistic
    convergence_results = {}
    
    for stat_key, stat_name in available_statistics.items():
        print(f"\nAnalyzing {stat_name}...")
        
        # Generate bootstrap iteration plots (Mean, Std, Error, CI Width)
        try:
             plot_bootstrap_iteration(
                 bootstrap_results, stat_key, agent_names, output_dir
             )
        except Exception as e:
             print(f"  >> Error generating iteration plots for {stat_key}: {e}")
        
        # Generate confidence interval stability plots
        try:
            plot_confidence_interval_stability(
                bootstrap_results, stat_key, agent_names, output_dir
            )
        except Exception as e:
            print(f"  >> Error generating CI stability plots for {stat_key}: {e}")

        # Generate CI size evolution plot
        try:
            plot_ci_size_evolution(
                bootstrap_results, stat_key, stat_name, agent_names, output_dir
            )
        except Exception as e:
            print(f"  >> Error generating CI size evolution plot for {stat_key}: {e}")

        # Calculate final Monte Carlo errors and convergence assessment with proper error handling
        try:
            data = bootstrap_results[stat_key]
            
            # Convert data to numpy array with proper error handling
            try:
                data_array = np.array(data, dtype=np.float64)
                # Handle single-dimensional data
                if len(data_array.shape) == 1:
                    data_array = data_array.reshape(-1, 1)
            except Exception as e:
                print(f"Warning: Error converting {stat_key} data to array: {e}")
                # As a fallback, create array of consistent shape
                n_agents = len(agent_names)
                n_samples = len(data)
                data_array = np.zeros((n_samples, n_agents), dtype=np.float64)
                for i, sample in enumerate(data):
                    try:
                        if isinstance(sample, (list, np.ndarray)) and len(sample) == n_agents:
                            data_array[i, :] = sample
                        elif isinstance(sample, (int, float, np.number)):
                            data_array[i, 0] = float(sample)
                        else:
                            # Skip invalid data
                            data_array[i, :] = np.nan
                    except Exception as inner_e:
                        print(f"Error processing sample {i}: {inner_e}")
                        data_array[i, :] = np.nan
            
            n_samples = data_array.shape[0]
            n_agents = data_array.shape[1]
            
            # Calculate final statistics safely
            try:
                final_means = np.nanmean(data_array, axis=0)
                final_stds = np.nanstd(data_array, axis=0, ddof=1)
                final_errors = np.zeros_like(final_stds)
                
                # Calculate standard error safely (SE = std / sqrt(n))
                for i in range(n_agents):
                    try:
                        # Count valid (non-NaN) samples for this agent
                        valid_samples = np.sum(~np.isnan(data_array[:, i]))
                        if valid_samples > 1:  # Need at least 2 samples for std
                            # Use float() to ensure we get a scalar value for sqrt
                            sqrt_n = float(np.sqrt(valid_samples))
                            final_errors[i] = final_stds[i] / sqrt_n
                        else:
                            final_errors[i] = 0.0
                    except Exception as calc_err:
                        print(f"Warning: Error calculating standard error for agent {i}: {calc_err}")
                        final_errors[i] = 0.0
                
                # Calculate relative errors safely
                relative_errors = np.zeros(n_agents)
                for i in range(n_agents):
                    try:
                        if abs(final_means[i]) > 1e-8:
                            relative_errors[i] = abs(final_errors[i] / final_means[i])
                        else:
                            relative_errors[i] = final_errors[i] / max(final_stds[i], 1e-8)
                    except Exception as rel_err:
                        print(f"Warning: Error calculating relative error for agent {i}: {rel_err}")
                        relative_errors[i] = 0.0
                
                # Assess convergence
                convergence_assessment = [""] * n_agents
                for i in range(n_agents):
                    if relative_errors[i] < 0.01:
                        convergence_assessment[i] = "Excellent"
                    elif relative_errors[i] < 0.05:
                        convergence_assessment[i] = "Good"
                    elif relative_errors[i] < 0.10:
                        convergence_assessment[i] = "Fair"
                    else:
                        convergence_assessment[i] = "Poor - More samples needed"
                
                # Store results
                convergence_results[stat_key] = {
                    'final_means': final_means,
                    'final_stds': final_stds,
                    'monte_carlo_errors': final_errors,
                    'relative_errors': relative_errors,
                    'convergence_assessment': convergence_assessment
                }
                
                # Print summary
                print(f"\n{stat_name} Monte Carlo Errors:")
                for i in range(min(n_agents, len(agent_names))):
                    agent_name = agent_names[i] if i < len(agent_names) else f"Agent {i}"
                    print(f"  {agent_name}: {final_errors[i]:.6f} (Relative: {relative_errors[i]:.2%}) - {convergence_assessment[i]}")
                
                # Save detailed convergence results to CSV
                results_df = pd.DataFrame({
                    'Agent': agent_names[:n_agents],
                    'Mean': final_means,
                    'Std Dev': final_stds,
                    'Monte Carlo Error': final_errors,
                    'Relative Error': relative_errors,
                    'Assessment': convergence_assessment
                })
                
                results_df.to_csv(os.path.join(output_dir, f"{stat_key}_convergence.csv"), index=False)
                print(f"Saved detailed convergence results to {os.path.join(output_dir, f'{stat_key}_convergence.csv')}")
                
            except Exception as stats_err:
                print(f"Error calculating statistics for {stat_key}: {stats_err}")
                # Create placeholder results
                convergence_results[stat_key] = {
                    'final_means': np.zeros(n_agents),
                    'final_stds': np.zeros(n_agents),
                    'monte_carlo_errors': np.zeros(n_agents),
                    'relative_errors': np.zeros(n_agents),
                    'convergence_assessment': ["Error in calculation"] * n_agents
                }
                
        except Exception as e:
            print(f"Error analyzing {stat_key}: {e}")
            # Create placeholder results
            convergence_results[stat_key] = {
                'error': str(e)
            }
    
    return convergence_results

def check_bootstrap_convergence(bootstrap_results, agent_names, window_size=None, verbose=True):
    """
    Check if bootstrap results have converged by examining the stability of statistics.
    
    Args:
        bootstrap_results: Dictionary of bootstrap results
        agent_names: List of agent names
        window_size: Size of the window for checking stability (default: 20% of samples)
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with convergence status for different metrics
    """
    # Extract key statistics
    try:
        ne_regrets = bootstrap_results.get('ne_regret', [])
        
        # Try both keys for expected utilities (for backward compatibility)
        expected_utils = bootstrap_results.get('agent_expected_utility', bootstrap_results.get('expected_utility', []))
        
        # rd_regrets = bootstrap_results.get('rd_regret', []) # Removed
        
        n_samples = len(ne_regrets) if ne_regrets else 0
        if n_samples == 0:
            print("No bootstrap samples available for convergence check.")
            # return {'ne_converged': False, 'eu_converged': False, 'rd_converged': False} # Modified
            return {'ne_converged': False, 'eu_converged': False} 
        
        # Set default window size if not specified
        if window_size is None:
            window_size = max(1, n_samples // 5)  # Use 20% of samples
        
        window_size = min(window_size, n_samples - 1)  # Ensure window size is valid
        
        if verbose:
            print(f"\nUsing window size of {window_size} for convergence analysis")
            print(f"Number of NE regret samples: {len(ne_regrets)}")
            print(f"Number of expected utility samples: {len(expected_utils)}")
            # print(f"Number of RD regret samples: {len(rd_regrets)}") # Removed
        
        # Calculate Monte Carlo errors with error handling
        def calculate_monte_carlo_error(data, agent_labels=None):
            try:
                # Check if we have any data
                if not data or len(data) == 0:
                    print("Warning: Empty data array in Monte Carlo error calculation")
                    return np.array([0.0])
                    
                # Convert to numpy array with proper error handling
                try:
                    # Try to handle both array and scalar data
                    if isinstance(data[0], (np.float64, float, int, np.int64)) or (hasattr(data[0], 'shape') and len(data[0].shape) == 0):
                        # Convert scalar data to proper array first
                        data_array = np.array(data, dtype=np.float64).reshape(-1, 1)
                    else:
                        # Use standard array calculation
                        data_array = np.array(data, dtype=np.float64)
                    
                    # Ensure proper dimensions
                    if len(data_array.shape) == 1:
                        data_array = data_array.reshape(-1, 1)
                    
                    # Calculate statistics
                    means = np.mean(data_array, axis=0)
                    stds = np.std(data_array, axis=0, ddof=1)
                    
                    # Handle division by sqrt manually to avoid type errors
                    n = len(data_array)
                    if n > 1:
                        sqrt_n = float(np.sqrt(n))  # Explicitly convert to float
                        mc_errors = stds / sqrt_n
                        
                        # Add agent names to the output if provided
                        if agent_labels and verbose:
                            for i, (error, label) in enumerate(zip(mc_errors, agent_labels)):
                                print(f"{label}: {error:.6f}")
                        return mc_errors
                    else:
                        return np.zeros_like(means)
                        
                except (ValueError, TypeError) as e:
                    print(f"Warning: Error converting data for Monte Carlo calculation: {e}")
                    # Create placeholder results with agent names
                    errors = np.zeros(len(agent_labels) if agent_labels else 1)
                    if agent_labels and verbose:
                        for i, label in enumerate(agent_labels):
                            print(f"{label}: 0.000000")
                    return errors
                    
            except Exception as e:
                print(f"Error calculating Monte Carlo error: {e}")
                # Return zero errors with agent names
                errors = np.zeros(len(agent_labels) if agent_labels else 1)
                if agent_labels and verbose:
                    for i, label in enumerate(agent_labels):
                        print(f"{label}: 0.000000")
                return errors
        
        # Check convergence of a metric
        def check_metric_convergence(data_array, threshold=0.05):
            try:
                if len(data_array) < window_size + 1:
                    # return False, "Not enough samples" # Original
                    return False, "Not enough samples for convergence check"


                # Convert to numpy array if needed
                if not isinstance(data_array, np.ndarray):
                    data_array = np.array(data_array, dtype=np.float64)
                
                # Check if array shape is correct
                if len(data_array.shape) == 1:
                    data_array = data_array.reshape(-1, 1)
                
                # Calculate statistics for final window vs full dataset
                window_mean = np.mean(data_array[-window_size:], axis=0)
                full_mean = np.mean(data_array, axis=0)
                
                # Calculate relative difference
                abs_diff = np.abs(window_mean - full_mean)
                rel_diff = np.where(
                    np.abs(full_mean) > 1e-10,
                    abs_diff / np.abs(full_mean),
                    np.zeros_like(abs_diff)
                )
                
                # Check if all agents have converged
                max_rel_diff = np.max(rel_diff)
                
                return max_rel_diff < threshold, f"Max relative difference: {max_rel_diff:.4f}"
            except Exception as e:
                print(f"Error checking convergence: {e}")
                return False, f"Error in convergence check: {e}"
        
        # Check convergence of the three key metrics
        ne_converged, ne_status = check_metric_convergence(ne_regrets)
        eu_converged, eu_status = check_metric_convergence(expected_utils)
        # rd_converged, rd_status = check_metric_convergence(rd_regrets) # Removed
        
        # Print convergence summary
        if verbose:
            print("\nBootstrap Convergence Analysis:")
            print("--------------------------------------------------")
            
            print("\nNash Equilibrium Regrets:")
            print(f"Status: {ne_status}")
            print("Monte Carlo Errors:")
            ne_errors = calculate_monte_carlo_error(ne_regrets, agent_names)
            
            print("\nExpected Utilities:")
            print(f"Status: {eu_status}")
            print("Monte Carlo Errors:")
            eu_errors = calculate_monte_carlo_error(expected_utils, agent_names)
            
            # print("\\nReplicator Dynamics Regrets:") # Removed
            # print(f"Status: {rd_status}") # Removed
            # print("Monte Carlo Errors:") # Removed
            # rd_errors = calculate_monte_carlo_error(rd_regrets, agent_names) # Removed
            
            # if not (ne_converged and eu_converged and rd_converged): # Modified
            if not (ne_converged and eu_converged):
                print("\nWARNING: Some statistics have not converged. Consider increasing the number of bootstrap samples.")
        
        return {
            'ne_converged': ne_converged, 
            'eu_converged': eu_converged, 
            # 'rd_converged': rd_converged, # Removed
            'ne_errors': ne_errors if 'ne_errors' in locals() else np.zeros(len(agent_names) if agent_names else 1),
            'eu_errors': eu_errors if 'eu_errors' in locals() else np.zeros(len(agent_names) if agent_names else 1),
            # 'rd_errors': rd_errors if 'rd_errors' in locals() else np.zeros(len(agent_names) if agent_names else 1) # Removed
        }
    except Exception as e:
        print(f"Error during convergence analysis: {e}")
        return {
            'ne_converged': False, 
            'eu_converged': False, 
            # 'rd_converged': False, # Removed
            'ne_errors': np.zeros(len(agent_names) if agent_names else 1),
            'eu_errors': np.zeros(len(agent_names) if agent_names else 1),
            # 'rd_errors': np.zeros(len(agent_names) if agent_names else 1) # Removed
        }

def analyze_bootstrap_convergence(bootstrap_results, agent_names, window_size=None, threshold=0.05, output_dir=None):
    """
    Analyze convergence of bootstrap results with detailed statistics and visualization options.
    
    Args:
        bootstrap_results: Dictionary containing bootstrap results
        agent_names: List of agent names
        window_size: Size of the window for checking stability (default: 20% of samples)
        threshold: Threshold for determining convergence (default: 0.05)
        output_dir: Directory to save output plots and statistics
        
    Returns:
        Dictionary with convergence status and detailed metrics
    """
    print("\nAnalyzing bootstrap convergence using methods from the bootstrap paper...")
    
    # First, use the basic convergence check
    basic_convergence = check_bootstrap_convergence(bootstrap_results, agent_names, window_size)
    
    # If output_dir is provided, run the more comprehensive analysis
    if output_dir is not None:
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Run enhanced analysis that creates plots and saves results
            enhanced_results = analyze_bootstrap_results_for_convergence(
                bootstrap_results, agent_names, output_dir
            )
            
            # Combine the results from both analyses
            combined_results = {**basic_convergence, 'enhanced': enhanced_results}
            return combined_results
        except Exception as e:
            print(f"Error running enhanced convergence analysis: {e}")
            print("Continuing with basic convergence analysis.")
    
    return basic_convergence

def plot_regret_distributions(regrets_list, agent_names, title="Regret Distribution", plot_type="histogram"):
    """
    Create distribution plots for regrets across agents.
    
    Args:
        regrets_list: List of regret arrays from bootstrap samples
        agent_names: List of agent names
        title: Title for the plot
        plot_type: Type of plot to create ('histogram', 'box', or 'running_mean')
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    # --- Start Refactoring ---
    # Rename function and generalize parameters
def plot_bootstrap_distributions(bootstrap_results, statistic_key, metric_label, agent_names, title=None, plot_type="histogram"):
        """
        Create distribution plots for a given bootstrap statistic across agents.
        
        Args:
            bootstrap_results: Dictionary containing bootstrap results.
            statistic_key: Key in bootstrap_results for the statistic to plot 
                         (e.g., 'ne_regret', 'agent_expected_nash_welfare').
            metric_label: Human-readable label for the metric (e.g., 'NE Regret', 'Nash Welfare vs NE').
            agent_names: List of agent names.
            title: Title for the plot (defaults to metric_label Distribution).
            plot_type: Type of plot to create ('histogram', 'box', or 'running_mean').
            
        Returns:
            matplotlib.figure.Figure: The generated plot, or None if an error occurs.
        """
        if title is None:
            title = f"{metric_label} Distribution"

        # Extract the relevant data using statistic_key
        if not isinstance(bootstrap_results, dict) or statistic_key not in bootstrap_results:
            print(f"Error: Statistic key '{statistic_key}' not found in bootstrap_results.")
            return None
        
        data_list = bootstrap_results[statistic_key]

        if not data_list:
            print(f"Error: No data found for statistic '{statistic_key}'.")
            return None
        eq_key_map = {
            'agent_expected_ef1_freq': 'equilibrium_ef1_freq',
            'agent_expected_percent_max_util_welfare': 'equilibrium_percent_max_util_welfare',
            'agent_expected_normalized_nash_welfare': 'equilibrium_normalized_nash_welfare',
            'agent_expected_normalized_nash_welfare_adv': 'equilibrium_normalized_nash_welfare_adv',
        }
        eq_key = eq_key_map.get(statistic_key)
        if eq_key and bootstrap_results.get(eq_key):
            agent_names = list(agent_names) + ['NE_mix']
        # Append the scalar equilibrium value to each bootstrap sample
            data_list = [
                np.append(
                    np.asarray(sample, dtype=np.float64).reshape(-1),
                    float(bootstrap_results[eq_key][i]) if i < len(bootstrap_results[eq_key]) else np.nan
                )
                for i, sample in enumerate(data_list)
            ]

        # --- End Refactoring --- 
        # Wrap the main logic in a try-except block
        try:
            # Create figure and axes based on plot type
            if plot_type == "histogram":
                n_agents = len(agent_names)
                n_cols = 3 # Or choose based on desired layout
                n_rows = int(np.ceil(n_agents / n_cols))
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False, sharey=False)
                axs = axs.flatten() # Flatten the grid for easy iteration
            else: # For boxplot or running_mean, use a single axis
                 fig, ax = plt.subplots(figsize=(12, 8)) # Adjust figsize as needed for single plots
                 axs = None # Indicate single axis mode
            
            max_samples = 1000  #SHould be number of bootstrap samples
            if len(data_list) > max_samples:
                print(f"Warning: Too many bootstrap samples ({len(data_list)}). Using a random subset of {max_samples} samples.")
                random.seed(42)  
                data_list = random.sample(data_list, max_samples)
            
            n_agents = len(agent_names)
            
            
            min_value = float('inf')
            max_value = float('-inf')
            
            chunk_size = 100 
            for i in range(0, len(data_list), chunk_size):
                chunk = data_list[i:i+chunk_size]
                for sample in chunk:
                    if not isinstance(sample, np.ndarray):
                        sample = np.array(sample, dtype=np.float64)
                    
                    if len(sample.shape) == 0:
                        sample = np.array([sample])
                        
                    valid_len = min(len(sample), n_agents)
                    sample = sample[:valid_len]
                    
                    sample_min = np.nanmin(sample) if sample.size > 0 else 0
                    sample_max = np.nanmax(sample) if sample.size > 0 else 0
                    min_value = min(min_value, sample_min)
                    max_value = max(max_value, sample_max)
            
            print(f"\nValue range for {title}: [{min_value:.8f}, {max_value:.8f}]")
            
            if "regret" in statistic_key.lower() and max_value > 1e-6: # Use tolerance
                print(f"WARNING: Detected positive values (max: {max_value:.8f}) in {title}")
            
            if plot_type == "histogram":
                for agent_idx, agent_name in enumerate(agent_names):
                    if axs is None or agent_idx >= len(axs): # Check if axs exists and is valid
                         print(f"Warning: Error getting subplot for agent {agent_name}")
                         continue
                    
                    current_ax = axs[agent_idx] 

                    agent_data = []
                    for i in range(0, len(data_list), chunk_size):
                        chunk = data_list[i:i+chunk_size]
                        for sample in chunk:
                            if not isinstance(sample, np.ndarray):
                                sample = np.array(sample, dtype=np.float64)
                            if len(sample.shape) == 0:
                                sample = np.array([sample])
                            if agent_idx < len(sample):
                                val = sample[agent_idx]
                                if not np.isnan(val):
                                    agent_data.append(val)
                    
                    # Only plot if we have data
                    if agent_data:
                         # Calculate agent-specific min/max for binning and limits
                         min_agent_val = np.min(agent_data)
                         max_agent_val = np.max(agent_data)
                         
                         # Determine bins based on agent's data range
                         agent_range = max_agent_val - min_agent_val
                         if np.isclose(agent_range, 0):
                              n_bins = 10
                              bin_edges = np.linspace(min_agent_val - 0.5, max_agent_val + 0.5, n_bins + 1)
                         else:
                              # Heuristic for bins based on range 
                              if agent_range > 100:
                                  n_bins = 20
                              elif agent_range > 10:
                                  n_bins = 30
                              else:
                                  n_bins = 40
                              bin_edges = np.linspace(min_agent_val, max_agent_val, n_bins + 1)
                              
                         hist, _ = np.histogram(agent_data, bins=bin_edges)
                         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                         width = (bin_edges[1] - bin_edges[0]) * 0.8
                         
                         # Plot histogram on the agent's specific subplot
                         current_ax.bar(bin_centers, hist, width=width, alpha=0.7, color='darkgreen')

                         # Add Mean and CI lines
                         mean_val = np.mean(agent_data)
                         lower_ci = np.percentile(agent_data, 2.5)
                         upper_ci = np.percentile(agent_data, 97.5)

                         current_ax.axvline(mean_val, color='r', linestyle='--', 
                                    label=f'Mean: {mean_val:.4f}') 
                         current_ax.axvline(lower_ci, color='orange', linestyle=':', linewidth=2,
                                    label=f'95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]') 
                         current_ax.axvline(upper_ci, color='orange', linestyle=':', linewidth=2)
                         
                         # Add Standard Deviation to the legend
                         std_val = np.std(agent_data)
                         # Add a placeholder line for Std Dev, or incorporate into existing legend text
                         current_ax.plot([], [], linestyle='None', marker='None', label=f'Std Dev: {std_val:.4f}')
                         
                         # Add zero line if relevant for the metric (e.g., regret)
                         if min_agent_val < 0 < max_agent_val:
                              current_ax.axvline(0, color='black', linestyle='-', alpha=0.7, label='Zero Reference')

                         # Set titles and labels for the subplot
                         current_ax.set_title(agent_name, fontsize=10)
                         current_ax.set_xlabel(metric_label, fontsize=9)
                         current_ax.set_ylabel('Frequency', fontsize=9)
                         current_ax.tick_params(axis='both', which='major', labelsize=8)
                         current_ax.legend(fontsize='x-small')
                         
                         # Add text box with full range (optional, like example)
                         #current_ax.text(0.05, 0.95, f"Full range: [{min_agent_val:.2f}, {max_agent_val:.2f}]", 
                                # transform=current_ax.transAxes, ha='left', va='top',
                               #  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'),
                               #  fontsize=7)
                                 
                         # Set x-axis limits for this subplot
                         buffer = abs(max_agent_val - min_agent_val) * 0.1 if not np.isclose(agent_range, 0) else 0.5
                         current_ax.set_xlim(min_agent_val - buffer, max_agent_val + buffer)

                    else:
                         # Handle cases where agent has no data
                         current_ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                         current_ax.set_title(agent_name, fontsize=10)
                         current_ax.set_xticks([])
                         current_ax.set_yticks([])

                # Turn off axes for any unused subplots in the grid
                if axs is not None:
                    for i in range(n_agents, len(axs)):
                         axs[i].axis('off')

                # Add a single main title above all subplots
                fig.suptitle(title, fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

            elif plot_type == "box":
                # Box plot needs data for all agents together
                valid_data = {agent_name: [] for agent_name in agent_names[:n_agents]}
                
                for i in range(0, len(data_list), chunk_size):
                    chunk = data_list[i:i+chunk_size]
                    for sample in chunk:
                        # Convert sample to numpy array if needed
                        if not isinstance(sample, np.ndarray):
                            sample = np.array(sample, dtype=np.float64)
                        
                        # Flatten if needed
                        if len(sample.shape) == 0:
                            sample = np.array([sample])
                            
                        # Add each agent's value to its list
                        for agent_idx, agent_name in enumerate(agent_names):
                            if agent_idx < len(sample):
                                val = sample[agent_idx]
                                if not np.isnan(val):
                                    valid_data[agent_name].append(val)
                
                df_data = {k: v for k, v in valid_data.items() if v}
                if not df_data:
                    print(f"Warning: No valid data for boxplot in {title}")
                    ax.text(0.5, 0.5, "No valid data for boxplot", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                else:
                    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in df_data.items() ])) # Handle uneven lengths
                    # Use the single 'ax' created for non-histogram plots
                    sns.boxplot(data=df, ax=ax)
                    plt.xlabel('Agent')
                    plt.ylabel(metric_label) # Use generic label
                    plt.xticks(rotation=45, ha='right')
                    
                    # Set y-axis range based on actual data
                    buffer = abs(max_value - min_value) * 0.05 if not np.isclose(max_value, min_value) else 0.5
                    ax.set_ylim(min_value - buffer, max_value + buffer)
            
            elif plot_type == "running_mean":
                window_size = max(1, min(len(data_list) // 10, 100))  
                running_mean_min = float('inf')
                running_mean_max = float('-inf')
                
                for agent_idx, agent_name in enumerate(agent_names):
                    if agent_idx >= n_agents:
                        continue
                    
                    # Calculate running means for this agent
                    all_values = []
                    
                    # First collect all values for this agent
                    for sample in data_list:
                        # Convert sample to numpy array if needed
                        if not isinstance(sample, np.ndarray):
                            sample = np.array(sample, dtype=np.float64)
                        
                        # Flatten if needed
                        if len(sample.shape) == 0:
                            sample = np.array([sample])
                            
                        # Get this agent's value if available
                        if agent_idx < len(sample):
                            val = sample[agent_idx]
                            if not np.isnan(val):
                                all_values.append(val)
                
                    # Calculate running means
                    if len(all_values) > window_size:
                        running_means = []
                        # Limit the number of points plotted for very large datasets
                        stride = max(1, (len(all_values) - window_size) // 500)
                        indices = range(window_size, len(all_values) + 1, stride)
                        for j in indices:
                            mean_value = np.mean(all_values[j-window_size:j])
                            running_means.append(mean_value)
                            running_mean_min = min(running_mean_min, mean_value)
                            running_mean_max = max(running_mean_max, mean_value)
                        
                        x_vals = list(indices)
                        if len(x_vals) == len(running_means):
                            plt.plot(x_vals, running_means, label=agent_name)
                        else:
                            print(f"Warning: x_vals ({len(x_vals)}) and running_means ({len(running_means)}) have different lengths for {agent_name}")
            
                # Use the single 'ax' for labels and limits
                ax.set_xlabel('Number of Bootstrap Samples')
                ax.set_ylabel(f'Running Mean {metric_label}') # Use generic label
                
                # Set y-axis range for running mean plots
                buffer = abs(running_mean_max - running_mean_min) * 0.05 if not np.isclose(running_mean_max, running_mean_min) else 0.5
                if running_mean_min != float('inf') and running_mean_max != float('-inf'):
                    ax.set_ylim(running_mean_min - buffer, running_mean_max + buffer)
            
            else:
                raise ValueError(f"Unsupported plot_type: {plot_type}. Use 'histogram', 'box', or 'running_mean'.")
            
            # Set title only if not histogram (histograms have suptitle)
            if plot_type != "histogram":
                 ax.set_title(title) # Use single 'ax' for title
                 ax.grid(True, alpha=0.3) # Use single 'ax'
            else: # For histogram, grid was handled per subplot
                 pass # Grid is applied per subplot for histograms
            
            # Add legend if not boxplot or histogram (they handle legends differently)
            if plot_type != "box" and plot_type != "histogram":
                ax.legend(loc='best') # Use single 'ax'
            
            # --- Move Zero Line Logic for Histogram INSIDE Agent Loop --- 
            # # Add a zero line for reference if range crosses zero
            # if min_value < 0 and max_value > 0:
            #     current_ax_for_line = ax # Default to single ax
            #     if plot_type == "histogram":
            #          pass # Zero line added per subplot for histogram
            #     elif plot_type == "box" or plot_type == "running_mean":
            #         current_ax_for_line.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
            # --- Zero Line logic is now handled within the histogram agent loop --- 
            # --- Add Zero Line Logic for Box/Running Mean --- 
            if plot_type != "histogram" and min_value < 0 < max_value:
                 ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
            
            # Adjust layout only if not histogram (histogram uses tight_layout with rect)
            if plot_type != "histogram":
                plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating bootstrap distribution plot for '{statistic_key}': {e}")
            # Print traceback for debugging
            import traceback
            traceback.print_exc()
            return None 
        
# -----------------------------------------------------------------------------
# Backwards-compatibility: legacy API used by `meta_game_analysis`
# -----------------------------------------------------------------------------

# def nonparametric_bootstrap_from_raw_data(
#     all_results,
#     *,
#     num_bootstrap: int = 300,
#     confidence: float = 0.95,
#     global_max_nash_welfare=None,
#     global_max_util_welfare=None,
# ):
#     """Return structure matching the legacy `nonparametric_bootstrap_from_raw_data` API.

#     For historical reasons several modules import this symbol.  The original
#     implementation has been consolidated into :pyfunc:`run_value_bootstrap_agents`.
#     This thin wrapper delegates to that function and adapts its output to the
#     expected ``(bootstrap_results_dict, agent_names)`` signature so that
#     downstream imports succeed without modification.
#     """
#     # Defer heavy dependencies to avoid unnecessary overhead at import-time
#     summary = run_value_bootstrap_agents(
#         all_results,
#         n_boot=num_bootstrap,
#         alpha=1.0 - confidence,
#     )

#     agents = sorted({r.get("agent1") for r in all_results if r.get("agent1")} |
#                     {r.get("agent2") for r in all_results if r.get("agent2")})

#     bootstrap_results = {"statistics": summary}
#     return bootstrap_results, agents
    
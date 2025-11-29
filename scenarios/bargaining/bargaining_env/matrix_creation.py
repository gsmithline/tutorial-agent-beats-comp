#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from collections import defaultdict
import math
import sys
import os
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.metrics import compute_pareto_frontier

def create_performance_matrices(all_results, agent_performance_by_round, agent_game_counts_by_round):
    """
    Create performance matrices from the processed game results, separated by round.
    
    Args:
        all_results: List of all game results (used for some metrics like scaled perf, deal/walk)
        agent_performance_by_round: Dict {round: {agent1: {agent2: mean_value}}} 
        agent_game_counts_by_round: Dict {round: {agent1: {agent2: count}}} 
        
    Returns:
        dict: Dictionary keyed by round (1, 2, 3, 'aggregate'), where each value is a 
              dictionary of performance matrices (performance_matrix, std_dev_matrix, etc.) 
              for that round.
    """
    rounds = list(agent_performance_by_round.keys()) # Should be [1, 2, 3, 'aggregate']
    all_matrices_by_round = {r: {} for r in rounds}

    # --- Determine agent list reliably from counts --- 
    all_agents_from_counts = set()
    agg_counts = agent_game_counts_by_round.get('aggregate', {})
    for p1, opponents in agg_counts.items():
        all_agents_from_counts.add(p1)
        all_agents_from_counts.update(opponents.keys())
    all_agents = sorted(list(all_agents_from_counts))
    if not all_agents:
        print("Error: No agents found in game counts.")
        return {}
    # --- End agent list determination ---

    print(f"Creating performance matrices for {len(all_agents)} agents across rounds: {rounds}")

    for r in rounds:
        print(f"  Processing round: {r}")
        agent_performance = agent_performance_by_round.get(r, {})
        agent_game_counts = agent_game_counts_by_round.get(r, {})

        # Filter all_results for the current round (or use all for aggregate)
        if r == 'aggregate':
            current_round_results = all_results
        else:
            current_round_results = [game for game in all_results if game.get('final_round') == r]
        
        print(f"    - Games in this round/aggregate: {len(current_round_results)}")

        # Initialize matrices for this round
        performance_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        std_dev_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        variance_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        count_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        scaled_performance_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        deal_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        walk_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        rounds_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents) 
        
        actual_walk_when_optimal_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        actual_agree_when_optimal_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
        actual_walk_when_agree_optimal_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)

        for agent1, opponents in agent_performance.items():
            if agent1 not in all_agents: continue 
            for agent2, mean_value in opponents.items():
                if agent2 not in all_agents: continue 
                
                performance_matrix.loc[agent1, agent2] = mean_value
                
                agent1_perf_values = [g['agent1_value'] for g in current_round_results 
                                   if g.get('agent1') == agent1 and g.get('agent2') == agent2 and g.get('agent1_value') is not None]
                agent2_perf_values = [g['agent2_value'] for g in current_round_results 
                                   if g.get('agent1') == agent2 and g.get('agent2') == agent1 and g.get('agent2_value') is not None]
                
                all_agent1_scores = agent1_perf_values + agent2_perf_values # Use all scores for variance
                
                if all_agent1_scores:
                    std_dev_matrix.loc[agent1, agent2] = np.std(all_agent1_scores) if len(all_agent1_scores) > 1 else 0
                    variance_matrix.loc[agent1, agent2] = np.var(all_agent1_scores) if len(all_agent1_scores) > 1 else 0
                else:
                    std_dev_matrix.loc[agent1, agent2] = 0
                    variance_matrix.loc[agent1, agent2] = 0

        for agent1 in all_agents:
            for agent2 in all_agents:
                count = agent_game_counts.get(agent1, {}).get(agent2, 0) # Get count for this specific pair
                count_matrix.loc[agent1, agent2] = count # Assign directly

        for agent1 in all_agents:
            for agent2 in all_agents:
                scaled_values = []
                for result in current_round_results:
                    g_agent1 = result.get('agent1')
                    g_agent2 = result.get('agent2')
                    # Agent1 is Player 1
                    if (g_agent1 == agent1 and g_agent2 == agent2 and 
                        result.get('agent1_value') is not None and result.get('p1_values') is not None and result.get('full_items') is not None):
                        if result['p1_values']:
                            if len(result.get('full_items', [])) == len(result['p1_values']):
                                try:
                                    max_possible = max(
                                        sum(result['full_items'][i] * result['p1_values'][i] for i in range(len(result['p1_values']))),
                                        result.get('p1_outside_offer', 0)
                                    )
                                    if max_possible > 0:
                                        scaled_values.append(result['agent1_value'] / max_possible)
                                except IndexError:
                                    print(f"Warning: IndexError during scaled perf calc for {agent1} vs {agent2}. Skipping game.")
                                    pass
                            else:
                                 print(f"Warning: Mismatched lengths for full_items and p1_values for {agent1} vs {agent2}. Skipping scaled perf calc for this game.")
                        else:
                             print(f"Warning: p1_values is empty for {agent1} vs {agent2}. Skipping scaled perf calc for this game.")
                             
                    # Agent1 is Player 2
                    # The matrix definition performance_matrix.loc[agent1, agent2] means agent1's performance against agent2.
                    # So we only need the cases where agent1 is player 1.

                if scaled_values:
                    scaled_performance_matrix.loc[agent1, agent2] = np.mean(scaled_values)

        # --- Populate Deal/Walk/Rounds --- 
        # Requires iterating through filtered game results for this round
        processed_pairs = set()
        for agent1 in all_agents:
            for agent2 in all_agents:
                pair_key = tuple(sorted((agent1, agent2)))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                agent_pair_games_this_round = [
                    result for result in current_round_results 
                    if ((result.get('agent1') == agent1 and result.get('agent2') == agent2) or 
                        (result.get('agent1') == agent2 and result.get('agent2') == agent1))
                ]
                
                # Filter for games where final action is known and outcome is determined
                valid_outcome_games = [
                    game for game in agent_pair_games_this_round 
                    if game.get('final_action') in ['ACCEPT', 'WALK', 'INVALID WALK']
                ]

                # Filter further for games where optimal walk calculation was successful
                valid_optimal_calc_games = [
                    game for game in valid_outcome_games
                    if game.get('is_walk_optimal') is not None
                ]
                
                valid_count = len(valid_optimal_calc_games) # NEW: Use count of games where optimal calc was possible
                if valid_count > 0:
                    deal_count = sum(1 for game in valid_optimal_calc_games if game.get('final_action') == 'ACCEPT') # NEW
                    walk_count = sum(1 for game in valid_optimal_calc_games if game.get('final_action') == 'WALK') # NEW
                    
                    deal_percentage = (deal_count / valid_count) * 100
                    walk_percentage = (walk_count / valid_count) * 100
                    
                    deal_matrix.loc[agent1, agent2] = deal_matrix.loc[agent2, agent1] = deal_percentage
                    walk_matrix.loc[agent1, agent2] = walk_matrix.loc[agent2, agent1] = walk_percentage

                    # Average rounds for games *ending* in this round/aggregate
                    rounds_values = [game.get('final_round') for game in valid_outcome_games if game.get('final_round') is not None]
                    if rounds_values:
                        avg_rounds = np.mean(rounds_values)
                        rounds_matrix.loc[agent1, agent2] = rounds_matrix.loc[agent2, agent1] = avg_rounds
                    else:
                        # If no valid rounds found for the pair in this specific round filter, set to NaN or r
                        rounds_matrix.loc[agent1, agent2] = rounds_matrix.loc[agent2, agent1] = np.nan # Let's default to NaN 
                
                # --- Calculate new conditional frequencies --- 
                if valid_optimal_calc_games:
                    n_optimal_walk = sum(1 for game in valid_optimal_calc_games if game['is_walk_optimal'])
                    n_actual_walk_when_optimal = sum(1 for game in valid_optimal_calc_games
                                                   if game['is_walk_optimal'] and game['final_action'] == 'WALK') # Exclude INVALID WALK
                    n_actual_walk_when_agree_optimal = sum(1 for game in valid_optimal_calc_games # Count walks when agreeing was optimal
                                                     if not game['is_walk_optimal'] and game['final_action'] == 'WALK') # Exclude INVALID WALK

                    n_optimal_agree = sum(1 for game in valid_optimal_calc_games if not game['is_walk_optimal']) # Use 'not is_walk_optimal'
                    n_actual_agree_when_optimal = sum(1 for game in valid_optimal_calc_games
                                                    if not game['is_walk_optimal'] and game['final_action'] == 'ACCEPT')

                    # Calculate frequencies (handle division by zero)
                    freq_walk_when_optimal = (n_actual_walk_when_optimal / n_optimal_walk * 100) if n_optimal_walk > 0 else np.nan
                    freq_agree_when_optimal = (n_actual_agree_when_optimal / n_optimal_agree * 100) if n_optimal_agree > 0 else np.nan
                    freq_walk_when_agree_optimal = (n_actual_walk_when_agree_optimal / n_optimal_agree * 100) if n_optimal_agree > 0 else np.nan # Calculate new frequency

                    # Populate matrices (symmetric)
                    actual_walk_when_optimal_matrix.loc[agent1, agent2] = actual_walk_when_optimal_matrix.loc[agent2, agent1] = freq_walk_when_optimal
                    actual_agree_when_optimal_matrix.loc[agent1, agent2] = actual_agree_when_optimal_matrix.loc[agent2, agent1] = freq_agree_when_optimal
                    actual_walk_when_agree_optimal_matrix.loc[agent1, agent2] = actual_walk_when_agree_optimal_matrix.loc[agent2, agent1] = freq_walk_when_agree_optimal # Populate the new matrix
                # --- End calculation of new conditional frequencies --- 

        # --- DEBUG: Check specific count_matrix cells before returning ---
        try:
            soft_soft_val = count_matrix.loc['soft_agent_circle_0', 'soft_agent_circle_0']
            tough_soft_val = count_matrix.loc['tough_agent_circle_0', 'soft_agent_circle_0']
            print(f"DEBUG matrix_creation (Round {r}): count_matrix[soft, soft] = {soft_soft_val} (type: {type(soft_soft_val)})")
            print(f"DEBUG matrix_creation (Round {r}): count_matrix[tough, soft] = {tough_soft_val} (type: {type(tough_soft_val)})")
        except KeyError as e:
            print(f"DEBUG matrix_creation (Round {r}): KeyError accessing soft/tough cells - {e}")
        # --- END DEBUG ---

        # Store matrices for this round
        all_matrices_by_round[r] = {
            'performance_matrix': performance_matrix,
            'std_dev_matrix': std_dev_matrix,
            'variance_matrix': variance_matrix,
            'scaled_performance_matrix': scaled_performance_matrix,
            'count_matrix': count_matrix,
            'deal_matrix': deal_matrix,
            'walk_matrix': walk_matrix,
            'rounds_matrix': rounds_matrix,
            # Add new matrices
            'actual_walk_when_optimal_matrix': actual_walk_when_optimal_matrix,
            'actual_agree_when_optimal_matrix': actual_agree_when_optimal_matrix,
            'actual_walk_when_agree_optimal_matrix': actual_walk_when_agree_optimal_matrix, # Add the new matrix here
            # Add overall performance/final rounds derived from this round's data if needed
            # 'overall_agent_performance': {agent: np.nanmean(performance_matrix.loc[agent,:]) for agent in all_agents},
            # 'average_final_rounds': {agent: np.nanmean(rounds_matrix.loc[agent,:]) for agent in all_agents} # This might not be meaningful per round
        }
        print(f"    - Finished matrices for round: {r}")
        # print(all_matrices_by_round[r]['performance_matrix'].head()) # Optional: print head for debugging

    return all_matrices_by_round

def create_welfare_matrices(all_results, all_agents, global_max_nash_welfare, global_max_nash_welfare_adv = None):
    """
    Create matrices for various welfare and fairness metrics, separated by round.
    
    Args:
        all_results: List of all game results
        all_agents: List of all agents (should be consistent across rounds)
        global_max_nash_welfare: Global maximum Nash welfare for normalization
        
    Returns:
        dict: Dictionary keyed by round (1, 2, 3, 'aggregate'), where each value is a 
              dictionary of welfare/game characteristic matrices for that round.
    """
    # Import functions to compute global max values - ensure this is handled if needed
    # from meta_game_analysis.data_processing import compute_global_max_values 
    # Use a fixed value for global maximum utilitarian welfare (1400)
    global_max_utilitarian = 805.9 #1400 805.9

    rounds = [1, 2, 3, 4, 5, 'aggregate']
    all_welfare_matrices_by_round = {r: {} for r in rounds}

    print(f"Creating welfare/characteristic matrices for {len(all_agents)} agents across rounds: {rounds}")

    if not all_agents:
        print("Error: No agents provided to create_welfare_matrices.")
        return {}

    for r in rounds:
        print(f"  Processing round: {r}")

        # Filter all_results for the current round (or use all for aggregate)
        if r == 'aggregate':
            current_round_results = all_results
        else:
            current_round_results = [game for game in all_results if game.get('final_round') == r]
            
        print(f"    - Games in this round/aggregate: {len(current_round_results)}")

        matrix_names = [
            'nash_welfare_matrix', 'nash_welfare_adv_matrix','utilitarian_welfare_matrix', 'percent_utilitarian_matrix',
            'rawls_welfare_matrix', 'mad_matrix', 'gini_matrix', 'variance_welfare_matrix',
            'cv_matrix', 'jain_matrix', 'envy_free_matrix', 'ef1_matrix', 'pareto_matrix',
            'l2_distance_matrix', 'optimal_walk_freq_matrix', 'l1_distance_matrix', 'cosine_similarity_matrix',
            'optimal_agree_freq_matrix', 
            'spearman_rank_correlation_matrix',
            'same_top_item_freq_matrix', 'row_agent_gets_top_item_freq_matrix'
        ]
        round_matrices = {name: pd.DataFrame(np.nan, index=all_agents, columns=all_agents) for name in matrix_names}

        processed_pairs = set()

        for agent1_idx, agent1 in enumerate(all_agents):
            for agent2_idx, agent2 in enumerate(all_agents):
                # Skip if agent not in the main list (shouldn't happen if all_agents is correct)
                if agent1 not in all_agents or agent2 not in all_agents:
                    continue
                
                # Initialize lists for metrics specific to the agent1 vs agent2 interaction
                # (These will be used for non-symmetric matrices like row_agent_gets_top_item)
                row_agent_got_top_item_flags = []
                relevant_deal_count_for_row_agent_metric = 0

                # --- Pair-specific processing (only once per unique pair for symmetric metrics) --- #
                pair_key = tuple(sorted([agent1, agent2]))
                # If this pair hasn't been processed yet for symmetric metrics
                if pair_key not in processed_pairs:
                    processed_pairs.add(pair_key)

                    # Filter games for this specific pair *within the current round's results*
                    agent_pair_games_this_round = [result for result in current_round_results
                                          if ((result.get('agent1') == agent1 and result.get('agent2') == agent2) or
                                              (result.get('agent1') == agent2 and result.get('agent2') == agent1))]
                    
                    # --- DEBUG: Check game counts for soft/tough pairs ---
                    is_soft_tough_pair = ('soft_agent_circle_0' in pair_key or 'tough_agent_circle_0' in pair_key)
                    if is_soft_tough_pair:
                        # print(f"DEBUG welfare (Round {r}, Pair {pair_key}): Found {len(agent_pair_games_this_round)} games.")
                        pass
                    # --- END DEBUG ---

                    # Lists to store metric values for this pair in this round (for symmetric metrics)
                    nash_values, utilitarian_values, rawls_values, mad_values = [], [], [], []
                    gini_values, variance_values, cv_values, jain_values = [], [], [], []
                    nash_values_adv = []
                    envy_free_count, ef1_count, pareto_count = 0, 0, 0
                    valid_allocation_count = 0

                    # Lists for game characteristic metrics (symmetric)
                    l2_distances, l1_distances, cosine_similarities = [], [], []
                    # Changed variable name
                    spearman_rank_correlations = []
                    walk_optimal_flags = []
                    agree_optimal_flags = [] # Add list for agree flags
                    valid_optimal_walk_calc_count = 0
                    same_top_item_flags = [] # List for % same top item metric
                    valid_valuation_count_for_top_item_metric = 0 # Counter for games with valid valuations

                    # Calculate metrics for each game of the pair in this round
                    for game_idx, game in enumerate(agent_pair_games_this_round):
                        # Extract p1/p2 values and items, ensuring correct assignment based on agent1/agent2 roles
                        p1_vals_np, p2_vals_np = None, None
                        p1_items_np, p2_items_np = None, None
                        is_walk_optimal = game.get('is_walk_optimal')

                        # Determine which agent in the game corresponds to agent1 and agent2 of the pair
                        is_agent1_p1 = (game['agent1'] == agent1)

                        p1_values = game.get('p1_values')
                        p2_values = game.get('p2_values')
                        p1_items = game.get('p1_items')
                        p2_items = game.get('p2_items')
                        v1 = game.get('agent1_value')
                        v2 = game.get('agent2_value')
                        p1_outside_offer = game.get('p1_outside_offer')
                        p2_outside_offer = game.get('p2_outside_offer')
                        final_action = game.get('final_action')

                        # --- Calculate Game Characteristic Metrics (Symmetric) ---
                        if p1_values is not None and p2_values is not None:
                            try:
                                p1_vals_np = np.array(p1_values, dtype=float)
                                p2_vals_np = np.array(p2_values, dtype=float)

                                if p1_vals_np.shape == p2_vals_np.shape and p1_vals_np.ndim == 1 and p1_vals_np.size > 0: # Ensure valid 1D arrays of same shape and non-empty
                                    valid_valuation_count_for_top_item_metric += 1
                                    # L2 Distance
                                    l2_dist = np.linalg.norm(p1_vals_np - p2_vals_np)
                                    l2_distances.append(l2_dist)

                                    # L1 Distance
                                    l1_dist = np.sum(np.abs(p1_vals_np - p2_vals_np))
                                    l1_distances.append(l1_dist)

                                    # Cosine Similarity
                                    norm1 = np.linalg.norm(p1_vals_np)
                                    norm2 = np.linalg.norm(p2_vals_np)
                                    if norm1 > 0 and norm2 > 0:
                                        cos_sim = np.dot(p1_vals_np, p2_vals_np) / (norm1 * norm2)
                                        cosine_similarities.append(cos_sim)
                                    else:
                                        cosine_similarities.append(0.0)

                                    # Pearson Rank Correlation (Spearman's rho)
                                    try:
                                        if len(p1_vals_np) >= 2 and (np.ptp(p1_vals_np) > 0 or np.ptp(p2_vals_np) > 0):
                                            correlation, p_value = spearmanr(p1_vals_np, p2_vals_np)
                                            if np.isnan(correlation):
                                                # Changed variable name
                                                spearman_rank_correlations.append(np.nan)
                                            else:
                                                # Changed variable name
                                                spearman_rank_correlations.append(correlation)
                                        elif len(p1_vals_np) >= 1 and np.array_equal(p1_vals_np, p2_vals_np):
                                             # Changed variable name
                                             spearman_rank_correlations.append(1.0)
                                        else:
                                            # Changed variable name
                                            spearman_rank_correlations.append(np.nan)
                                    except Exception as spe:
                                         # Changed variable name in print
                                         print(f"Warning: Error calculating Spearman correlation for {pair_key} in round {r}: {spe}. Appending NaN.")
                                         # Changed variable name
                                         spearman_rank_correlations.append(np.nan)

                                    # NEW METRIC 1: Same Top Item
                                    p1_top_item_idx = np.argmax(p1_vals_np)
                                    p2_top_item_idx = np.argmax(p2_vals_np)
                                    same_top_item_flags.append(p1_top_item_idx == p2_top_item_idx)

                            except Exception as e:
                                print(f"Warning: Error calculating symmetric characteristics for {pair_key} in round {r}: {e}. Skipping game chars.")
                                pass

                        # Optimal Walk Frequency - check if the flag exists
                        if is_walk_optimal is not None:
                            walk_optimal_flags.append(is_walk_optimal)
                            agree_optimal_flags.append(not is_walk_optimal) # Append the opposite flag
                            valid_optimal_walk_calc_count += 1

                        #Calculate Standard Welfare/Fairness Metrics (Symmetric) ---
                        if v1 is not None and v2 is not None:
                            is_on_pareto = game.get('PF', False) # Assume PF calculation was done earlier if present

                            # Calculate standard welfare metrics
                            nash_welfare = np.sqrt(max(0, v1) * max(0, v2)) / global_max_nash_welfare if global_max_nash_welfare > 0 else 0
                            nash_welfare_adv = np.sqrt(max(0, v1 - p1_outside_offer ) * max(0, v2 - p2_outside_offer)) / global_max_nash_welfare_adv if global_max_nash_welfare > 0 else 0

                            # --- DEBUG: Print individual Nash Welfare ---
                            if is_soft_tough_pair:
                                # print(f"    DEBUG NashCalc (Game {game_idx}): v1={v1:.2f}, v2={v2:.2f} -> nash_welfare={nash_welfare:.4f}")
                                pass
                            # --- END DEBUG ---
                            nash_values.append(nash_welfare)
                            nash_values_adv.append(nash_welfare_adv)
                            utilitarian_welfare = v1 + v2
                            rawls_welfare = min(v1, v2)
                            mad = abs(v1 - v2)
                            mean_utility = (v1 + v2) / 2
                            gini = abs(v1 - v2) / (2 * 2 * mean_utility) if mean_utility > 0 else 0
                            variance = ((v1 - mean_utility)**2 + (v2 - mean_utility)**2) / 2
                            cv = np.sqrt(variance) / mean_utility if mean_utility > 0 else 0
                            jain = 1 / (1 + cv**2) if cv is not None and cv != 0 else 1 # Handle cv=0 case

                            utilitarian_values.append(utilitarian_welfare)
                            rawls_values.append(rawls_welfare)
                            mad_values.append(mad)
                            gini_values.append(gini)
                            variance_values.append(variance)
                            cv_values.append(cv)
                            jain_values.append(jain)

                            valid_allocation_count += 1 # Increment count for fairness % later

                            # --- Fairness calculations (Envy, EF1, Pareto) ---
                            if is_on_pareto:
                                 pareto_count += 1

                            # Envy-Freeness and EF1 Check (Level 5)
                            if final_action == "ACCEPT" and all(x is not None for x in [p1_items, p2_items, p1_values, p2_values]):
                                if p1_vals_np is None or p2_vals_np is None: # Recalculate if needed
                                     try:
                                        p1_vals_np = np.array(p1_values, dtype=float)
                                        p2_vals_np = np.array(p2_values, dtype=float)
                                     except: p1_vals_np, p2_vals_np = None, None

                                if p1_vals_np is not None and p2_vals_np is not None:
                                    try:
                                        p1_items_np = np.array(p1_items)
                                        p2_items_np = np.array(p2_items)

                                        if p1_vals_np.shape == p1_items_np.shape and p2_vals_np.shape == p2_items_np.shape and p1_vals_np.ndim == 1:
                                            p1_own_bundle_value = np.dot(p1_vals_np, p1_items_np)
                                            p1_other_bundle_value = np.dot(p1_vals_np, p2_items_np)
                                            p1_is_envy_free = p1_own_bundle_value >= p1_other_bundle_value

                                            p2_own_bundle_value = np.dot(p2_vals_np, p2_items_np)
                                            p2_other_bundle_value = np.dot(p2_vals_np, p1_items_np)
                                            p2_is_envy_free = p2_own_bundle_value >= p2_other_bundle_value

                                            if p1_is_envy_free and p2_is_envy_free:
                                                envy_free_count += 1

                                            # EF1 Check
                                            # p1_is_ef1 = p1_is_envy_free
                                            # if not p1_is_envy_free:
                                            #     for j in range(len(p1_vals_np)):
                                            #         if p2_items_np[j] > 0:
                                            #             val_without_j = p1_other_bundle_value - p1_vals_np[j]
                                            #             if p1_own_bundle_value >= val_without_j:
                                            #                 p1_is_ef1 = True
                                            #                 break

                                            # p2_is_ef1 = p2_is_envy_free
                                            # if not p2_is_envy_free:
                                            #     for j in range(len(p2_vals_np)):
                                            #         if p1_items_np[j] > 0:
                                            #             val_without_j = p2_other_bundle_value - p2_vals_np[j]
                                            #             if p2_own_bundle_value >= val_without_j:
                                            #                 p2_is_ef1 = True
                                            #                 break
                                            envy_gap_p1 = p1_other_bundle_value - p1_own_bundle_value
                                            mask_p2 = p2_items_np > 0
                                            max_p1_item = p1_vals_np[mask_p2].max() if mask_p2.any() else 0.0
                                            p1_is_ef1 = p1_is_envy_free or envy_gap_p1 <= max_p1_item

                                            envy_gap_p2 = p2_other_bundle_value - p2_own_bundle_value
                                            mask_p1 = p1_items_np > 0
                                            max_p2_item = p2_vals_np[mask_p1].max() if mask_p1.any() else 0.0
                                            p2_is_ef1 = p2_is_envy_free or envy_gap_p2 <= max_p2_item


                                            if p1_is_ef1 and p2_is_ef1:
                                                ef1_count += 1
                                    except Exception as e:
                                        print(f"Error during EF/EF1 calc for {pair_key}: {e}")
                                        pass

                    # --- Aggregate SYMMETRIC metrics for the pair in this round --- #
                    def safe_mean(values_list):
                        filtered_list = [v for v in values_list if v is not None and not np.isnan(v)]
                        # For boolean flags, mean directly gives frequency
                        return np.mean(filtered_list) if filtered_list else np.nan

                    # Calculate means/frequencies for existing symmetric metrics
                    if valid_allocation_count > 0:
                         mean_nash = safe_mean(nash_values)
                         mean_nash_adv = safe_mean(nash_values_adv)
                         mean_util = safe_mean(utilitarian_values)
                         mean_rawls = safe_mean(rawls_values)
                         mean_mad = safe_mean(mad_values)
                         mean_gini = safe_mean(gini_values)
                         mean_var = safe_mean(variance_values)
                         mean_cv = safe_mean(cv_values)
                         mean_jain = safe_mean(jain_values)

                         # --- DEBUG: Print Nash list and average ---
                         if is_soft_tough_pair:
                            # print(f"  DEBUG NashAgg (Pair {pair_key}): nash_values list = {nash_values}")
                            pass
                         # --- END DEBUG ---

                         # --- DEBUG: Check Utilitarian values for specific pair ---
                         target_pair_util = {'anthropic_3.7_sonnet_circle_6', 'soft_agent_circle_0'}
                         if set(pair_key) == target_pair_util:
                             # print(f"  DEBUG Util (Pair {pair_key}): utilitarian_values list = {utilitarian_values}")
                             # print(f"  DEBUG Util (Pair {pair_key}): mean_util = {mean_util}")
                             pass
                         # --- END DEBUG ---

                         # Update existing symmetric matrices
                         # --- DEBUG: Print value before assignment --- 
                         if is_soft_tough_pair:
                            # print(f"  DEBUG NashAssign (Pair {pair_key}): Assigning {mean_nash:.4f} to nash_welfare_matrix")
                            pass
                         # --- END DEBUG ---
                         round_matrices['nash_welfare_matrix'].loc[agent1, agent2] = round_matrices['nash_welfare_matrix'].loc[agent2, agent1] = mean_nash
                         round_matrices['nash_welfare_adv_matrix'].loc[agent1, agent2] = round_matrices['nash_welfare_adv_matrix'].loc[agent2, agent1] = mean_nash_adv
                         round_matrices['utilitarian_welfare_matrix'].loc[agent1, agent2] = round_matrices['utilitarian_welfare_matrix'].loc[agent2, agent1] = mean_util
                         round_matrices['rawls_welfare_matrix'].loc[agent1, agent2] = round_matrices['rawls_welfare_matrix'].loc[agent2, agent1] = mean_rawls
                         round_matrices['mad_matrix'].loc[agent1, agent2] = round_matrices['mad_matrix'].loc[agent2, agent1] = mean_mad
                         round_matrices['gini_matrix'].loc[agent1, agent2] = round_matrices['gini_matrix'].loc[agent2, agent1] = mean_gini
                         round_matrices['variance_welfare_matrix'].loc[agent1, agent2] = round_matrices['variance_welfare_matrix'].loc[agent2, agent1] = mean_var
                         round_matrices['cv_matrix'].loc[agent1, agent2] = round_matrices['cv_matrix'].loc[agent2, agent1] = mean_cv
                         round_matrices['jain_matrix'].loc[agent1, agent2] = round_matrices['jain_matrix'].loc[agent2, agent1] = mean_jain

                         # Calculate percent utilitarian
                         percent_util = np.nan # Default to NaN
                         if not np.isnan(mean_util) and global_max_utilitarian > 0:
                            percent_util = (mean_util / global_max_utilitarian) * 100
                            # --- DEBUG: Check Percent Util calc & Assignment ---
                            if set(pair_key) == target_pair_util:
                                # print(f"  DEBUG Util% (Pair {pair_key}): Calculated percent_util = {percent_util}")
                                # print(f"  DEBUG Util% (Pair {pair_key}): Assigning {percent_util:.2f} to percent_utilitarian_matrix")
                                pass
                            # --- END DEBUG ---
                            round_matrices['percent_utilitarian_matrix'].loc[agent1, agent2] = round_matrices['percent_utilitarian_matrix'].loc[agent2, agent1] = percent_util
                         elif set(pair_key) == target_pair_util: # Log if calculation failed
                             # print(f"  DEBUG Util% (Pair {pair_key}): Calculation skipped (mean_util={mean_util}, global_max={global_max_utilitarian})")
                             pass

                         # Calculate and store fairness percentages
                         ef_value = (envy_free_count / valid_allocation_count) * 100 if valid_allocation_count > 0 else np.nan
                         ef1_value = (ef1_count / valid_allocation_count) * 100 if valid_allocation_count > 0 else np.nan
                         pareto_value = (pareto_count / valid_allocation_count) * 100 if valid_allocation_count > 0 else np.nan
                         round_matrices['envy_free_matrix'].loc[agent1, agent2] = round_matrices['envy_free_matrix'].loc[agent2, agent1] = ef_value
                         round_matrices['ef1_matrix'].loc[agent1, agent2] = round_matrices['ef1_matrix'].loc[agent2, agent1] = ef1_value
                         round_matrices['pareto_matrix'].loc[agent1, agent2] = round_matrices['pareto_matrix'].loc[agent2, agent1] = pareto_value


                    # Calculate and store means/frequency for SYMMETRIC game characteristic metrics
                    mean_l2_dist = safe_mean(l2_distances)
                    mean_l1_dist = safe_mean(l1_distances)
                    mean_cos_sim = safe_mean(cosine_similarities)
                    # Changed variable name
                    mean_spearman_rank_corr = safe_mean(spearman_rank_correlations)
                    mean_same_top_item_freq = safe_mean(same_top_item_flags) * 100 if valid_valuation_count_for_top_item_metric > 0 else np.nan # Calculate freq for metric 1

                    if valid_optimal_walk_calc_count > 0:
                        optimal_walk_freq = (sum(walk_optimal_flags) / valid_optimal_walk_calc_count) * 100
                        optimal_agree_freq = (sum(agree_optimal_flags) / valid_optimal_walk_calc_count) * 100 # Calculate agree freq
                    else:
                        optimal_walk_freq = np.nan
                        optimal_agree_freq = np.nan # Set agree freq to NaN if no valid calcs

                    # Update SYMMETRIC matrices
                    round_matrices['l2_distance_matrix'].loc[agent1, agent2] = round_matrices['l2_distance_matrix'].loc[agent2, agent1] = mean_l2_dist
                    round_matrices['l1_distance_matrix'].loc[agent1, agent2] = round_matrices['l1_distance_matrix'].loc[agent2, agent1] = mean_l1_dist
                    round_matrices['cosine_similarity_matrix'].loc[agent1, agent2] = round_matrices['cosine_similarity_matrix'].loc[agent2, agent1] = mean_cos_sim
                    # Changed matrix key and variable name
                    round_matrices['spearman_rank_correlation_matrix'].loc[agent1, agent2] = round_matrices['spearman_rank_correlation_matrix'].loc[agent2, agent1] = mean_spearman_rank_corr
                    round_matrices['optimal_walk_freq_matrix'].loc[agent1, agent2] = round_matrices['optimal_walk_freq_matrix'].loc[agent2, agent1] = optimal_walk_freq
                    round_matrices['optimal_agree_freq_matrix'].loc[agent1, agent2] = round_matrices['optimal_agree_freq_matrix'].loc[agent2, agent1] = optimal_agree_freq # Populate agree freq matrix
                    round_matrices['same_top_item_freq_matrix'].loc[agent1, agent2] = round_matrices['same_top_item_freq_matrix'].loc[agent2, agent1] = mean_same_top_item_freq # Update symmetric matrix for metric 1

                # --- Calculate NON-SYMMETRIC metrics for agent1 vs agent2 (row vs column) --- #
                # We need to iterate through games again, focusing on agent1's perspective
                agent1_vs_agent2_games = [result for result in current_round_results
                                        if result.get('agent1') == agent1 and result.get('agent2') == agent2]

                for game in agent1_vs_agent2_games:
                     if game['final_action'] == "ACCEPT":
                         p1_values = game.get('p1_values')
                         p1_items = game.get('p1_items')
                         if p1_values is not None and p1_items is not None:
                             try:
                                 p1_vals_np = np.array(p1_values, dtype=float)
                                 p1_items_np = np.array(p1_items)
                                 if p1_vals_np.ndim == 1 and p1_vals_np.size > 0 and p1_vals_np.shape == p1_items_np.shape:
                                     relevant_deal_count_for_row_agent_metric += 1
                                     agent1_top_item_idx = np.argmax(p1_vals_np)
                                     # Check if agent1 received *any* amount of their top item
                                     agent1_got_top_item = (p1_items_np[agent1_top_item_idx] > 0)
                                     row_agent_got_top_item_flags.append(agent1_got_top_item)
                             except Exception as e:
                                 print(f"Warning: Error processing row_agent_gets_top_item for {agent1} vs {agent2}: {e}")
                                 pass # Skip this game for this metric if error

                # Calculate frequency for the non-symmetric metric
                if relevant_deal_count_for_row_agent_metric > 0:
                    freq_row_agent_got_top = (safe_mean(row_agent_got_top_item_flags)) * 100
                else:
                    freq_row_agent_got_top = np.nan # No relevant deals found

                # Populate the non-symmetric matrix
                round_matrices['row_agent_gets_top_item_freq_matrix'].loc[agent1, agent2] = freq_row_agent_got_top

                # --- DEBUG: Print final counts before percentage calculation ---
                if is_soft_tough_pair:
                    # print(f"  DEBUG welfare (Pair {pair_key}): Final Counts: valid_allocation={valid_allocation_count}, envy_free={envy_free_count}")
                    pass
                # --- END DEBUG ---

        # Store the completed matrices for this round
        all_welfare_matrices_by_round[r] = round_matrices
        print(f"    - Finished matrices for round: {r}")
        # print(all_welfare_matrices_by_round[r]['utilitarian_welfare_matrix'].head()) # Optional debug print

    return all_welfare_matrices_by_round

def clean_matrix_names(matrix, get_display_name_func):
    """
    Clean matrix index and column names using the get_display_name function.
    
    Args:
        matrix: DataFrame to clean
        get_display_name_func: Function to convert agent names to display names
        
    Returns:
        DataFrame: Matrix with cleaned names
    """
    clean_matrix = matrix.copy()
    
    clean_matrix.index = [get_display_name_func(agent) for agent in matrix.index]
    clean_matrix.columns = [get_display_name_func(agent) for agent in matrix.columns]
    
    return clean_matrix

def filter_matrices(matrices, exclude_agents=None):
    """
    Filter a dictionary of matrices to exclude specified agents.
    
    Args:
        matrices: Dictionary of DataFrames
        exclude_agents: List of agents to exclude
        
    Returns:
        dict: Filtered dictionary of DataFrames
    """
    if exclude_agents is None:
        exclude_agents = []
        
    filtered_matrices = {}
    for name, matrix in matrices.items():
        filtered_matrix = matrix
        for agent in exclude_agents:
            filtered_matrix = filtered_matrix[filtered_matrix.index != agent].drop(columns=[agent], errors='ignore')
        filtered_matrices[name] = filtered_matrix
        
    return filtered_matrices

def create_meta_game_performance_matrices(all_results, agent_performance_by_round, agent_game_counts_by_round):
    """
    Create meta-game performance matrices by combining three underlying game matrices with equal weights.
    This is used when full_game_mix=True.
    
    Args:
        all_results: List of all game results (with 'source_matrix' field)
        agent_performance_by_round: Dict {round: {agent1: {agent2: mean_value}}} 
        agent_game_counts_by_round: Dict {round: {agent1: {agent2: count}}} 
        
    Returns:
        dict: Dictionary keyed by round, containing combined meta-game matrices
    """
    print("\nCreating meta-game performance matrices with equal weighting...")
    
    results_by_source = {
       'game_matrix_1a': [],
       'game_matrix_2a': [],
        'game_matrix_3a': [],
        # 'game_matrix_1': [],
        #  'game_matrix_2': [],
        #  'game_matrix_3': []
    }
    
    for result in all_results:
        source = result.get('source_matrix')
        if source in results_by_source:
            results_by_source[source].append(result)

    print(f"Games per matrix: game_matrix_1={len(results_by_source['game_matrix_1a'])}, "
          f"game_matrix_2={len(results_by_source['game_matrix_2a'])}, "
          f"game_matrix_3={len(results_by_source['game_matrix_3a'])}")

    # print(f"Games per matrix: game_matrix_1={len(results_by_source['game_matrix_1'])}, "
    #       f"game_matrix_2={len(results_by_source['game_matrix_2'])}, "
    #       f"game_matrix_3={len(results_by_source['game_matrix_3'])}")
    
    all_agents_from_counts = set()
    agg_counts = agent_game_counts_by_round.get('aggregate', {})
    for p1, opponents in agg_counts.items():
        all_agents_from_counts.add(p1)
        all_agents_from_counts.update(opponents.keys())
    all_agents = sorted(list(all_agents_from_counts))
    
    rounds = list(agent_performance_by_round.keys())
    
    # Create separate performance matrices for each source
    matrices_by_source = {}
    for source in ['game_matrix_1a', 'game_matrix_2a', 'game_matrix_3a']:
        # Create agent performance dict for this source
        source_agent_performance_by_round = {r: defaultdict(lambda: defaultdict(list)) for r in rounds}
        
        for result in results_by_source[source]:
            agent1 = result.get('agent1')
            agent2 = result.get('agent2')
            final_round_num = result.get('final_round')
            
            if agent1 is None or agent2 is None:
                continue
                
            if final_round_num not in [1, 2, 3, 4, 5]:
                round_key = None
            else:
                round_key = final_round_num
            
            # Add to aggregate
            if result.get('agent1_value') is not None:
                source_agent_performance_by_round['aggregate'][agent1][agent2].append(result['agent1_value'])
            if result.get('agent2_value') is not None:
                source_agent_performance_by_round['aggregate'][agent2][agent1].append(result['agent2_value'])
            
            # Add to specific round
            if round_key is not None:
                if result.get('agent1_value') is not None:
                    source_agent_performance_by_round[round_key][agent1][agent2].append(result['agent1_value'])
                if result.get('agent2_value') is not None:
                    source_agent_performance_by_round[round_key][agent2][agent1].append(result['agent2_value'])
        
        # Convert lists to means
        source_agent_perf_means = {r: defaultdict(lambda: defaultdict(lambda: np.nan)) for r in rounds}
        for r in rounds:
            for agent1, opponents in source_agent_performance_by_round[r].items():
                for agent2, values in opponents.items():
                    if values:
                        source_agent_perf_means[r][agent1][agent2] = np.mean(values)
        
        # Create performance matrix for this source
        source_matrices = {}
        for r in rounds:
            perf_matrix = pd.DataFrame(np.nan, index=all_agents, columns=all_agents)
            for agent1 in all_agents:
                for agent2 in all_agents:
                    if agent1 in source_agent_perf_means[r] and agent2 in source_agent_perf_means[r][agent1]:
                        perf_matrix.loc[agent1, agent2] = source_agent_perf_means[r][agent1][agent2]
            source_matrices[r] = perf_matrix
        
        matrices_by_source[source] = source_matrices
    
    # Now create the combined meta-game matrices
    combined_matrices_by_round = {r: {} for r in rounds}
    
    for r in rounds:
        # Initialize combined matrix
        combined_matrix = pd.DataFrame(0.0, index=all_agents, columns=all_agents)
        count_matrix = pd.DataFrame(0, index=all_agents, columns=all_agents)
        
        # Add weighted contributions from each source
        for source in ['game_matrix_1a', 'game_matrix_2a', 'game_matrix_3a']:
            source_matrix = matrices_by_source[source][r]
            
            for agent1 in all_agents:
                for agent2 in all_agents:
                    val = source_matrix.loc[agent1, agent2]
                    if not np.isnan(val):
                        # Add 1/3 of the value from this source
                        combined_matrix.loc[agent1, agent2] += val / 3.0
                        count_matrix.loc[agent1, agent2] += 1
        
        
        for agent1 in all_agents:
            for agent2 in all_agents:
                count = count_matrix.loc[agent1, agent2]
                if count > 0 and count < 3:
                  
                    current_sum = combined_matrix.loc[agent1, agent2]
                    combined_matrix.loc[agent1, agent2] = current_sum * 3.0 / count
                elif count == 0:
                    combined_matrix.loc[agent1, agent2] = np.nan
        
        combined_matrices_by_round[r]['performance_matrix'] = combined_matrix
        combined_matrices_by_round[r]['source_count_matrix'] = count_matrix  # Track how many sources had data
        
        print(f"  Round {r}: Combined matrix created with shape {combined_matrix.shape}")
    
    return combined_matrices_by_round

def create_meta_game_welfare_matrices(all_results, all_agents, global_max_nash_welfare, global_max_nash_welfare_adv):
    """
    Create meta-game welfare matrices by combining three underlying game matrices with equal weights.
    
    Args:
        all_results: List of all game results (with 'source_matrix' field)
        all_agents: List of all agents
        global_max_nash_welfare: Global maximum Nash welfare for normalization
        
    Returns:
        dict: Dictionary keyed by round, containing combined welfare matrices
    """
    print("\nCreating meta-game welfare matrices with equal weighting...")
    
    results_by_source = {
        'game_matrix_1a': [],
        'game_matrix_2a': [],
        'game_matrix_3a': [],
        #  'game_matrix_1': [],
        #  'game_matrix_2': [],
        #  'game_matrix_3': []
    }
    
    for result in all_results:
        source = result.get('source_matrix')
        if source in results_by_source:
            results_by_source[source].append(result)
    
    rounds = [1, 2, 3, 4, 5, 'aggregate']
    
    welfare_matrices_by_source = {}
    for source in ['game_matrix_1a', 'game_matrix_2a', 'game_matrix_3a']:
        source_welfare_matrices = create_welfare_matrices(
            results_by_source[source], 
            all_agents, 
            global_max_nash_welfare,
            global_max_nash_welfare_adv=global_max_nash_welfare_adv

        )
        welfare_matrices_by_source[source] = source_welfare_matrices
    
    combined_welfare_by_round = {r: {} for r in rounds}

    matrix_names = list(welfare_matrices_by_source['game_matrix_1']['aggregate'].keys())

    for r in rounds:
        for matrix_name in matrix_names:
            combined_matrix = pd.DataFrame(0.0, index=all_agents, columns=all_agents)
            count_matrix = pd.DataFrame(0, index=all_agents, columns=all_agents)

            for source in ['game_matrix_1a', 'game_matrix_2a', 'game_matrix_3a']:
                if r in welfare_matrices_by_source[source] and matrix_name in welfare_matrices_by_source[source][r]:
                    source_matrix = welfare_matrices_by_source[source][r][matrix_name]
                    
                    for agent1 in all_agents:
                        for agent2 in all_agents:
                            if agent1 in source_matrix.index and agent2 in source_matrix.columns:
                                val = source_matrix.loc[agent1, agent2]
                                if not pd.isna(val):
                                    combined_matrix.loc[agent1, agent2] += val / 3.0
                                    count_matrix.loc[agent1, agent2] += 1
            
            for agent1 in all_agents:
                for agent2 in all_agents:
                    count = count_matrix.loc[agent1, agent2]
                    if count > 0 and count < 3:
                        current_sum = combined_matrix.loc[agent1, agent2]
                        combined_matrix.loc[agent1, agent2] = current_sum * 3.0 / count
                    elif count == 0:
                        combined_matrix.loc[agent1, agent2] = np.nan
            
            combined_welfare_by_round[r][matrix_name] = combined_matrix
        
        print(f"  Round {r}: Created {len(matrix_names)} welfare matrices")
    
    return combined_welfare_by_round 
#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from scipy.stats import f_oneway 
import math
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison # Import for Tukey HSD

os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser('~'), '.matplotlib_temp')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_game_analysis.data_processing import (
    process_all_games,
    get_display_name,
    compute_global_max_values,
    plot_frobenius_norm_evolution,
    plot_metric_evolution
)
from meta_game_analysis.matrix_creation import (
    create_performance_matrices,
    create_welfare_matrices,
    clean_matrix_names,
    filter_matrices,
    create_meta_game_performance_matrices,
    create_meta_game_welfare_matrices
)
from meta_game_analysis.visualization import (
    create_best_response_graph,
    create_complete_best_response_graph,
    visualize_welfare_matrices,
    visualize_nash_equilibrium,
    visualize_rd_regret_heatmaps,
    visualize_nash_comparison,
    save_results_to_csv,
    create_average_best_response_graph,
    visualize_matrix,
    visualize_metric_comparison
)
#from meta_game_analysis.temporal_analysis import (
   # analyze_temporal_bargaining_patterns,
    # analyze_bargaining_by_agent_pair
#)
from meta_game_analysis.nash_analysis import (
    run_nash_analysis,
    run_raw_data_nash_analysis,
    plot_nash_distributions,
    save_nash_plots,
    print_nash_summary,
    calculate_acceptance_ratio,
    print_pure_nash_info,
    find_nash_with_replicator_dynamics,
    generate_all_nash_stats,
    print_rd_nash_summary,
    print_nash_comparison
)
from nash_equilibrium.nash_regret_viz import create_matrix_heatmap_with_nash_regret
from meta_game_analysis.bootstrap_nonparametric import run_bootstrap_analysis, analyze_bootstrap_convergence, plot_regret_distributions
from meta_game_analysis.nash_analysis import analyze_bootstrap_results
from meta_game_analysis.bootstrap_nonparametric import plot_regret_distributions, plot_ci_size_evolution, plot_bootstrap_distributions
from meta_game_analysis.bootstrap_nonparametric import analyze_bootstrap_results_for_convergence
from nash_equilibrium.nash_solver import milp_max_sym_ent_2p
#from meta_game_analysis.data_convergence import track_data_convergence
import traceback

def run_anova_analysis(all_results, all_agents, metric_key, metric_name):
    '''
    This function does basic Statistical Analysis between the agents to tell if the
    observed outcomes are statistically significant.
    '''
    print(f"\n--- Running ANOVA for: {metric_name} ---")
    agent_metric_values = defaultdict(list)
    games_counted = 0
    
    for game in all_results:
        metric_value = game.get(metric_key)
        agent1 = game.get('agent1')
        agent2 = game.get('agent2')
        
        if metric_value is not None and isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
            games_counted += 1
            if agent1 in all_agents:
                agent_metric_values[agent1].append(metric_value)
            if agent2 in all_agents: 
                agent_metric_values[agent2].append(metric_value) 
                
    print(f"Total valid data points for {metric_name}: {games_counted}")

    data_for_anova = []
    agents_in_anova = []
    for agent in all_agents:
        values = agent_metric_values.get(agent, [])
        if len(values) >= 2:
            data_for_anova.append(values)
            agents_in_anova.append(get_display_name(agent))
        else:
            print(f"Excluding agent {get_display_name(agent)} from {metric_name} ANOVA (found {len(values)} data points, need >= 2)")

    if len(data_for_anova) < 2:
        print(f"Cannot perform ANOVA for {metric_name}: Need at least 2 agents with sufficient data.")
        return

    try:
        f_statistic, p_value = f_oneway(*data_for_anova)
        print(f"ANOVA Results for {metric_name} (comparing {len(agents_in_anova)} agents):")
        print(f"F-statistic: {f_statistic:.4f}")
        print(f"P-value: {p_value:.4g}") 
        if p_value < 0.05:
            print("Result: Statistically significant difference found between agent means (p < 0.05).")
            
            print(f"Running Tukey HSD test for {metric_name}")
            try:
                all_scores = []
                group_labels = []
                for agent in all_agents:
                    values = agent_metric_values.get(agent, [])
                    if len(values) >= 2: # Ensure agent was included in ANOVA
                        agent_display_name = get_display_name(agent)
                        all_scores.extend(values)
                        group_labels.extend([agent_display_name] * len(values))
                
                if len(all_scores) > 0 and len(group_labels) == len(all_scores):
                    mc = MultiComparison(np.array(all_scores), group_labels)
                    tukey_result = mc.tukeyhsd()
                    
                    print("Tukey HSD Post-hoc Results:")
                    print(tukey_result)
                    # summary_df = pd.read_html(tukey_result.summary().as_html(), header=0, index_col=0)[0]
                    # summary_df.to_csv(f"tukey_hsd_{metric_name.replace(' ', '_')}.csv")
                else:
                    print("Could not prepare data for Tukey HSD test.")

            except Exception as tukey_e:
                print(f"Error running Tukey HSD for {metric_name}: {tukey_e}")
       
                
        else:
            print("    Result: No statistically significant difference found between agent means (p >= 0.05).")
    except Exception as e:
        print(f"  Error running ANOVA for {metric_name}: {e}")
    print("---")

def print_sample_game_data(games, sample_size=1):
    """
    Print the structure of a sample game to help debug extraction issues.
    Args:
        games: List of games
        sample_size: Number of games to sample
    """
    if not games:
        print("No games to sample")
        return
    
    sample = games[:sample_size]
    for i, game in enumerate(sample):
        print(f"\n=== Sample Game {i+1} ===")
        
        print(f"Top-level keys: {list(game.keys())}")
        
        if 'round_data' in game and game['round_data']:
            print(f"Number of rounds: {len(game['round_data'])}")
            
            if len(game['round_data']) > 0:
                first_round = game['round_data'][0]
                print(f"First round keys: {list(first_round.keys())}")
                
                if 'offers' in first_round and first_round['offers']:
                    print(f"Number of offers in first round: {len(first_round['offers'])}")
                    
                    if len(first_round['offers']) > 0:
                        first_offer = first_round['offers'][0]
                        print(f"First offer keys: {list(first_offer.keys())}")
                        
                        if 'utilities' in first_offer:
                            print(f"Utilities in first offer: {first_offer['utilities']}")
                        elif 'utility' in first_offer:
                            print(f"Utility in first offer: {first_offer['utility']}")
                        
                if 'utilities' in first_round:
                    print(f"Direct utilities in first round: {first_round['utilities']}")
                if 'p1_utility' in first_round:
                    print(f"P1 utility in first round: {first_round['p1_utility']}")
                if 'p2_utility' in first_round:
                    print(f"P2 utility in first round: {first_round['p2_utility']}")
        else:
            print("No round_data found")

def load_game_data_from_files(input_dir):
    """
    Load and preprocess game data from JSON files in the input directory.
    Handles both direct game format and nested 'all_game_data' format.
    
    Args:
        input_dir: Directory containing game data files
        
    Returns:
        List of game data objects
    """
    print(f"Loading game data from {input_dir}...")
    all_results = []
    files_processed = 0
    files_with_nested_data = 0
    total_games_loaded = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    files_processed += 1
                    games_in_file = 0
                    
                    if 'all_game_data' in data and isinstance(data['all_game_data'], list):
                        games_in_file = len(data['all_game_data'])
                        all_results.extend(data['all_game_data'])
                        files_with_nested_data += 1
                        print(f"Found nested game data in {file_path}: {games_in_file} games")
                    else:
                        if 'matches' in data and isinstance(data['matches'], list):
                            for match in data['matches']:
                                if isinstance(match, dict) and 'samples' in match and isinstance(match['samples'], list):
                                    for sample in match['samples']:
                                        all_results.append(sample)
                                        games_in_file += 1
                            if games_in_file > 0:
                                print(f"Found complex nested data in {file_path}: {games_in_file} games")
                        elif isinstance(data, list):
                            all_results.extend(data)
                            games_in_file = len(data)
                            print(f"Found direct list of games in {file_path}: {games_in_file} games")
                        else:
                            all_results.append(data)
                            games_in_file = 1
                    total_games_loaded += games_in_file
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    print(f"Processed {files_processed} files ({files_with_nested_data} with nested data)")
    print(f"Loaded {len(all_results)} total game records")
    
    valid_games = []
    for game in all_results:
        has_round_data = 'round_data' in game and isinstance(game['round_data'], list)
        has_final_data = 'agent1' in game and 'agent2' in game
        
        if has_round_data or has_final_data:
            valid_games.append(game)
    
    print(f"Found {len(valid_games)} valid games for analysis")
    return valid_games

def run_analysis(input_dir="crossplay/game_matrix_2a", output_dir="meta_game_analysis/results", 
                 num_bootstrap=100, confidence=0.95, global_samples=1000, 
                 use_raw_bootstrap=False, discount_factor=0.9,
                 track_br_evolution=False, batch_size=100, num_br_bootstraps=100, random_seed=42,
                 use_cell_based_bootstrap=False, run_temporal_analysis=False, full_game_mix=True):
    """
    Run the full meta-game analysis pipeline.
    
    Args:
        input_dir: Directory containing game data files
        output_dir: Directory to save results
        num_bootstrap: Number of bootstrap samples for Nash analysis
        confidence: Confidence level for bootstrap intervals
        global_samples: Number of samples for computing global max values
        use_raw_bootstrap: Whether to use raw game data for non-parametric bootstrapping
        discount_factor: Discount factor for utility calculations (gamma)
        track_br_evolution: Whether to track the evolution of best response graphs
        batch_size: Number of games in first batch, with subsequent batches being multiples (n, 2n, 3n)
        num_br_bootstraps: Number of bootstrap samples for each batch in BR analysis
        random_seed: Fixed random seed for reproducibility
        use_cell_based_bootstrap: Whether to use cell-based bootstrapping instead of game-based
        run_temporal_analysis: Whether to run temporal bargaining analysis
        full_game_mix: Boolean flag to return a game matrix that is mix of all game matrices
    """
    print(f"Starting meta-game analysis on {input_dir}")
    print(f"Results will be saved to {output_dir}")
    print(f"Using discount factor (gamma): {discount_factor}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nStep 1: Processing game data...")
    
    processed_data = process_all_games(
        input_dir,
        discount_factor=discount_factor,
        batch_size=batch_size if track_br_evolution else 100,
        num_bootstraps=num_br_bootstraps if track_br_evolution else 100,
        track_br_evolution=track_br_evolution,
        random_seed=random_seed,
        use_cell_based_bootstrap=use_cell_based_bootstrap, 
        full_game_mix=full_game_mix
    )
    
    if track_br_evolution:
        (all_results, agent_performance_by_round, agent_final_rounds_by_round,
         agent_game_counts_by_round, agent_final_rounds_self_play_by_round, br_evolution_data) = processed_data
        br_evolution_dir = os.path.join(output_dir, 'br_evolution')
        os.makedirs(br_evolution_dir, exist_ok=True)
        
        print("\nDebugging BR evolution data:")
        print(f"Number of batches: {len(br_evolution_data['batch_numbers'])}")
        print(f"Frobenius norms: {br_evolution_data['frobenius_norms']}")
        
        if all(abs(norm) < 1e-6 for norm in br_evolution_data["frobenius_norms"]):
            print("WARNING: All Frobenius norms are zero or nearly zero. This suggests an issue with the bootstrapping.")
            
            if len(br_evolution_data["avg_br_matrices"]) >= 2:
                sample_diff = np.abs(br_evolution_data["avg_br_matrices"][0] - br_evolution_data["avg_br_matrices"][1])
                print(f"Sample matrix difference (first two batches):\n{sample_diff}")
                print(f"Sum of absolute differences: {np.sum(sample_diff)}")
        
        plot_frobenius_norm_evolution(
            br_evolution_data["batch_numbers"],
            br_evolution_data["frobenius_norms"],
            output_dir=br_evolution_dir,
            filename=f"br_graph_evolution_seed{random_seed}.png"
        )
        
        if br_evolution_data["avg_br_matrices"] and len(br_evolution_data["avg_br_matrices"]) > 0:
            if "all_known_agents" in br_evolution_data:
                agent_names_br = br_evolution_data["all_known_agents"]
            else:
                agent_names_br = sorted(list(agent_performance_by_round['aggregate'].keys()))
                print(f"WARNING: Using {len(agent_names_br)} agents from performance data for visualization. This may not match matrix dimensions.")
                
                matrix_shape = br_evolution_data["avg_br_matrices"][0].shape
                print(f"Matrix dimensions: {matrix_shape}")
                
                if len(agent_names_br) != matrix_shape[0]:
                    print(f"Agent list length ({len(agent_names_br)}) doesn't match matrix dimensions ({matrix_shape[0]})")
                    print("Will use generic agent names to avoid visualization errors")
                    agent_names_br = [f"Agent_{i}" for i in range(matrix_shape[0])]
            
            agent_display_names_br = [get_display_name(name) for name in agent_names_br]

            batches_to_visualize = []
            
            batches_to_visualize = list(range(len(br_evolution_data["batch_numbers"])))
            
            
            for idx in batches_to_visualize:
                batch_num = br_evolution_data["batch_numbers"][idx]
                avg_br_matrix = br_evolution_data["avg_br_matrices"][idx]
                #if avg_br_matrix.shape(3, 3):

                try:
                    create_average_best_response_graph(
                        avg_br_matrix, 
                        agent_display_names_br, 
                        #["Anthropic", "Gemini", "OpenAI"],     

                        filename=f'avg_br_graph_batch_{batch_num}',
                        save_dir=br_evolution_dir
                    )
                except Exception as e:
                    print(f"Warning: Could not create visualization for batch {batch_num}: {e}")
        
        print(f"Best response graph evolution analysis saved to {br_evolution_dir}")
    else:
        (all_results, agent_performance_by_round, agent_final_rounds_by_round,
         agent_game_counts_by_round, agent_final_rounds_self_play_by_round) = processed_data
        br_evolution_data = None
    
    print("\nStep 2: Computing global maximum values...")
    #global_max_nash_welfare, global_standard_max = compute_global_max_values(num_samples=global_samples)
    global_max_nash_welfare = 378.7  #378.7  610.23378.7
    global_standard_max = 805.9  #  1400 805.9
    global_max_nash_welfare_adv = 81.7 #81.7
    print(f"Global max Nash welfare: {global_max_nash_welfare:.2f}")
    print(f"Global max social welfare: {global_standard_max:.2f}")
    
    print("\nStep 3: Creating performance matrices (by round)...")
    if full_game_mix:
        # Use the new meta-game matrix creation with equal weighting
        performance_matrices_by_round = create_meta_game_performance_matrices(
            all_results, 
            agent_performance_by_round, 
            agent_game_counts_by_round
        )
    else:
        performance_matrices_by_round = create_performance_matrices(
            all_results, 
            agent_performance_by_round, 
            agent_game_counts_by_round
        )
    
    all_agents = sorted(list(performance_matrices_by_round['aggregate']['performance_matrix'].index))
    print(f"Agents considered for analysis: {len(all_agents)}")

    print("\nStep 4: Creating welfare matrices (by round)...")
    if full_game_mix: 
        welfare_matrices_by_round = create_meta_game_welfare_matrices(
            all_results, 
            all_agents, 
            global_max_nash_welfare,
            global_max_nash_welfare_adv
        )
    else:
        welfare_matrices_by_round = create_welfare_matrices(
            all_results, 
            all_agents, 
            global_max_nash_welfare,
            global_max_nash_welfare_adv
        )
    
    
    print("\nStep 5 & 6: Cleaning and filtering matrices (by round)...")
    rounds_to_process = list(performance_matrices_by_round.keys()) 
    
    all_matrices_by_round = {r: {} for r in rounds_to_process}
    for r in rounds_to_process:
        all_matrices_by_round[r].update(performance_matrices_by_round.get(r, {}))
        all_matrices_by_round[r].update(welfare_matrices_by_round.get(r, {}))

    cleaned_matrices_by_round = {r: {} for r in rounds_to_process}
    for r in rounds_to_process:
        print(f"  Cleaning round: {r}")
        for name, matrix in all_matrices_by_round[r].items():
            if matrix is not None and not matrix.empty:
                 try:
                     cleaned_matrices_by_round[r][name] = clean_matrix_names(matrix, get_display_name)
                 except Exception as e:
                      print(f"Warning: Could not clean matrix {name} for round {r}: {e}")
                      cleaned_matrices_by_round[r][name] = matrix 
            else:
                 cleaned_matrices_by_round[r][name] = matrix 

    filtered_matrices_by_round = {r: {} for r in rounds_to_process}
    exclude_agents = ["gemini_2.0_flash_circle_6"] 
    for r in rounds_to_process:
         print(f"  Filtering round: {r}")
         if exclude_agents:
            filtered_matrices_by_round[r] = filter_matrices(cleaned_matrices_by_round[r], exclude_agents)
         else:
            filtered_matrices_by_round[r] = cleaned_matrices_by_round[r] 

    print("\nStep 7: Running Nash equilibrium analysis (on AGGREGATE data)...")
    aggregate_performance_matrix = filtered_matrices_by_round['aggregate'].get('performance_matrix')

    if aggregate_performance_matrix is None or aggregate_performance_matrix.empty:
         print("Error: Aggregate performance matrix not found or empty. Cannot run Nash analysis.")
         return

    print("\nChecking for pure Nash equilibria in the aggregate performance matrix:")
    print_pure_nash_info(aggregate_performance_matrix)
    
    print("\nFinding Nash equilibrium using replicator dynamics (on aggregate matrix):")
    rd_nash_df = find_nash_with_replicator_dynamics(
        aggregate_performance_matrix, 
        num_restarts=1,
        num_iterations=1,
       verbose=True
    )
    print(rd_nash_df)

    print("\nCalculating Max Entropy Nash Equilibrium for the AGGREGATE performance matrix...")
    
    aggregate_performance_matrix_np = aggregate_performance_matrix.to_numpy()
    
    for i in range(aggregate_performance_matrix_np.shape[0]):
        for j in range(aggregate_performance_matrix_np.shape[1]):
            if np.isnan(aggregate_performance_matrix_np[i, j]):
                col_mean = np.nanmean(aggregate_performance_matrix_np[:, j])
                if not np.isnan(col_mean):
                    aggregate_performance_matrix_np[i, j] = col_mean
                else:
                    row_mean = np.nanmean(aggregate_performance_matrix_np[i, :])
                    aggregate_performance_matrix_np[i, j] = row_mean if not np.isnan(row_mean) else 0
    
    me_nash_strategy = milp_max_sym_ent_2p(aggregate_performance_matrix_np)
    
    agents = aggregate_performance_matrix.index.tolist()
    me_strategy_df = pd.DataFrame({
        'Agent': agents,
        'Nash Probability': me_nash_strategy
    }).sort_values(by='Nash Probability', ascending=False)
    
    print("\nMax Entropy Nash Equilibrium for aggregate performance matrix:")
    print(me_strategy_df)

    me_nash_expected_utilities = np.dot(aggregate_performance_matrix_np, me_nash_strategy)
    me_ne_vs_agent_utilities = np.dot(me_nash_strategy, aggregate_performance_matrix_np)
    nash_value = np.dot(me_nash_strategy, me_nash_expected_utilities)

    performance_with_ne = filtered_matrices_by_round['aggregate']['performance_matrix'].copy()
    performance_with_ne['ME_Nash_Equilibrium'] = pd.Series(
        {agent: me_nash_expected_utilities[i] for i, agent in enumerate(agents)}
    )
    performance_with_ne.loc['ME_Nash_Equilibrium'] = pd.Series(
        {agent: me_ne_vs_agent_utilities[i] for i, agent in enumerate(agents)}, name='ME_Nash_Equilibrium'
    )
    performance_with_ne.loc['ME_Nash_Equilibrium', 'ME_Nash_Equilibrium'] = nash_value
    performance_with_ne = clean_matrix_names(performance_with_ne, get_display_name)
    filtered_matrices_by_round['aggregate']['performance_with_ne'] = performance_with_ne

    for matrix_name in ['utilitarian_welfare_matrix', 'nash_welfare_matrix', 'percent_utilitarian_matrix']:
        if matrix_name in filtered_matrices_by_round['aggregate']:
            welfare_matrix = filtered_matrices_by_round['aggregate'][matrix_name].copy()
            if 'ME_Nash_Equilibrium' not in welfare_matrix.index:
                 welfare_matrix.loc['ME_Nash_Equilibrium', :] = np.nan
            if 'ME_Nash_Equilibrium' not in welfare_matrix.columns:
                 welfare_matrix['ME_Nash_Equilibrium'] = np.nan
            
            global_max_utilitarian = 805.9

            for agent in list(welfare_matrix.index): 
                if agent == 'ME_Nash_Equilibrium': continue
                if agent not in agents: 
                    print(f"Warning: Agent '{agent}' found in welfare matrix index but not in performance matrix agents list. Skipping ME NE calculation for it.")
                    continue
                idx = agents.index(agent) 
                agent_vs_ne_util = me_nash_expected_utilities[idx]
                ne_vs_agent_util = me_ne_vs_agent_utilities[idx] 

                if matrix_name == 'utilitarian_welfare_matrix':
                    welfare = agent_vs_ne_util + ne_vs_agent_util
                    welfare_matrix.loc[agent, 'ME_Nash_Equilibrium'] = welfare
                    welfare_matrix.loc['ME_Nash_Equilibrium', agent] = welfare
                
                elif matrix_name == 'nash_welfare_matrix':
                    if global_max_nash_welfare > 0:
                        term1 = max(0, agent_vs_ne_util)
                        term2 = max(0, ne_vs_agent_util) 
                        welfare = np.sqrt(term1 * term2) / global_max_nash_welfare
                    else:
                        welfare = 0
                    welfare_matrix.loc[agent, 'ME_Nash_Equilibrium'] = welfare
                    welfare_matrix.loc['ME_Nash_Equilibrium', agent] = welfare

                elif matrix_name == 'percent_utilitarian_matrix':
                    util_welfare = agent_vs_ne_util + ne_vs_agent_util
                    if global_max_utilitarian > 0:
                        percent = (util_welfare / global_max_utilitarian) * 100
                    else:
                        percent = 0 
                    welfare_matrix.loc[agent, 'ME_Nash_Equilibrium'] = percent
                    welfare_matrix.loc['ME_Nash_Equilibrium', agent] = percent
            
            if matrix_name == 'utilitarian_welfare_matrix':
                 me_vs_me_welfare = nash_value + nash_value 
            elif matrix_name == 'nash_welfare_matrix':
                 if global_max_nash_welfare > 0:
                      me_vs_me_welfare = np.sqrt(max(0, nash_value) * max(0, nash_value)) / global_max_nash_welfare
                 else:
                      me_vs_me_welfare = 0
            elif matrix_name == 'percent_utilitarian_matrix':
                 util_welfare_me = nash_value + nash_value
                 if global_max_utilitarian > 0:
                      me_vs_me_welfare = (util_welfare_me / global_max_utilitarian) * 100
                 else:
                      me_vs_me_welfare = 0
            else:
                me_vs_me_welfare = np.nan 
            welfare_matrix.loc['ME_Nash_Equilibrium', 'ME_Nash_Equilibrium'] = me_vs_me_welfare
           

            filtered_matrices_by_round['aggregate'][matrix_name] = clean_matrix_names(welfare_matrix, get_display_name)

    if 'utilitarian_welfare_matrix' in filtered_matrices_by_round['aggregate'] and \
       'ME_Nash_Equilibrium' in filtered_matrices_by_round['aggregate']['utilitarian_welfare_matrix'].index:
           util_matrix = filtered_matrices_by_round['aggregate']['utilitarian_welfare_matrix']
           efficiency_loss_matrix = pd.DataFrame(index=util_matrix.index, columns=util_matrix.columns)
           ne_welfare = util_matrix.loc['ME_Nash_Equilibrium', 'ME_Nash_Equilibrium']
           for i, agent1 in enumerate(util_matrix.index):
               for j, agent2 in enumerate(util_matrix.columns):
                   if agent1 != 'ME_Nash_Equilibrium' and agent2 != 'ME_Nash_Equilibrium':
                       pair_welfare = util_matrix.loc[agent1, agent2]
                       if pair_welfare is not None and ne_welfare is not None and ne_welfare > 0:
                           efficiency_diff = ((pair_welfare - ne_welfare) / ne_welfare) * 100
                           efficiency_loss_matrix.loc[agent1, agent2] = efficiency_diff
           filtered_matrices_by_round['aggregate']['nash_efficiency_loss_matrix'] = efficiency_loss_matrix
    
    if 'performance_matrix' in filtered_matrices_by_round['aggregate'] and performance_with_ne is not None:
        regret_against_ne_matrix = pd.DataFrame(index=performance_with_ne.index, columns=['Regret vs Nash'])
        expected_utils = np.dot(aggregate_performance_matrix_np, me_nash_strategy)
        nash_value = me_nash_strategy @ expected_utils
        for i, agent in enumerate(agents):
            if i < len(expected_utils):
                best_response = aggregate_performance_matrix_np[i].max()
                ne_utility = expected_utils[i]
                regret = best_response - ne_utility
                regret_against_ne_matrix.loc[agent, 'Regret vs Nash'] = regret
        regret_against_ne_matrix.loc['ME_Nash_Equilibrium', 'Regret vs Nash'] = 0
        filtered_matrices_by_round['aggregate']['regret_against_ne_matrix'] = regret_against_ne_matrix

    if use_raw_bootstrap:
        print("\nUsing non-parametric bootstrapping with raw game data...")
        bootstrap_results, bootstrap_stats, ne_strategy_df_bootstrap, agent_names_bootstrap = run_raw_data_nash_analysis(
            all_results,
            num_bootstrap_samples=num_bootstrap,
            confidence_level=confidence,
            global_max_nash_welfare=global_max_nash_welfare,
            global_max_nash_welfare_adv = global_max_nash_welfare_adv,
            global_max_util_welfare=global_standard_max
        )
        
        bootstrap_dir = os.path.join(output_dir, 'bootstrap_analysis')
        os.makedirs(bootstrap_dir, exist_ok=True)
        
        print("\nGenerating bootstrap distribution plots...")
        
        try:
            if 'ne_regret' in bootstrap_results and bootstrap_results['ne_regret']:
                ne_dist_fig = plot_bootstrap_distributions(bootstrap_results, 'ne_regret', 'NE Regret', 
                                                         agent_names_bootstrap, title="Nash Equilibrium Regret Distribution")
                if ne_dist_fig:
                    ne_dist_fig.savefig(os.path.join(bootstrap_dir, 'ne_regret_distribution.png'))
                else:
                    print("Warning: Failed to create NE Regret Distribution figure")
        except Exception as e:
            print(f"Error creating NE regret distribution plot: {str(e)}")

        # try:
        #     if 'rd_regret' in bootstrap_results and bootstrap_results['rd_regret']:
        #         rd_dist_fig = plot_bootstrap_distributions(bootstrap_results, 'rd_regret', 'RD Regret', 
        #                                                  agent_names_bootstrap, title="Replicator Dynamics Regret Distribution")
        #         if rd_dist_fig:
        #             rd_dist_fig.savefig(os.path.join(bootstrap_dir, 'rd_regret_distribution.png'))
        #         else:
        #             print("Warning: Failed to create RD Regret Distribution figure")
        # except Exception as e:
        #     print(f"Error creating RD regret distribution plot: {str(e)}")

        # --- New Metric Distribution Plots (using EXPECTED keys/labels) ---
        metrics_to_plot = {
            'agent_expected_normalized_nash_welfare': 'Expected Norm Nash Welfare', 
            'agent_expected_normalized_nash_welfare_adv': 'Expected Norm Nash Welfare Adv', 
            'agent_expected_percent_max_util_welfare': 'Expected % Max Util Welfare', 
            'agent_expected_ef1_freq': 'Expected EF1 Freq (%)' 
        }
        
        # --- DEBUG: Print sample data for Nash Welfare (using EXPECTED key) --- 
        if 'agent_expected_normalized_nash_welfare' in bootstrap_results and bootstrap_results['agent_expected_normalized_nash_welfare']:
            print("\nDEBUG: Sample of agent_expected_normalized_nash_welfare data:")
            sample_data = bootstrap_results['agent_expected_normalized_nash_welfare'][:min(5, len(bootstrap_results['agent_expected_normalized_nash_welfare']))]
            for i, sample in enumerate(sample_data):
                print(f"  Sample {i+1}: {sample[:min(5, len(sample))]}...") # Print first 5 agent values
        else:
             print("\nDEBUG: agent_expected_normalized_nash_welfare data not found or empty in bootstrap_results.")
        # --- END DEBUG ---

        for key, label in metrics_to_plot.items():
            try:
                if key in bootstrap_results and bootstrap_results[key]:
                    fig = plot_bootstrap_distributions(bootstrap_results, key, label, agent_names_bootstrap, 
                                                      title=f"{label} Distribution")
                    if fig:
                        filename = f"{key}_distribution.png"
                        fig.savefig(os.path.join(bootstrap_dir, filename))
                        print(f"Saved {label} distribution plot to {filename}")
                    else:
                        print(f"Warning: Failed to create {label} Distribution figure")
            except Exception as e:
                print(f"Error creating {label} distribution plot: {str(e)}")

        # --- Generate Box Plots (using generalized function) ---
        plot_types = ["box", "running_mean"]
        # Update keys and labels for box/running mean plots (using EXPECTED)
        metrics_and_labels = {
            'ne_regret': 'NE Regret',
            # 'rd_regret': 'RD Regret',
            'agent_expected_normalized_nash_welfare': 'Expected Norm Nash Welfare',
            'agent_expected_normalized_nash_welfare_adv': 'Expected Norm Nash Welfare Adv',
            'agent_expected_percent_max_util_welfare': 'Expected % Max Util Welfare',
            'agent_expected_ef1_freq': 'Expected EF1 Freq (%)'
        }

        for plot_type in plot_types:
             print(f"\nGenerating {plot_type} plots...")
             for key, label in metrics_and_labels.items():
                 try:
                     if key in bootstrap_results and bootstrap_results[key]:
                         fig = plot_bootstrap_distributions(bootstrap_results, key, label, agent_names_bootstrap,
                                                           title=f"{label} {plot_type.replace('_',' ').title()} Plot",
                                                           plot_type=plot_type)
                         if fig:
                             filename = f"{key}_{plot_type}.png"
                             fig.savefig(os.path.join(bootstrap_dir, filename))
                             print(f"Saved {label} {plot_type} plot to {filename}")
                         else:
                             print(f"Warning: Failed to create {label} {plot_type} plot figure")
                 except Exception as e:
                     print(f"Error creating {label} {plot_type} plot: {str(e)}")

        print("\nGenerating confidence interval evolution plots...")
        # Update keys and labels for CI evolution plots (using EXPECTED)
        ci_metrics_to_plot = {
             'ne_regret': 'NE Regret',
             # 'rd_regret': 'RD Regret',
             'agent_expected_utility': 'Expected Utility vs NE',
             'agent_expected_normalized_nash_welfare': 'Expected Norm Nash Welfare',
             'agent_expected_normalized_nash_welfare_adv': 'Expected Norm Nash Welfare Adv',
             'agent_expected_percent_max_util_welfare': 'Expected % Max Util Welfare',
             'agent_expected_ef1_freq': 'Expected EF1 Freq (%)'
        }

        for key, label in ci_metrics_to_plot.items():
             try:
                 if key in bootstrap_results and bootstrap_results[key]:
                     plot_ci_size_evolution(
                         bootstrap_results, key, label, agent_names_bootstrap, bootstrap_dir
                     )
                     print(f"Created {label} CI evolution plot in {bootstrap_dir}")
             except Exception as e:
                 print(f"Error generating {label} CI evolution plots: {str(e)}")
                 traceback.print_exc()

        # Call the comprehensive convergence analysis function
        # This function now internally handles plotting for all available metrics
        print("\nRunning explicit bootstrap convergence analysis to generate all plots...")
        try:
            analyze_bootstrap_results_for_convergence(bootstrap_results, agent_names_bootstrap, bootstrap_dir)
            print(f"Completed explicit convergence analysis, check {bootstrap_dir} for all plots")
        except Exception as e:
            print(f"Error in explicit convergence analysis: {str(e)}")
            traceback.print_exc()

        # --- Add calls to the new comparison visualization function --- 
        print("\nGenerating comparison plots for key bootstrap metrics...")
        # Update keys and labels for comparison plots (referencing bootstrap_stats columns with EXPECTED)
        comparison_pairs = [
            {'key1': 'Mean Expected Norm Nash Welfare', 'label1': 'Exp Norm Nash Welf', 
             'key2': 'Mean Expected % Max Util Welfare', 'label2': 'Exp % Max Util Welf'},
             
            {'key1': 'Mean Expected Norm Nash Welfare', 'label1': 'Exp Norm Nash Welf', 
             'key2': 'Mean Expected EF1 Freq (%)', 'label2': 'Exp EF1 Freq (%)'},
             
            {'key1': 'Mean Expected Utility', 'label1': 'Expected Utility vs NE', 
             'key2': 'Mean Expected Norm Nash Welfare', 'label2': 'Exp Norm Nash Welf'}
        ]

        for pair in comparison_pairs:
            try:
                visualize_metric_comparison(
                    bootstrap_stats=bootstrap_stats, 
                    metric1_key=pair['key1'], 
                    metric1_label=pair['label1'], 
                    metric2_key=pair['key2'], 
                    metric2_label=pair['label2'], 
                    save_dir=bootstrap_dir # Save in the same bootstrap analysis directory
                )
            except Exception as e:
                 print(f"Error generating comparison plot for {pair['label1']} vs {pair['label2']}: {e}")
                 traceback.print_exc()
        # --- End comparison plots --- 

    else:
        print("\nUsing traditional bootstrapping with AGGREGATE performance matrix...")
        bootstrap_results, bootstrap_stats, acceptance_matrix, ne_strategy_df_bootstrap = run_nash_analysis(
            aggregate_performance_matrix, 
            num_bootstrap_samples=num_bootstrap,
            confidence_level=confidence
        )
        
        if 'acceptance_matrix' not in filtered_matrices_by_round['aggregate'] and len(all_results) > 0:
            acceptance_matrix_calc = calculate_acceptance_ratio(all_results, agents)
            filtered_matrices_by_round['aggregate']['acceptance_matrix'] = clean_matrix_names(acceptance_matrix_calc, get_display_name)

    print("\nMax Entropy Nash Equilibrium from bootstrapping:")
    print(ne_strategy_df_bootstrap)
    
    print("\nGenerating comprehensive Nash equilibrium statistics...")
    comparison_df, rd_regret_df, rd_nash_value = generate_all_nash_stats(
        aggregate_performance_matrix, 
        bootstrap_stats, 
        me_strategy_df,
        rd_nash_df
    )
    
    print_rd_nash_summary(rd_regret_df, rd_nash_df, rd_nash_value)
    print_nash_comparison(comparison_df)
    
    print("\nStep 8: Creating visualizations (by round)...")
    welfare_figures_by_round = visualize_welfare_matrices(
        filtered_matrices_by_round,
        os.path.join(output_dir, 'heatmaps')
    )
    
    print("Creating best response graphs (using AGGREGATE data)...")
    if aggregate_performance_matrix is not None:
         br_graph_dir = os.path.join(output_dir, 'graphs')
         best_response_graph = create_best_response_graph(
             aggregate_performance_matrix, 
             filename='best_response_graph', 
             save_dir=br_graph_dir
         )
         complete_best_response_graph = create_complete_best_response_graph(
             aggregate_performance_matrix,
             filename='complete_best_response', 
             save_dir=br_graph_dir
         )
    else:
         print("Skipping BR graphs due to missing aggregate performance matrix.")

    print("Creating Max Entropy Nash equilibrium visualizations (using AGGREGATE data)...")
    max_entropy_nash_dir = os.path.join(output_dir, 'max_entropy_nash')
    os.makedirs(max_entropy_nash_dir, exist_ok=True)
    nash_figures = visualize_nash_equilibrium(
        bootstrap_stats, 
        me_strategy_df,
        save_dir=max_entropy_nash_dir
    )

    print("Creating Replicator Dynamics Nash equilibrium visualizations (using AGGREGATE data)...")
    rd_nash_dir = os.path.join(output_dir, 'rd_nash')
    os.makedirs(rd_nash_dir, exist_ok=True)
    rd_regret_figures = visualize_rd_regret_heatmaps(
        aggregate_performance_matrix,
        rd_regret_df,
        save_dir=rd_nash_dir
    )
    
    print("Creating Nash comparison visualizations (using AGGREGATE data)...")
    comparison_dir = os.path.join(output_dir, 'nash_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_figures = visualize_nash_comparison(
        comparison_df,
        save_dir=comparison_dir
    )
    
    print("Creating Nash distribution plots (using AGGREGATE bootstrap results)...")
    agents_list = agents
    has_rd_regrets = False
    # if use_raw_bootstrap and bootstrap_results and 'rd_regret' in bootstrap_results: # This will now be false
         # has_rd_regrets = True

    nash_plot_figures = plot_nash_distributions(
        bootstrap_results,
        agents_list,
        include_rd_regrets=has_rd_regrets
    )
    save_nash_plots(nash_plot_figures, max_entropy_nash_dir)
    
    print("Creating performance matrix with Nash regret visualization (using AGGREGATE data)...")
    if aggregate_performance_matrix is not None:
         performance_with_regret_fig = create_matrix_heatmap_with_nash_regret(
             aggregate_performance_matrix,
             bootstrap_stats,
             title="Performance Matrix with Nash Equilibrium Regret"
         )
         if performance_with_regret_fig is not None:
             performance_with_regret_path = os.path.join(output_dir, "performance_matrix_with_regret.png")
             performance_with_regret_fig.savefig(performance_with_regret_path, bbox_inches='tight', dpi=300)
             plt.close(performance_with_regret_fig)
    
    
    
    run_anova_analysis(all_results, all_agents, 'utilitarian_welfare', 'Utilitarian Welfare')
    run_anova_analysis(all_results, all_agents, 'nash_welfare', 'Nash Welfare')
    run_anova_analysis(all_results, all_agents, 'nash_welfare_adv', 'Nash Welfare Adv')
    

    ef1_anova_results = []
    for game in all_results:
        temp_game = game.copy()
        is_ef1 = temp_game.get('is_ef1')
        if is_ef1 is True:
             temp_game['ef1_numeric'] = 1
        elif is_ef1 is False:
             temp_game['ef1_numeric'] = 0
        else: # is_ef1 is None
             temp_game['ef1_numeric'] = np.nan 
        ef1_anova_results.append(temp_game)

    run_anova_analysis(ef1_anova_results, all_agents, 'ef1_numeric', 'EF1 Frequency')

    print("\nStep 9: Saving results to CSV files (by round)...")
    csv_base_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_base_dir, exist_ok=True)

    for r, matrices in filtered_matrices_by_round.items():
        round_csv_dir_name = f"round_{r}" if r != 'aggregate' else 'aggregate'
        round_csv_dir = os.path.join(csv_base_dir, round_csv_dir_name)
        os.makedirs(round_csv_dir, exist_ok=True)
        
        for name, matrix in matrices.items():
            if isinstance(matrix, pd.DataFrame) and not matrix.empty:
                matrix.to_csv(os.path.join(round_csv_dir, f"{name}.csv"))
        print(f"  - Saved matrices for round {r} to {round_csv_dir}")

    aggregate_csv_dir = os.path.join(csv_base_dir, 'aggregate')
    if bootstrap_stats is not None:
         bootstrap_stats.to_csv(os.path.join(aggregate_csv_dir, 'bootstrap_statistics.csv'))
    if me_strategy_df is not None:
         me_strategy_df.to_csv(os.path.join(aggregate_csv_dir, 'me_nash_strategy.csv'), index=False)
    if rd_regret_df is not None:
         rd_regret_df.to_csv(os.path.join(aggregate_csv_dir, 'rd_nash_regret.csv'), index=False)
    if comparison_df is not None:
         comparison_df.to_csv(os.path.join(aggregate_csv_dir, 'nash_comparison.csv'), index=False)
    print(f"  - Saved aggregate analysis stats to {aggregate_csv_dir}")
    
    print("\nStep 10: Summary of results (using AGGREGATE data):")
    print(f"Total games analyzed: {len(all_results)}")
    print(f"Unique agent types: {len(all_agents)}")
    
    print("\nAnalysis complete. Results saved to:", output_dir)
    
    summary_dict = {
        'bootstrap_stats': bootstrap_stats.to_dict() if bootstrap_stats is not None else None,
        'ne_strategy_df': me_strategy_df.to_dict() if me_strategy_df is not None else None,
    }
    
    print("\n--- Calculating Agent-Specific Walk Rates (Based on Optimal Calc Games) ---")
    agent_total_games_optimal_calc = defaultdict(int)
    agent_total_walks_optimal_calc = defaultdict(int)
    agent_names_found = set()

    for game in all_results:
        if game.get('is_walk_optimal') is not None: # Denominator consistency
            agent1 = game.get('agent1')
            agent2 = game.get('agent2')
            final_action = game.get('final_action')
            final_actor_role = game.get('final_actor_role') # Role ('agent1' or 'agent2') who acted

            if agent1:
                 agent_total_games_optimal_calc[agent1] += 1
                 agent_names_found.add(agent1)
            if agent2:
                 agent_total_games_optimal_calc[agent2] += 1
                 agent_names_found.add(agent2)

            if final_action == 'WALK':
                if final_actor_role == 'agent1' and agent1:
                    agent_total_walks_optimal_calc[agent1] += 1
                elif final_actor_role == 'agent2' and agent2:
                    agent_total_walks_optimal_calc[agent2] += 1
    
    print("Agent Walk Rates (Walks / Games where Optimality Determined):")
    agent_walk_rates = {}
    for agent in sorted(list(agent_names_found)):
        total_games = agent_total_games_optimal_calc[agent]
        total_walks = agent_total_walks_optimal_calc[agent]
        if total_games > 0:
            walk_rate = (total_walks / total_games) * 100
            agent_walk_rates[agent] = walk_rate
            print(f"  {get_display_name(agent)}: {walk_rate:.2f}% ({total_walks} / {total_games} games)")
        else:
            agent_walk_rates[agent] = np.nan
            print(f"  {get_display_name(agent)}: N/A (0 relevant games)")
    print("---")

    print("\nScript finished.")
    
    # Save bootstrap statistics and aggregate results
    bootstrap_stats.to_csv(os.path.join(output_dir, 'bootstrap_statistics.csv'))
    
    # Save bootstrap_results as JSON since it's a dictionary, not a DataFrame
    try:
        import json
        # First convert numpy arrays to lists for JSON serialization
        json_compatible_results = {}
        for key, value in bootstrap_results.items():
            if isinstance(value, list):
                if value and hasattr(value[0], 'tolist'):
                    json_compatible_results[key] = [item.tolist() if hasattr(item, 'tolist') else item for item in value]
                else:
                    json_compatible_results[key] = value
            elif hasattr(value, 'tolist'):
                json_compatible_results[key] = value.tolist()
            else:
                json_compatible_results[key] = value
                
        with open(os.path.join(output_dir, 'bootstrap_results.json'), 'w') as f:
            json.dump(json_compatible_results, f)
        print(f"  - Saved bootstrap results summary to {os.path.join(output_dir, 'bootstrap_results.json')}")
    except Exception as e:
        print(f"  - Warning: Could not save bootstrap_results to JSON: {e}")
    
    # Save all bootstrap data points for each statistic to separate CSV files
    bootstrap_data_dir = os.path.join(output_dir, 'bootstrap_data_points')
    os.makedirs(bootstrap_data_dir, exist_ok=True)
    
    bootstrap_stats_to_save = [
        'ne_regret', 'rd_regret', 'agent_expected_utility', 
        'agent_expected_normalized_nash_welfare', 'agent_expected_percent_max_util_welfare',
        'agent_expected_normalized_nash_welfare_adv', 
        'agent_expected_ef1_freq', 'ne_strategy', 'rd_strategy',
        'bootstrapped_nash_welfare_matrices', 'bootstrapped_util_welfare_matrices', 'bootstrapped_ef1_freq_matrices',
        'agent_expected_nash_welfare', 'agent_expected_nash_welfare_adv', 'agent_avg_normalized_nash_welfare',
        'agent_avg_normalized_nash_welfare_adv',
        'agent_expected_util_welfare', 'agent_avg_percent_max_util_welfare',
        'agent_avg_ef1_freq'
    ]
    
    for stat_key in bootstrap_stats_to_save:
        if stat_key in bootstrap_results and bootstrap_results[stat_key]:
            try:
                # Special handling for bootstrapped matrices
                if stat_key.startswith('bootstrapped_') and stat_key.endswith('_matrices'):
                    print(f"  - Saving {stat_key} matrices...")
                    matrices_dir = os.path.join(bootstrap_data_dir, f'{stat_key}')
                    os.makedirs(matrices_dir, exist_ok=True)
                    
                    # Save each bootstrapped matrix to a separate file
                    for i, matrix in enumerate(bootstrap_results[stat_key]):
                        if isinstance(matrix, pd.DataFrame):
                            matrix.to_csv(os.path.join(matrices_dir, f'bootstrap_{i+1}.csv'))
                        else:
                            # If not a DataFrame, convert to one
                            try:
                                pd.DataFrame(matrix, 
                                             index=agent_names_bootstrap,
                                             columns=agent_names_bootstrap).to_csv(
                                                 os.path.join(matrices_dir, f'bootstrap_{i+1}.csv'))
                            except Exception as matrix_e:
                                print(f"    - Error saving matrix {i+1}: {matrix_e}")
                    
                    print(f"    - Saved {len(bootstrap_results[stat_key])} {stat_key} matrices")
                else:
                    # Standard handling for regular data arrays
                    # Convert to DataFrame for easy saving
                    data_array = np.array(bootstrap_results[stat_key])
                    
                    # Create column names: one per agent
                    if len(data_array.shape) > 1 and data_array.shape[1] == len(agent_names_bootstrap):
                        columns = agent_names_bootstrap
                    else:
                        # For other statistics, use generic column names
                        columns = [f'dim_{i}' for i in range(data_array.shape[1] if len(data_array.shape) > 1 else 1)]
                    
                    # Create and save DataFrame
                    df = pd.DataFrame(data_array, columns=columns)
                    df.to_csv(os.path.join(bootstrap_data_dir, f'{stat_key}_all_samples.csv'), index=False)
                    print(f"  - Saved all bootstrap samples for {stat_key} to CSV")
            except Exception as e:
                print(f"  - Error saving {stat_key} bootstrap samples: {e}")
    
    return summary_dict

def main():
    """Main function to run bootstrap analysis on a performance matrix."""
    # Load the performance matrix
    performance_matrix = pd.read_csv('performance_matrix.csv', index_col=0)
    
    
    
    output_dir = 'bootstrap_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    bootstrap_results = run_bootstrap_analysis(performance_matrix)
    
    agent_names = performance_matrix.index.tolist()
    bootstrap_stats = analyze_bootstrap_results(bootstrap_results, agent_names)

    bootstrap_results.to_csv(os.path.join(output_dir, 'bootstrap_results.csv'))

    
    bootstrap_stats.to_csv(os.path.join(output_dir, 'bootstrap_statistics.csv'))
    
    print("\nGenerating convergence analysis plots...")
    convergence_analysis = analyze_bootstrap_convergence(bootstrap_results, agent_names)
    
    # Extract regrets from bootstrap results for distribution plots
    ne_regrets = [result['ne_regrets'] for result in bootstrap_results]
    # rd_regrets = [result['rd_regrets'] for result in bootstrap_results]
    
    # Create and save distribution plots
    print("\nGenerating regret distribution plots...")
    ne_dist_fig = plot_regret_distributions(ne_regrets, agent_names, 
                                          title="Nash Equilibrium Regret Distribution")
    if ne_dist_fig is not None:
        ne_dist_fig.savefig(os.path.join(output_dir, 'ne_regret_distribution.png'))
    else:
        print("Warning: Failed to create Nash Equilibrium Regret Distribution figure")
    
    # rd_dist_fig = plot_regret_distributions(rd_regrets, agent_names,
    #                                       title="Replicator Dynamics Regret Distribution")
    # if rd_dist_fig is not None:
    #     rd_dist_fig.savefig(os.path.join(output_dir, 'rd_regret_distribution.png'))
    # else:
    #     print("Warning: Failed to create Replicator Dynamics Regret Distribution figure")
    
    print("\nGenerating regret box plots...")
    ne_box_fig = plot_regret_distributions(ne_regrets, agent_names, 
                                          title="Nash Equilibrium Regret Box Plot", 
                                          plot_type="box")
    if ne_box_fig is not None:
        ne_box_fig.savefig(os.path.join(output_dir, 'ne_regret_boxplot.png'))
    else:
        print("Warning: Failed to create Nash Equilibrium Regret Box Plot figure")
    
    # rd_box_fig = plot_regret_distributions(rd_regrets, agent_names,
    #                                       title="Replicator Dynamics Regret Box Plot", 
    #                                       plot_type="box")
    # if rd_box_fig is not None:
    #     rd_box_fig.savefig(os.path.join(output_dir, 'rd_regret_boxplot.png'))
    # else:
    #     print("Warning: Failed to create Replicator Dynamics Regret Box Plot figure")
    
    # Generate running mean plots
    print("\nGenerating running mean plots...")
    ne_running_fig = plot_regret_distributions(ne_regrets, agent_names, 
                                             title="Nash Equilibrium Regret Running Mean", 
                                             plot_type="running_mean")
    if ne_running_fig is not None:
        ne_running_fig.savefig(os.path.join(output_dir, 'ne_regret_running_mean.png'))
    else:
        print("Warning: Failed to create Nash Equilibrium Regret Running Mean figure")
    
    # rd_running_fig = plot_regret_distributions(rd_regrets, agent_names,
    #                                          title="Replicator Dynamics Regret Running Mean", 
    #                                          plot_type="running_mean")
    # if rd_running_fig is not None:
    #     rd_running_fig.savefig(os.path.join(output_dir, 'rd_regret_running_mean.png'))
    # else:
    #     print("Warning: Failed to create Replicator Dynamics Regret Running Mean figure")
    
    # Print summary statistics
    print("\nBootstrap Analysis Summary:")
    print("=" * 50)
    print("\nNash Equilibrium Statistics:")
    print("-" * 30)
    print(f"Mean NE Regret: {bootstrap_stats['Mean NE Regret'].mean():.6f}")
    print(f"Std NE Regret: {bootstrap_stats['Std NE Regret'].mean():.6f}")
    print(f"95% CI NE Regret: [{bootstrap_stats['CI Lower NE Regret'].mean():.6f}, {bootstrap_stats['CI Upper NE Regret'].mean():.6f}]")
    
    # print("\nReplicator Dynamics Statistics:")
    # print("-" * 30)
    # print(f"Mean RD Regret: {bootstrap_stats['Mean RD Regret'].mean():.6f}")
    # print(f"Std RD Regret: {bootstrap_stats['Std RD Regret'].mean():.6f}")
    # print(f"95% CI RD Regret: [{bootstrap_stats['CI Lower RD Regret'].mean():.6f}, {bootstrap_stats['CI Upper RD Regret'].mean():.6f}]")
    
    print("\nExpected Utility Statistics:")
    print("-" * 30)
    print(f"Mean Expected Utility: {bootstrap_stats['Mean Expected Utility'].mean():.6f}")
    print(f"Std Expected Utility: {bootstrap_stats['Std Expected Utility'].mean():.6f}")
    print(f"95% CI Expected Utility: [{bootstrap_stats['CI Lower Expected Utility'].mean():.6f}, {bootstrap_stats['CI Upper Expected Utility'].mean():.6f}]")
    
    print("\nConvergence Analysis:")
    print("-" * 30)
    print(f"NE Regrets Converged: {'Yes' if convergence_analysis['ne_converged'] else 'No'}")
    print(f"Expected Utilities Converged: {'Yes' if convergence_analysis['eu_converged'] else 'No'}")
    # print(f"RD Regrets Converged: {'Yes' if convergence_analysis['rd_converged'] else 'No'}")
    
    if not (convergence_analysis['ne_converged'] and convergence_analysis['eu_converged']):
        print("\nWARNING: Some statistics have not converged. Consider increasing the number of bootstrap samples.")
    
    print("\nAll plots and statistics have been saved to the 'bootstrap_analysis' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run meta-game analysis on negotiation data.")
    parser.add_argument("--input", default="crossplay/game_matrix_1a", help="Input directory containing game data")
    parser.add_argument("--output", default="meta_game_analysis/results", help="Output directory for results")
    parser.add_argument("--bootstrap", type=int, default=100, help="Number of bootstrap samples")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for intervals")
    parser.add_argument("--global-samples", type=int, default=100000, help="Number of samples for global max values")
    parser.add_argument("--raw-bootstrap", action="store_true", default=True, help="Use raw game data for non-parametric bootstrapping")
    parser.add_argument("--discount", type=float, default=0.9, help="Discount factor (gamma) for utilities")
    parser.add_argument("--track-br-evolution", action="store_true", help="Track best response graph evolution")
    parser.add_argument("--br-batch-size", type=int, default=100, help="Number of games in first batch for BR graph evolution")
    parser.add_argument("--br-bootstraps", type=int, default=100, help="Number of bootstrap samples for each batch in BR evolution")
    parser.add_argument("--random-seed", type=int, default=42, help="Fixed random seed for reproducibility")
    parser.add_argument("--cell-bootstrap", action="store_true", help="Use cell-based bootstrapping for BR evolution")
    parser.add_argument("--run-temporal-analysis", action="store_true", help="Run temporal bargaining analysis")
    parser.add_argument("--full-meta-game", action="store_true", help="Mix all underlying games into one meta-game")

    args = parser.parse_args()

    run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        num_bootstrap=args.bootstrap,
        confidence=args.confidence,
        global_samples=args.global_samples,
        use_raw_bootstrap=args.raw_bootstrap,
        discount_factor=args.discount,
        track_br_evolution=args.track_br_evolution,
        batch_size=args.br_batch_size,
        num_br_bootstraps=args.br_bootstraps,
        random_seed=args.random_seed,
        use_cell_based_bootstrap=args.cell_bootstrap,
        run_temporal_analysis=args.run_temporal_analysis,
        full_game_mix=args.full_meta_game,
    )

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from nash_equilibrium.nash_solver import (
    milp_max_sym_ent_2p,
    replicator_dynamics_nash,
    calculate_max_regret,
    minimize_max_regret,
    compute_regret,
    _simplex_projection,
    milp_nash_2p
)


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nash_equilibrium import (
    bootstrap_performance_metrics,
    analyze_bootstrap_results,
    compute_acceptance_ratio_matrix,
    plot_normal_regret_distributions,
    visualize_normal_regret_comparison,
    visualize_ne_nbs
)
from nash_equilibrium.bootstrap import visualize_dual_regret, visualize_nash_mixture_with_ci
from meta_game_analysis.bootstrap_nonparametric import (
        nonparametric_bootstrap_from_raw_data, 
        analyze_bootstrap_convergence,
        analyze_bootstrap_results_for_convergence,
        plot_bootstrap_iteration,
        plot_confidence_interval_stability,
        plot_bootstrap_distributions
    )
from scipy.stats import pearsonr


def run_nash_analysis(performance_matrix, num_bootstrap_samples=100, confidence_level=0.95):
    """
    Run Nash equilibrium analysis on the performance matrix.
    
    Args:
        performance_matrix: Performance matrix DataFrame
        num_bootstrap_samples: Number of bootstrap samples to use
        confidence_level: Confidence level for bootstrap intervals
        
    Returns:
        tuple: (bootstrap_results, bootstrap_stats, acceptance_matrix, ne_strategy_df)
    """
    agents = performance_matrix.index.tolist()
    tqdm.write(f"Running bootstrapping with {num_bootstrap_samples} samples...")
    bootstrap_results = bootstrap_performance_metrics(
        performance_matrix, 
        num_bootstrap=num_bootstrap_samples, 
        data_matrix={} 
    )

    tqdm.write(f"\nComputing {confidence_level*100:.0f}% confidence intervals...")
    bootstrap_stats = analyze_bootstrap_results(
        bootstrap_results, 
        agents, 
        confidence=confidence_level
    )
    
    acceptance_matrix = None
    #tqdm.q
    avg_ne_strategy = np.mean([s for s in bootstrap_results['ne_strategy']], axis=0)
    ne_strategy_df = pd.DataFrame({
        'Agent': agents,
        'Nash Probability': avg_ne_strategy
    }).sort_values(by='Nash Probability', ascending=False)
    
    return bootstrap_results, bootstrap_stats, acceptance_matrix, ne_strategy_df

def plot_nash_distributions(bootstrap_results, agents, include_rd_regrets=False):
    """
    Plot the distributions of Nash equilibrium metrics.
    
    Args:
        bootstrap_results: Bootstrap results dictionary
        agents: List of agent names
        include_rd_regrets: Whether to look for RD regrets in bootstrap_results (for non-parametric bootstrapping)
        
    Returns:
        dict: Dictionary of figure objects with keys like 'me_ne_regret', 'rd_ne_regret', etc.
    """
    figures = {}
    
    try:
        me_ne_regret_fig = plot_bootstrap_distributions(
            bootstrap_results,
            statistic_key='ne_regret',
            metric_label='NE Regret',
            agent_names=agents,
            title="Max Entropy Nash Equilibrium Regret Distributions"
        )
        if me_ne_regret_fig is not None:
            figures['me_ne_regret'] = me_ne_regret_fig
        
        has_rd_regrets = ('rd_regret' in bootstrap_results and bootstrap_results['rd_regret']) or include_rd_regrets
        
        if has_rd_regrets:
            rd_ne_regret_fig = plot_bootstrap_distributions(
                bootstrap_results,
                statistic_key='rd_regret',
                metric_label='RD Regret',
                agent_names=agents,
                title="Replicator Dynamics Nash Equilibrium Regret Distributions"
            )
            if rd_ne_regret_fig is not None:
                figures['rd_ne_regret'] = rd_ne_regret_fig
        
        has_ne_nbs = 'ne_nbs' in bootstrap_results and bootstrap_results['ne_nbs']
        if has_ne_nbs:
            ne_nbs_fig = visualize_ne_nbs(bootstrap_results, agents)
            if ne_nbs_fig is not None:
                figures['ne_nbs'] = ne_nbs_fig
        
        has_me_normal = 'me_normal_regret' in bootstrap_results and bootstrap_results['me_normal_regret']
        has_rd_normal = 'rd_normal_regret' in bootstrap_results and bootstrap_results['rd_normal_regret']
        
        if has_me_normal:
            me_normal_regret_fig = plot_normal_regret_distributions(bootstrap_results, agents, regret_type='me_normal_regret')
            if me_normal_regret_fig is not None:
                figures['me_normal_regret'] = me_normal_regret_fig
        
        if has_rd_normal:
            rd_normal_regret_fig = plot_normal_regret_distributions(bootstrap_results, agents, regret_type='rd_normal_regret')
            if rd_normal_regret_fig is not None:
                figures['rd_normal_regret'] = rd_normal_regret_fig
        
        if has_me_normal and has_rd_normal:
            normal_comparison_fig = visualize_normal_regret_comparison(bootstrap_results, agents)
            if normal_comparison_fig is not None:
                figures['normal_regret_comparison'] = normal_comparison_fig
        
        # Plot dual regret visualization comparing both equilibrium types
        dual_regret_fig = visualize_dual_regret(bootstrap_results, agents)
        if dual_regret_fig is not None:
            figures['dual_ne_regret'] = dual_regret_fig
        
        nash_mixture_fig = visualize_nash_mixture_with_ci(bootstrap_results, agents)
        if nash_mixture_fig is not None:
            figures['nash_mixture'] = nash_mixture_fig
    
    except Exception as e:
        print(f"Error generating Nash distribution plots: {e}")
        import traceback
        traceback.print_exc()
    
    return figures

def save_nash_plots(figures, save_dir):
    """
    Save Nash equilibrium plot figures to files.
    
    Args:
        figures: Dictionary of figure objects
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for name, fig in figures.items():
        if fig is None:
            print(f"Warning: Figure '{name}' is None, skipping")
            continue
            
        if not hasattr(fig, 'savefig'):
            print(f"Warning: Figure '{name}' is not a valid figure object (type: {type(fig)}), skipping")
            continue
            
        try:
            filename = f"{name}.png"
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving figure '{name}': {e}")
            continue

def print_nash_summary(bootstrap_stats, ne_strategy_df, bootstrap_results):
    """
    Print a summary of Nash equilibrium analysis results.
    
    Args:
        bootstrap_stats: DataFrame with bootstrap statistics
        ne_strategy_df: DataFrame with Nash equilibrium strategy probabilities
        bootstrap_results: Bootstrap results dictionary
    """
    print("\nNash Equilibrium Mixed Strategy (Probability Distribution):")
    print(ne_strategy_df)

    try:
        if isinstance(bootstrap_results['ne_regret'], list) and len(bootstrap_results['ne_regret']) > 0:
            if isinstance(bootstrap_results['ne_regret'][0], np.ndarray):
                mean_regrets = bootstrap_stats['Mean NE Regret'].values
                std_regrets = bootstrap_stats['Std NE Regret'].values
                
                has_me_normal = 'Mean ME Normal Regret' in bootstrap_stats.columns
                has_rd_normal = 'Mean RD Normal Regret' in bootstrap_stats.columns
                
                if has_me_normal:
                    mean_me_normal_regrets = bootstrap_stats['Mean ME Normal Regret'].values
                    std_me_normal_regrets = bootstrap_stats['Std ME Normal Regret'].values
                
                if has_rd_normal:
                    mean_rd_normal_regrets = bootstrap_stats['Mean RD Normal Regret'].values
                    std_rd_normal_regrets = bootstrap_stats['Std RD Normal Regret'].values
            else:
                mean_regrets = bootstrap_stats['Mean NE Regret'].values
                std_regrets = bootstrap_stats['Std NE Regret'].values
                
                has_me_normal = 'Mean ME Normal Regret' in bootstrap_stats.columns
                has_rd_normal = 'Mean RD Normal Regret' in bootstrap_stats.columns
                
                if has_me_normal:
                    mean_me_normal_regrets = bootstrap_stats['Mean ME Normal Regret'].values
                    std_me_normal_regrets = bootstrap_stats['Std ME Normal Regret'].values
                
                if has_rd_normal:
                    mean_rd_normal_regrets = bootstrap_stats['Mean RD Normal Regret'].values
                    std_rd_normal_regrets = bootstrap_stats['Std RD Normal Regret'].values
            
            print("\nStatistical Summary of Nash Equilibrium Analysis:")
            print(f"Average NE regret across all agents: {np.mean(mean_regrets):.4f}")
            print(f"Maximum average regret: {np.max(mean_regrets):.4f}")
            print(f"Minimum average regret: {np.min(mean_regrets):.4f}")
            print(f"Standard deviation of average regrets: {np.std(mean_regrets, ddof=1):.4f}")
            
            if has_me_normal:
                print(f"\nAverage ME normal regret across all agents: {np.mean(mean_me_normal_regrets):.4f}")
                print(f"Maximum average ME normal regret: {np.max(mean_me_normal_regrets):.4f}")
                print(f"Minimum average ME normal regret: {np.min(mean_me_normal_regrets):.4f}")
                print(f"Standard deviation of average ME normal regrets: {np.std(mean_me_normal_regrets, ddof=1):.4f}")
            
            if has_rd_normal:
                print(f"\nAverage RD normal regret across all agents: {np.mean(mean_rd_normal_regrets):.4f}")
                print(f"Maximum average RD normal regret: {np.max(mean_rd_normal_regrets):.4f}")
                print(f"Minimum average RD normal regret: {np.min(mean_rd_normal_regrets):.4f}")
                print(f"Standard deviation of average RD normal regrets: {np.std(mean_rd_normal_regrets, ddof=1):.4f}")
        else:
            print("\nNo valid bootstrap regret data available.")
            return
    except Exception as e:
        print(f"\nError computing bootstrap statistics: {e}")
        print("Using pre-computed statistics from bootstrap_stats instead.")
        mean_regrets = bootstrap_stats['Mean NE Regret'].values
        std_regrets = bootstrap_stats['Std NE Regret'].values

    top_agents = bootstrap_stats.head(5)
    print("\nTop 5 agents by Nash Equilibrium analysis (lowest regret):")
    print(top_agents[['Agent', 'Mean NE Regret', 'Std NE Regret']])
    
    print("\nTop 5 agents by Expected Utility (higher is better):")
    print(bootstrap_stats.sort_values('Mean Expected Utility', ascending=False).head(5)[['Agent', 'Mean Expected Utility', 'Mean NE Regret']])

def calculate_acceptance_ratio(all_results, agents):
    """
    Calculate the acceptance ratio matrix for all agents.
    
    Args:
        all_results: List of all game results
        agents: List of agent names
        
    Returns:
        DataFrame: Acceptance ratio matrix
    """
    acceptance_matrix = compute_acceptance_ratio_matrix(all_results, agents)
    return acceptance_matrix

def run_raw_data_nash_analysis(all_results, num_bootstrap_samples=100, confidence_level=0.95,
                               # Add global max arguments with defaults
                               global_max_nash_welfare=None, global_max_nash_welfare_adv=None, global_max_util_welfare=None):
    """
    Run Nash equilibrium analysis using non-parametric bootstrapping on raw game data.
    This is the preferred method for direct bootstrapping from individual game outcomes.
    
    Args:
        all_results: List of dictionaries containing raw game results
        num_bootstrap_samples: Number of bootstrap samples to use
        confidence_level: Confidence level for bootstrap intervals
        global_max_nash_welfare: Global maximum Nash welfare
        global_max_util_welfare: Global maximum utility welfare
        
    Returns:
        tuple: (bootstrap_results, bootstrap_stats, ne_strategy_df, agent_names)
    """
    
    
    print(f"Running non-parametric bootstrapping with {num_bootstrap_samples} samples...")
    bootstrap_results, agent_names = nonparametric_bootstrap_from_raw_data(
        all_results,
        num_bootstrap=num_bootstrap_samples,
        confidence=confidence_level,
        global_max_nash_welfare=global_max_nash_welfare,
        global_max_nash_welfare_adv=global_max_nash_welfare_adv,
        global_max_util_welfare=global_max_util_welfare
    )
    
    bootstrap_stats = bootstrap_results['statistics']
    
    # Create output directory for bootstrap analysis
    output_dir = 'bootstrap_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform convergence analysis and generate plots
    try:
        print(f"\nAnalyzing bootstrap convergence with {num_bootstrap_samples} samples...")
        
        convergence_analysis = analyze_bootstrap_convergence(bootstrap_results, agent_names)
        
        convergence_df = pd.DataFrame({
            'Metric': ['NE Regrets', 'Expected Utilities', 'RD Regrets'],
            'Converged': [
                convergence_analysis.get('ne_converged', False),
                convergence_analysis.get('eu_converged', False), 
                convergence_analysis.get('rd_converged', False)
            ],
            'Mean Monte Carlo Error': [
                np.mean(convergence_analysis.get('ne_errors', [0])),
                np.mean(convergence_analysis.get('eu_errors', [0])),
                np.mean(convergence_analysis.get('rd_errors', [0]))
            ]
        })
        
        # Perform enhanced convergence analysis based on bootstrap paper methods
        print("\nRunning detailed bootstrap convergence analysis following the bootstrap paper methods...")
        print("This analysis will help determine if more bootstrap samples or simulator data are needed.")
        print("The following convergence metrics will be generated:")
        print("1. Bootstrap Iteration Plots - Shows how statistics stabilize with more bootstrap samples")
        print("2. Confidence Interval Stability - Shows how CIs stabilize as bootstrap sample size increases")
        print("3. Monte Carlo Errors - Measures uncertainty in bootstrap estimates")
        print("4. Relative Errors - Assesses reliability relative to the magnitude of estimates")
        
        enhanced_convergence = analyze_bootstrap_results_for_convergence(
            bootstrap_results, 
            agent_names,
            output_dir
        )
        
        convergence_df.to_csv(os.path.join(output_dir, 'convergence_analysis.csv'), index=False)
        
        bootstrap_data_dir = os.path.join(output_dir, 'bootstrap_data_points')
        os.makedirs(bootstrap_data_dir, exist_ok=True)
        
        # Define statistics to save
        bootstrap_stats_to_save = [
            'ne_regret', 'rd_regret', 'agent_expected_utility', 
            'agent_expected_normalized_nash_welfare', 'agent_expected_percent_max_util_welfare',
            'agent_expected_ef1_freq', 'ne_strategy', 'rd_strategy',
            'agent_max_utility', 'nash_value', 'rd_nash_value', 'ne_nbs',
            # Add full bootstrap matrices
            'bootstrapped_nash_welfare_matrices', 'bootstrapped_util_welfare_matrices', 'bootstrapped_ef1_freq_matrices',
            # Add both expected and average versions of welfare metrics
            'agent_expected_nash_welfare', 'agent_avg_normalized_nash_welfare',
            'agent_expected_util_welfare', 'agent_avg_percent_max_util_welfare',
            'agent_avg_ef1_freq'
        ]
        
        # Save each statistic to a separate CSV file
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
                                                index=agent_names,
                                                columns=agent_names).to_csv(
                                                    os.path.join(matrices_dir, f'bootstrap_{i+1}.csv'))
                                except Exception as matrix_e:
                                    print(f"    - Error saving matrix {i+1}: {matrix_e}")
                        
                        print(f"    - Saved {len(bootstrap_results[stat_key])} {stat_key} matrices")
                    else:
                        # Standard handling for regular data arrays
                        # Convert to DataFrame for easy saving
                        data_array = np.array(bootstrap_results[stat_key])
                        
                        # Create column names based on data type
                        if len(data_array.shape) > 1 and data_array.shape[1] == len(agent_names):
                            columns = agent_names
                        else:
                            # For other statistics, use generic column names
                            columns = [f'dim_{i}' for i in range(data_array.shape[1] if len(data_array.shape) > 1 else 1)]
                        
                        # Create and save DataFrame
                        df = pd.DataFrame(data_array, columns=columns)
                        df.to_csv(os.path.join(bootstrap_data_dir, f'{stat_key}_all_samples.csv'), index=False)
                        print(f"  - Saved all bootstrap samples for {stat_key} to CSV")
                except Exception as e:
                    print(f"  - Error saving {stat_key} bootstrap samples: {e}")
    except Exception as e:
        print(f"Error during convergence analysis: {e}")
        print("Continuing with other analyses...")
        
        convergence_df = pd.DataFrame({
            'Metric': ['NE Regrets', 'Expected Utilities', 'RD Regrets'],
            'Converged': [False, False, False],
            'Mean Monte Carlo Error': [0.0, 0.0, 0.0]
        })
    
    # Compute Max Entropy Nash equilibrium mixed strategy
    avg_ne_strategy = np.mean([s for s in bootstrap_results['ne_strategy']], axis=0)
    ne_strategy_df = pd.DataFrame({
        'Agent': agent_names,
        'Nash Probability': avg_ne_strategy
    }).sort_values(by='Nash Probability', ascending=False)
    
    # Compute Replicator Dynamics Nash equilibrium mixed strategy
    if 'rd_strategy' in bootstrap_results and bootstrap_results['rd_strategy']:
        avg_rd_strategy = np.mean([s for s in bootstrap_results['rd_strategy']], axis=0)
        rd_strategy_df = pd.DataFrame({
            'Agent': agent_names,
            'RD Nash Probability': avg_rd_strategy
        }).sort_values(by='RD Nash Probability', ascending=False)
        
        # Add RD Nash Probability to the main ne_strategy_df for comparison
        for agent in agent_names:
            rd_prob = rd_strategy_df.loc[rd_strategy_df['Agent'] == agent, 'RD Nash Probability'].values
            if len(rd_prob) > 0:
                ne_strategy_df.loc[ne_strategy_df['Agent'] == agent, 'RD Nash Probability'] = rd_prob[0]
    
    print("\nNash Equilibrium Analysis Complete")
    print_nash_summary(bootstrap_stats, ne_strategy_df, bootstrap_results)
    
    # Print convergence analysis summary if available
    try:
        if 'convergence_analysis' in locals():
            print("\nBasic Convergence Analysis Summary:")
            print("-" * 50)
            
            # Check if convergence_df exists in the local scope
            if 'convergence_df' in locals():
                for metric, converged in zip(convergence_df['Metric'], convergence_df['Converged']):
                    status = "Converged" if converged else "Not converged"
                    print(f"{metric}: {status}")
                
                if not all(convergence_df['Converged']):
                    print("\nWARNING: Some statistics have not converged. Consider increasing the number of bootstrap samples.")
            else:
                print("Convergence analysis data not available.")
            
            print("\nHow to interpret bootstrap convergence results:")
            print("1. Check if Monte Carlo errors are small relative to the estimates")
            print("   - Relative errors < 5% indicate good convergence")
            print("   - Relative errors > 10% suggest more bootstrap samples are needed")
            print("2. Examine bootstrap iteration plots to see if statistics have stabilized")
            print("3. Review CI stability plots to see if confidence intervals have stabilized")
            print("4. Large fluctuations in any statistics may indicate insufficient simulator data")
            print("\nAll convergence plots and statistics have been saved to the 'bootstrap_analysis' directory.")
    except Exception as e:
        print(f"Error printing convergence summary: {e}")
    
    return bootstrap_results, bootstrap_stats, ne_strategy_df, agent_names

def find_pure_nash_equilibria(performance_matrix):
    """
    Find all pure Nash equilibria in the performance matrix.
    
    Args:
        performance_matrix: DataFrame with performance data (rows=player 1 strategies, cols=player 2 strategies)
        
    Returns:
        list: List of tuples (row_idx, col_idx) representing pure Nash equilibria
    """
    if not isinstance(performance_matrix, pd.DataFrame):
        raise ValueError("Performance matrix must be a pandas DataFrame")
    
    pure_nash = []
    
    # Convert to numpy array for faster processing
    matrix = performance_matrix.to_numpy()
    agent_names = performance_matrix.index.tolist()
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]) or np.isnan(matrix[j, i]):
                continue
            best_response_to_j = np.nanargmax(matrix[:, j])
            
            best_response_to_i = np.nanargmax(matrix[i, :])
            
            if best_response_to_j == i and best_response_to_i == j:
                pure_nash.append((agent_names[i], agent_names[j]))
    
    return pure_nash

def print_pure_nash_info(performance_matrix):
    """
    Check for pure Nash equilibria and print the results.
    
    Args:
        performance_matrix: DataFrame with performance data
    """
    pure_nash = find_pure_nash_equilibria(performance_matrix)
    
    if pure_nash:
        print("\nPure Nash Equilibria Found:")
        for i, (row, col) in enumerate(pure_nash):
            p1_value = performance_matrix.loc[row, col]
            p2_value = performance_matrix.loc[col, row]
            print(f"  {i+1}. ({row}, {col}) with values: ({p1_value:.2f}, {p2_value:.2f})")
    else:
        print("\nNo pure Nash equilibria found in the performance matrix.")
        
    diag_values = []
    for agent in performance_matrix.index:
        if agent in performance_matrix.columns:
            value = performance_matrix.loc[agent, agent]
            if not np.isnan(value):
                diag_values.append((agent, value))
    
    if diag_values:
        best_self_play = max(diag_values, key=lambda x: x[1])
        print(f"\nBest self-play: {best_self_play[0]} with value {best_self_play[1]:.2f}")
    else:
        print("\nNo valid self-play values found.")

def find_nash_with_replicator_dynamics(performance_matrix, num_restarts=10, num_iterations=2000, 
                                      convergence_threshold=1e-6, verbose=False, return_all=False):
    """
    Find Nash equilibrium using replicator dynamics with multiple random restarts.
    
    Args:
        performance_matrix: DataFrame or 2D numpy array representing payoff matrix
        num_restarts: Number of random initializations to try
        num_iterations: Maximum number of iterations for each run
        convergence_threshold: Convergence threshold for stopping condition
        verbose: Whether to print progress
        return_all: Whether to return all found equilibria or just the best one
        
    Returns:
        tuple: (best_nash_strategy, all_strategies, all_convergence_info)
    """

    if isinstance(performance_matrix, pd.DataFrame):
        agent_names = performance_matrix.index.tolist()
        payoff_matrix = performance_matrix.to_numpy()
    else:
        payoff_matrix = performance_matrix
        agent_names = [f"Strategy {i}" for i in range(payoff_matrix.shape[0])]
    
    n = payoff_matrix.shape[0]
    
    for j in range(n):
        col_mean = np.nanmean(payoff_matrix[:, j])
        for i in range(n):
            if np.isnan(payoff_matrix[i, j]):
                payoff_matrix[i, j] = col_mean if not np.isnan(col_mean) else 0
    
    best_strategy, best_iter = replicator_dynamics_nash(
        payoff_matrix,
        max_iter=num_iterations,
        epsilon=convergence_threshold
    )
    best_converged = max(compute_regret(best_strategy, payoff_matrix)[0]) <= convergence_threshold
    
    all_strategies = [best_strategy]
    all_convergence = [(best_converged, best_iter)]
    
    if verbose:
        expected_payoffs = np.dot(payoff_matrix, best_strategy)
        average_payoff = np.dot(best_strategy, expected_payoffs)
        print(f"Initial uniform run - Avg payoff: {average_payoff:.4f}, Converged: {best_converged}")
    
    for i in range(num_restarts):
        #random_strategy = np.random.dirichlet(np.ones(n))
        
        strategy, iterations = replicator_dynamics_nash(
            payoff_matrix,
            max_iter=num_iterations,
            epsilon=convergence_threshold
        )
         
        all_strategies.append(strategy)
        converged = max(compute_regret(strategy, payoff_matrix)[0]) <= convergence_threshold

        all_convergence.append((converged, iterations))
        
        expected_payoffs = np.dot(payoff_matrix, strategy)
        average_payoff = np.dot(strategy, expected_payoffs)
        
        best_expected_payoffs = np.dot(payoff_matrix, best_strategy)
        best_average_payoff = np.dot(best_strategy, best_expected_payoffs)
        
        if verbose:
            print(f"Restart {i+1}/{num_restarts} - Avg payoff: {average_payoff:.4f}, Converged: {converged}")
        
        if average_payoff > best_average_payoff:
            best_strategy = strategy
            best_converged = converged
            best_iter = iterations
    
    if return_all:
        all_strategies_df = pd.DataFrame(all_strategies, columns=agent_names)
        all_strategies_df['Converged'] = [conv for conv, _ in all_convergence]
        all_strategies_df['Iterations'] = [iters for _, iters in all_convergence]
        
        payoffs = []
        for strat in all_strategies:
            expected_payoffs = np.dot(payoff_matrix, strat)
            average_payoff = np.dot(strat, expected_payoffs)
            payoffs.append(average_payoff)
        
        all_strategies_df['Average Payoff'] = payoffs
        all_strategies_df = all_strategies_df.sort_values('Average Payoff', ascending=False)
        
        return best_strategy, all_strategies_df
    
    best_nash_df = pd.DataFrame({
        'Agent': agent_names,
        'Nash Probability': best_strategy
    }).sort_values(by='Nash Probability', ascending=False)
    
    return best_nash_df

def calculate_regrets_against_replicator_nash(performance_matrix, rd_nash_strategy):
    """
    Calculate regrets against the replicator dynamics Nash equilibrium.
    
    Args:
        performance_matrix: DataFrame with performance data
        rd_nash_strategy: DataFrame with Nash probabilities from replicator dynamics
        
    Returns:
        DataFrame with regret statistics
    """
    if isinstance(performance_matrix, pd.DataFrame):
        agent_names = performance_matrix.index.tolist()
        payoff_matrix = performance_matrix.to_numpy()
    else:
        raise ValueError("Performance matrix must be a pandas DataFrame")
    
    nash_strategy = np.zeros(len(agent_names))
    for i, agent in enumerate(agent_names):
        idx = rd_nash_strategy[rd_nash_strategy['Agent'] == agent].index
        if len(idx) > 0:
            nash_strategy[i] = rd_nash_strategy.loc[idx[0], 'Nash Probability']
    
    # NOTE: There really shouldn't be any but just in case
    for i in range(payoff_matrix.shape[0]):
        for j in range(payoff_matrix.shape[1]):
            if np.isnan(payoff_matrix[i, j]):
                col_mean = np.nanmean(payoff_matrix[:, j])
                payoff_matrix[i, j] = col_mean if not np.isnan(col_mean) else 0
    
    is_pure_nash = np.max(nash_strategy) == 1.0
    if is_pure_nash:
        pure_nash_idx = np.argmax(nash_strategy)
        pure_nash_agent = agent_names[pure_nash_idx]
        print(f"\nDetected pure Nash equilibrium: {pure_nash_agent}")
    
    expected_utils = np.dot(payoff_matrix, nash_strategy)
    
    nash_vs_agents = np.zeros(len(agent_names))
    for i in range(len(agent_names)): #add in payoff for pure strategy agent agaisnt nash
        nash_vs_agents[i] = np.dot(nash_strategy, payoff_matrix[:, i])
    
    #relative_regrets = nash_vs_agents - expected_utils
    relative_regrets = expected_utils - nash_vs_agents
    
    nash_value = nash_strategy.reshape((1, -1)) @ payoff_matrix @ nash_strategy.reshape((-1, 1))
    nash_value = nash_value.item()  
    print(f"Nash equilibrium value: {nash_value:.2f}")
    
    for i, agent in enumerate(agent_names):
        print(f"Agent {agent}: Payoff vs Nash = {expected_utils[i]:.2f}, Nash vs Agent = {nash_vs_agents[i]:.2f}, Regret = {relative_regrets[i]:.2f}")
    
    max_utils = np.max(payoff_matrix, axis=1)
    
    results = pd.DataFrame({
        'Agent': agent_names,
        'RD Nash Regret': relative_regrets,
        'RD Expected Utility': expected_utils,
        'RD Max Utility': max_utils,
        'RD Nash Value': nash_value
    })
    
    return results, nash_value

def generate_all_nash_stats(performance_matrix, bootstrap_stats, ne_strategy_df, rd_nash_df):
    """
    Generate comprehensive statistics for both Nash equilibrium approaches.
    
    Args:
        performance_matrix: DataFrame with performance data
        bootstrap_stats: DataFrame with bootstrap statistics
        ne_strategy_df: DataFrame with Nash probabilities from max entropy Nash
        rd_nash_df: DataFrame with Nash probabilities from replicator dynamics
        
    Returns:
        tuple: (comparison_df, rd_regret_df)
    """
    rd_regret_df, rd_nash_value = calculate_regrets_against_replicator_nash(performance_matrix, rd_nash_df)
    
    comparison = []
    for agent in performance_matrix.index:
        bs_row = bootstrap_stats[bootstrap_stats['Agent'] == agent]
        me_regret = bs_row['Mean NE Regret'].values[0] if len(bs_row) > 0 else np.nan
        
        # Find agent in RD regret DataFrame
        rd_row = rd_regret_df[rd_regret_df['Agent'] == agent]
        rd_regret = rd_row['RD Nash Regret'].values[0] if len(rd_row) > 0 else np.nan
        
        # Find Nash probabilities
        me_prob_row = ne_strategy_df[ne_strategy_df['Agent'] == agent]
        me_prob = me_prob_row['Nash Probability'].values[0] if len(me_prob_row) > 0 else 0
        
        rd_prob_row = rd_nash_df[rd_nash_df['Agent'] == agent]
        rd_prob = rd_prob_row['Nash Probability'].values[0] if len(rd_prob_row) > 0 else 0
        
        comparison.append({
            'Agent': agent,
            'ME Nash Probability': me_prob,
            'RD Nash Probability': rd_prob,
            'ME Nash Regret': me_regret,
            'RD Nash Regret': rd_regret
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    return comparison_df, rd_regret_df, rd_nash_value

def print_rd_nash_summary(rd_regret_df, rd_nash_df, rd_nash_value):
    """
    Print a summary of replicator dynamics Nash equilibrium results.
    
    Args:
        rd_regret_df: DataFrame with regret statistics against RD Nash
        rd_nash_df: DataFrame with Nash probabilities from replicator dynamics
        rd_nash_value: Nash equilibrium value from replicator dynamics
    """
    print("\nReplicator Dynamics Nash Equilibrium Summary:")
    print(f"Nash equilibrium value: {rd_nash_value:.4f}")
    
    # Sort by Nash regret
    rd_top_agents = rd_regret_df.sort_values('RD Nash Regret', ascending=False).head(5)
    
    print("\nTop 5 agents by RD Nash Regret (higher is better):")
    print(rd_top_agents[['Agent', 'RD Nash Regret']])
    
    print("\nTop 5 agents by RD Expected Utility (higher is better):")
    print(rd_regret_df.sort_values('RD Expected Utility', ascending=False).head(5)[['Agent', 'RD Expected Utility']])

def print_nash_comparison(comparison_df):
    """
    Print a comparison of the two Nash equilibrium concepts.
    
    Args:
        comparison_df: DataFrame with comparison of both Nash approaches
    """
    print("\nNash Equilibrium Comparison:")
    
    me_ordered = comparison_df.sort_values('ME Nash Probability', ascending=False)
    print("\nAgents by Max Entropy Nash probability:")
    for i, row in me_ordered.iterrows():
        print(f"{row['Agent']}: {row['ME Nash Probability']:.4f}")
    
    rd_ordered = comparison_df.sort_values('RD Nash Probability', ascending=False)
    print("\nAgents by Replicator Dynamics Nash probability:")
    for i, row in rd_ordered.iterrows():
        print(f"{row['Agent']}: {row['RD Nash Probability']:.4f}")
    
    comparison_df['Probability Difference'] = np.abs(comparison_df['ME Nash Probability'] - comparison_df['RD Nash Probability'])
    comparison_df['Regret Difference'] = np.abs(comparison_df['ME Nash Regret'] - comparison_df['RD Nash Regret'])
    
    print("\nAgents with largest discrepancy between Nash concepts:")
    top_diff = comparison_df.sort_values('Probability Difference', ascending=False).head(3)
    for i, row in top_diff.iterrows():
        print(f"{row['Agent']}: ME prob={row['ME Nash Probability']:.4f}, RD prob={row['RD Nash Probability']:.4f}")
    
    # NOTE: might be helpful? need more data? 
    me_probs = comparison_df['ME Nash Probability'].values
    rd_probs = comparison_df['RD Nash Probability'].values
    try:
        corr, p = pearsonr(me_probs, rd_probs)
        print(f"\nCorrelation between ME and RD Nash probabilities: {corr:.4f} (p={p:.4f})")
    except:
        print("\nCould not calculate correlation (scipy.stats not available)")
    
    print("\nCorrelation between ME and RD Nash regrets:")
    me_regrets = comparison_df['ME Nash Regret'].values
    rd_regrets = comparison_df['RD Nash Regret'].values
    try:
        corr, p = pearsonr(me_regrets, rd_regrets)
        print(f"Correlation: {corr:.4f} (p={p:.4f})")
    except:
        print("Could not calculate correlation (scipy.stats not available)") 
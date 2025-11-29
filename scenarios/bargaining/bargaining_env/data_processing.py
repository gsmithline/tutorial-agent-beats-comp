#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import math
import random
from itertools import product
import time
import re
import nashpy as nash  # Added for company-level meta-game NE computation
import warnings
import cvxpy as cp   

DEFAULT_DISCOUNT_FACTOR = 0.9

def parse_values(line):
    """Parse the line containing private values"""
    # Format: "Your private values are 16 for item 1, 46 for item 2, 100 for item 3, 23 for item 4, 47 for item 5."
    values = []
    parts = line.split("are ")[1].split(", ")
    for part in parts:
        value = int(part.split(" for item")[0])
        values.append(value)
    return values

def parse_outside_offer(line):
    """Parse the line containing outside offer"""
    # Format: "Your outside offer value is 145. Your objective..."
    return int(line.split("value is ")[1].split(".")[0])

def calculate_value(items_received, values):
    """Calculate value of items received given the player's values"""
    return sum(items_received[i] * values[i] for i in range(len(values)))

def compute_max_nash_welfare(item_counts, p1_valuations, p2_valuations):
    """Compute the maximum Nash welfare for a given set of item counts and valuations."""
    if len(item_counts) != len(p1_valuations) or len(item_counts) != len(p2_valuations):
        raise ValueError("item_counts, p1_valuations, p2_valuations must have the same length.")

    K = len(item_counts)
    max_nash = -1.0
    best_alloc = None
    outside_offer_player1 = np.random.randint(1, np.dot(item_counts, p1_valuations))
    outside_offer_player2 = np.random.randint(1, np.dot(item_counts, p2_valuations))
    ranges = [range(n_i + 1) for n_i in item_counts] 
    for allocation in product(*ranges):
        p1_util = 0.0
        p2_util = 0.0
        for i in range(K):
            x_i = allocation[i]
            n_i = item_counts[i]
            p1_util += x_i * p1_valuations[i]
            p2_util += (n_i - x_i) * p2_valuations[i]

        w = math.sqrt(max(p1_util, 0) * max(p2_util, 0))

        if w > max_nash:
            max_nash = w
            best_alloc = allocation

        #outside offer check
        if max_nash < math.sqrt(outside_offer_player1 * outside_offer_player2):
            max_nash = math.sqrt(outside_offer_player1 * outside_offer_player2)
            best_alloc = [0, 0, 0, 0, 0]

    return max_nash, list(best_alloc)

def analyze_single_game(file_path, discount_factor=DEFAULT_DISCOUNT_FACTOR):
    """
    Analyze a single game JSON file and extract relevant metrics.
    
    Args:
        file_path: Path to the JSON file containing the game data
        discount_factor: Discount factor to apply to utilities (default: DEFAULT_DISCOUNT_FACTOR)
        
    Returns:
        List of dictionaries containing analyzed game metrics
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    for game in data['all_game_data']:
        agent1 = game['agent1']
        agent2 = game['agent2']
        
        # Remove Agent1_ and Agent2_ prefixes if they exist
        if agent1.startswith("Agent1_"):
            agent1 = agent1[7:]
        if agent2.startswith("Agent2_"):
            agent2 = agent2[7:]
        
        # Track the final state
        final_action = None
        final_round_index = len(game['round_data']) - 1
        # Convert to actual round number (2 turns = 1 round)
        final_round_number = (final_round_index // 2) + 1
        final_offer = None
        p1_outside_offer = None
        p2_outside_offer = None
        p1_values = None
        p2_values = None
        num_items = None
        full_items = None

        for round_idx, round_data in enumerate(game['round_data']):
            prompt = round_data['prompt']
            
            if round_idx == final_round_index:
                final_action = round_data['action']
            
            if "You are Player 1" in prompt:
                for line in prompt.split('\n'):
                    if "Your private values are" in line:
                        p1_values = parse_values(line)
                    elif "Your outside offer value is" in line:
                        p1_outside_offer = parse_outside_offer(line)
                    elif "There are" in line and ("unit of item" in line or "units of item" in line):
                        if num_items is None:
                           
                            nums = []
                            # Extract all leading integers preceding 'unit'/'units'
                            matches = re.findall(r"(\d+)\s+unit(?:s)?\s+of\s+item", line)
                            if matches:
                                nums = [int(m) for m in matches]
                            else:
                                # Fallback: attempt previous comma split with regex extraction
                                parts = line.split("There are ")[1].split(", ")
                                for part in parts:
                                    m = re.search(r"(\d+)", part)
                                    if m:
                                        nums.append(int(m.group(1)))
                            num_items = nums
                    
            elif "You are Player 2" in prompt:
                for line in prompt.split('\n'):
                    if "Your private values are" in line:
                        p2_values = parse_values(line)
                    elif "Your outside offer value is" in line:
                        p2_outside_offer = parse_outside_offer(line)
            
            # Track the current offer
            if "Current offer on the table" in prompt:
                offer_line = prompt.split("Current offer on the table")[1].split("\n")[0]
                if "None" not in offer_line and "[" in offer_line:
                    try:
                        offer_str = offer_line[offer_line.find("["):offer_line.find("]")+1]
                        final_offer = eval(offer_str)
                        if num_items is not None and len(final_offer) != len(num_items):
                            print(f"Warning: Offer length mismatch in game {game.get('game_id', 'N/A')}. Offer: {final_offer}, Items: {num_items}")
                            final_offer = None 
                    except (SyntaxError, NameError, TypeError) as e:
                        print(f"Warning: Error parsing final offer in game {game.get('game_id', 'N/A')}. Offer line: '{offer_line}'. Error: {e}")
                        final_offer = None

        full_items = num_items

        is_walk_optimal = None # Initialize
        is_agree_optimal = None  # Initialize to ensure defined even if calculations are skipped
        if p1_values is not None and p2_values is not None and full_items is not None and p1_outside_offer is not None and p2_outside_offer is not None:
            try:
                items_np = np.array(full_items)
                p1_values_np = np.array(p1_values)
                p2_values_np = np.array(p2_values)

                total_value1 = np.dot(items_np, p1_values_np)
                total_value2 = np.dot(items_np, p2_values_np)

                if total_value1 > 1 and total_value2 > 1: #Eensure positive total values
                    walk_welfare = p1_outside_offer + p2_outside_offer

                    list_total_items = [0.0, 0.0]
                    min_diff = np.inf
                    min_index = -1

                    for i in range(len(items_np)):
                        if items_np[i] == 0:
                            continue

                        diff = p1_values_np[i] - p2_values_np[i]
                        if abs(diff) < min_diff:
                            min_diff = abs(diff)
                            min_index = i

                        if diff > 0:
                            list_total_items[0] += items_np[i] * p1_values_np[i]
                        else:
                            list_total_items[1] += items_np[i] * p2_values_np[i]

                    if min_index != -1:
                         if list_total_items[0] == 0 and list_total_items[1] > 0:
                             if p2_values_np[min_index] > 0:
                                 list_total_items[0] += items_np[min_index] * p1_values_np[min_index] 
                                 # list_total_items[1] -= items_np[min_index] * p2_values_np[min_index]
                         elif list_total_items[1] == 0 and list_total_items[0] > 0:
                             if p1_values_np[min_index] > 0:
                                 list_total_items[1] += items_np[min_index] * p2_values_np[min_index] # Assign disputed item to P2
                                 # list_total_items[0] -= items_np[min_index] * p1_values_np[min_index] 

                    list_total_items[0] = max(0, list_total_items[0])
                    list_total_items[1] = max(0, list_total_items[1])
                    internal_welfare = sum(list_total_items)

                    is_walk_optimal = walk_welfare > internal_welfare
                    is_agree_optimal = None
                    if is_walk_optimal is not None:
                         is_agree_optimal = not is_walk_optimal 
                else:
                    is_walk_optimal = False 
                    is_agree_optimal = False 

            except Exception as e:
                print(f"Warning: Error during optimal walk calculation for game {game.get('game_id', 'N/A')}: {e}")
                is_walk_optimal = None 
                is_agree_optimal = None 


        p1_final_value = None
        p2_final_value = None
        print(f"Discount factor in use {discount_factor}")
        round_discount = discount_factor ** (final_round_number - 1)
        p1_items = None
        p2_items = None
        if final_action == "WALK" or final_action == "INVALID WALK":
            if final_round_number == 1 and "You are Player 1" in game['round_data'][final_round_index]['prompt']:
                p1_final_value = None
                p2_final_value = None
            else:
                p1_final_value = p1_outside_offer * round_discount
                p2_final_value = p2_outside_offer * round_discount
        elif final_action == "ACCEPT":
            accepting_player = None
            for round_idx, round_data in enumerate(game['round_data']):
                if round_idx == final_round_index and round_data['action'] == "ACCEPT":
                    accepting_player = 1 if "You are Player 1" in round_data['prompt'] else 2
            
            if accepting_player == 2:
                p2_items = final_offer
                p1_items = [num_items[i] - final_offer[i] for i in range(len(final_offer))]
            else:
                p1_items = final_offer
                p2_items = [num_items[i] - final_offer[i] for i in range(len(final_offer))]
            
            p1_final_value = calculate_value(p1_items, p1_values) * round_discount
            p2_final_value = calculate_value(p2_items, p2_values) * round_discount

        elif final_action == "INVALID WALK":
            print(f"Invalid walk in game {game['game_id']}")
            
        try:
            circle_data = data['all_game_data'][0]['circle']
            if isinstance(circle_data, int):
                p1_circle = circle_data
                p2_circle = circle_data
            else:
                p1_circle = circle_data[0]
                p2_circle = circle_data[1]
        except (KeyError, TypeError, IndexError):
            # Handle missing or malformed data
            p1_circle = None
            p2_circle = None
        
        # Determine which player role took the final action
        final_actor_role = None
        if final_action:
            final_prompt = game['round_data'][final_round_index]['prompt']
            if "You are Player 1" in final_prompt:
                final_actor_role = 'agent1' # Corresponds to the agent1 key in the result dict
            elif "You are Player 2" in final_prompt:
                final_actor_role = 'agent2' # Corresponds to the agent2 key in the result dict

        # Add circle values to agent names if available
        agent1_with_circle = f"{agent1}_circle_{p1_circle}" if p1_circle is not None else agent1
        agent2_with_circle = f"{agent2}_circle_{p2_circle}" if p2_circle is not None else agent2
      
        results.append({
            'agent1': agent1_with_circle,
            'agent2': agent2_with_circle,
            'final_action': final_action,
            'final_actor_role': final_actor_role, # Store the role (agent1/agent2) that acted last
            'final_round': final_round_number,
            'discount_factor': round_discount,
            'agent1_value': p1_final_value,
            'agent2_value': p2_final_value,
            'p1_values': p1_values,
            'p2_values': p2_values,
            'p1_items': p1_items,
            'p2_items': p2_items,
            'full_items': num_items,
            'num_items': len(p1_values) if p1_values else None,
            'p1_outside_offer': p1_outside_offer,
            'p2_outside_offer': p2_outside_offer,
            'is_walk_optimal': is_walk_optimal,
            'is_agree_optimal': is_agree_optimal
        })

        game_result = results[-1] # Get the dict we just appended
        v1 = game_result.get('agent1_value')
        v2 = game_result.get('agent2_value')
        game_util_welfare = None
        game_nash_welfare = None
        game_nash_welfare_adv = None
        if v1 is not None and v2 is not None:
            v1_undiscounted = v1 / game_result['discount_factor'] if game_result['discount_factor'] else v1
            v2_undiscounted = v2 / game_result['discount_factor'] if game_result['discount_factor'] else v2
            game_util_welfare = v1_undiscounted + v2_undiscounted
            # game_util_welfare = v1 + v2
            try:
                term1 = max(0, v1_undiscounted)
                term2 = max(0, v2_undiscounted)
                game_nash_welfare = math.sqrt(term1 * term2)
                # game_nash_welfare_adv = math.sqrt(max(0, term1 - (game_result['p1_outside_offer'] 
                #                                       * game_result['discount_factor'])) 
                #                                       * max(0, term2 - (game_result['p2_outside_offer'] 
                #                                                         * game_result['discount_factor'])))
                game_nash_welfare_adv = math.sqrt(max(0, term1 - game_result['p1_outside_offer'])
                                                      * max(0, term2 - game_result['p2_outside_offer'] 
                                                                        ))

            except (TypeError, ValueError) as e:
                print(f"Warning: Could not calculate Nash welfare for game {game.get('game_id', 'N/A')}: v1={v1_undiscounted}, v2={v2_undiscounted}, Error: {e}")
                game_nash_welfare = None 
                game_nash_welfare_adv = None

        results[-1]['utilitarian_welfare'] = game_util_welfare
        results[-1]['nash_welfare'] = game_nash_welfare
        results[-1]['nash_welfare_adv'] = game_nash_welfare_adv

        game_is_ef1 = None 
        if final_action == "ACCEPT" and all(x is not None for x in [p1_items, p2_items, p1_values, p2_values]):
            try:
                p1_vals_np = np.array(p1_values, dtype=float)
                p2_vals_np = np.array(p2_values, dtype=float)
                p1_items_np = np.array(p1_items, dtype=int)
                p2_items_np = np.array(p2_items, dtype=int) 

                if p1_vals_np.shape == p1_items_np.shape and p2_vals_np.shape == p2_items_np.shape and p1_vals_np.ndim == 1:
                    p1_own_bundle_value = np.dot(p1_vals_np, p1_items_np)
                    p1_other_bundle_value = np.dot(p1_vals_np, p2_items_np)
                    p2_own_bundle_value = np.dot(p2_vals_np, p2_items_np)
                    p2_other_bundle_value = np.dot(p2_vals_np, p1_items_np)

                    p1_is_envy_free = p1_own_bundle_value >= p1_other_bundle_value
                    p2_is_envy_free = p2_own_bundle_value >= p2_other_bundle_value

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
                    #          if p1_items_np[j] > 0: 
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
                    game_is_ef1 = p1_is_ef1 and p2_is_ef1

                else:
                     print(f"Warning: Shape mismatch during EF1 calc for game {game.get('game_id', 'N/A')}. Skipping EF1.")

            except Exception as e:
                print(f"Warning: Error calculating EF1 for game {game.get('game_id', 'N/A')}: {e}")
                game_is_ef1 = None 
        else:
            game_is_ef1 = None

        results[-1]['is_ef1'] = game_is_ef1

    return results

def get_canonical_name(agent_name):
    """Convert agent names to canonical format for data loading"""
    model_mapping = {
        'openai_4o_circle_4': 'openai_4o_2024-08-06_circle_4',
        'openai_4o_circle_5': 'openai_4o_2024-08-06_circle_5',
        'openai_4o_circle_6': 'openai_4o_2024-08-06_circle_6',
        'gemini_2.0_flash_circle_2': 'gemini_2.0_flash_circle_2', 
        'gemini_2.0_flash_circle_5': 'gemini_2.0_flash_circle_5',  
        
        'anthropic_3.7_sonnet_circle_5': 'anthropic_3.7_sonnet_2025-02-19_circle_5',
        'anthropic_3.7_sonnet_circle_6': 'anthropic_3.7_sonnet_2025-02-19_circle_6',
        "anthropic_sonnet_3.7_reasoning_circle_0": "anthropic_sonnet_3.7_reasoning_2025-02-19_circle_0",
        "rnad_agent_circle_0": "rnad_agent_circle_0",
        
        'openai_o3_mini_circle_0': 'openai_o3_mini_2025-01-31_circle_0',
        'soft_agent_circle_0': 'soft_agent_circle_0',
        'tough_agent_circle_0': 'tough_agent_circle_0',
        'nfsp_agent_circle_0': 'nfsp_agent_circle_0',
        'nfsp_agent_2_circle_0': 'nfsp_agent_2_circle_0',
        'walk_agent_circle_0': 'walk_agent_circle_0',
        'conceder_agent_circle_0': 'conceder_agent_circle_0'

    }
    
    return model_mapping.get(agent_name, agent_name)

def get_display_name(agent_name):
    """Convert agent names to a short display format like '4o-c-5'."""
    
    base_to_short_display_map = {
        'openai_4o_circle_4': '4o-c-4',
        'openai_4o_circle_5': '4o-c-5',
        'openai_4o_circle_6': '4o-c-6',
        'gemini_2.0_flash_circle_2': "gem-2.0-f-c-2",
        'gemini_2.0_flash_circle_5': "gem-2.0-f-c-5",
        'anthropic_3.7_sonnet_circle_5': 'sonnet-3.7-c-5',
        'anthropic_3.7_sonnet_circle_6': 'sonnet-3.7-c-6',
        "anthropic_sonnet_3.7_reasoning_circle_0": 'sonnet-3.7-r-c-0',
        'openai_o3_mini_circle_0': 'o3-mini-c-0',
        'soft_agent_circle_0': 'soft',
        'tough_agent_circle_0': 'tough',
        'nfsp_agent_circle_0': 'nfsp',
        'nfsp_agent_2_circle_0': 'nfsp',
        'walk_agent_circle_0': 'walk',
        'rnad_agent_circle_0': 'rnad',
        'conceder_agent_circle_0': 'aspire'
    }

    # Remove date to get the base name (e.g., "openai_4o_circle_5" from "openai_4o_2024-08-06_circle_5")
    base_name = re.sub(r'_\d{4}-\d{2}-\d{2}', '', agent_name)

    if base_name in base_to_short_display_map:
        short_name = base_to_short_display_map[base_name]
        return short_name # Return only the short name
    else:
       
        return base_name

def compute_global_max_values(num_samples=1000000):
    """Compute global maximum values for Nash welfare and social welfare for comparison."""
    global_max_nash_welfare = []
    global_max_social_welfare = []

    for _ in range(num_samples):
        items = np.random.poisson(4, 5) + 1
        player_values1 = np.random.randint(1, 101, 5)
        player_values2 = np.random.randint(1, 101, 5)

        max_nash, _ = compute_max_nash_welfare(items, player_values1, player_values2)
        global_max_nash_welfare.append(max_nash)

        best_value_per_item = np.maximum(player_values1, player_values2)
        global_max_social_welfare.append(np.dot(items, best_value_per_item))

    return np.mean(global_max_nash_welfare), np.mean(global_max_social_welfare)


def compute_frobenius_norm(matrix1, matrix2):
    """
    Compute the Frobenius norm between two matrices.
    
    Args:
        matrix1: First matrix (numpy array or pandas DataFrame)
        matrix2: Second matrix (numpy array or pandas DataFrame)
        
    Returns:
        float: Frobenius norm between the two matrices
    """
    if isinstance(matrix1, pd.DataFrame):
        matrix1 = matrix1.to_numpy()
    if isinstance(matrix2, pd.DataFrame):
        matrix2 = matrix2.to_numpy()
    
    matrix1 = np.nan_to_num(matrix1)
    matrix2 = np.nan_to_num(matrix2)
    
    if matrix1.shape != matrix2.shape:
        raise ValueError(f"Matrices must have the same shape. Got {matrix1.shape} and {matrix2.shape}")
    
    return np.linalg.norm(matrix1 - matrix2, 'fro')

def compute_best_response_matrix(performance_matrix):
    """
    Compute a best response matrix from a performance matrix.
    
    Args:
        performance_matrix: Performance matrix DataFrame where entry (i,j) is player i's payoff against player j
        
    Returns:
        numpy.ndarray: Best response matrix where entry (i,j) is 1 if player i is (one of) the best responses to player j, 0 otherwise
    """
    agents = performance_matrix.index.tolist()
    n_agents = len(agents)
    best_response_matrix = np.zeros((n_agents, n_agents))

    # Convert to numeric values and handle NaNs
    perf = performance_matrix.astype(float)

    eps = 1e-9  # tolerance for numerical equality

    for j in range(n_agents):
        col = perf.iloc[:, j]

        if col.isna().all():
            continue  # no data for this opponent

        max_val = col.max(skipna=True)

        # Mark every strategy within eps of the max as a best response
        for i in range(n_agents):
            try:
                val = col.iloc[i]
            except Exception:
                val = np.nan
            if np.isnan(val):
                continue
            if abs(val - max_val) <= eps:
                best_response_matrix[i, j] = 1

    return best_response_matrix

def nonparametric_bootstrap_best_response(games_batch, num_bootstrap=1000, all_known_agents=None, random_seed=42):
    """
    Perform non-parametric bootstrap on a batch of games and compute the average best response graph.
    Uses the same methodology as nonparametric_bootstrap_from_raw_data for consistency.
    
    Args:
        games_batch: List of game results
        num_bootstrap: Number of bootstrap samples to generate
        all_known_agents: Complete list of all agents (to ensure consistent matrix dimensions)
        random_seed: Fixed random seed for reproducibility
        
    Returns:
        numpy.ndarray: Average best response matrix across all bootstrap samples
    """
    # Set fixed random seed for reproducibility
    np.random.seed(random_seed)
    
    # Ensure we have a valid agent list
    if not all_known_agents or len(all_known_agents) == 0:
        # Fallback: extract agents from the batch if no list is provided
        unique_agents = set()
        for game in games_batch:
            if game['agent1']:
                unique_agents.add(game['agent1'])
            if game['agent2']:
                unique_agents.add(game['agent2'])
        all_agents = sorted(list(unique_agents))
        print(f"WARNING: No valid agent list provided, extracted {len(all_agents)} agents from batch")
    else:
        all_agents = all_known_agents
    
    # Print some debug info
    print(f"Bootstrap: {len(games_batch)} games, {len(all_agents)} agents, {num_bootstrap} iterations, seed={random_seed}")
    
    # Initialize array to accumulate best response matrices
    accumulated_br_matrix = None
    
    # For each bootstrap iteration
    for b in range(num_bootstrap):
        # Generate a bootstrap sample by resampling ENTIRE GAMES with replacement
        # This is identical to the approach in nonparametric_bootstrap_from_raw_data
        bootstrap_indices = np.random.choice(
            range(len(games_batch)), 
            size=len(games_batch), 
            replace=True
        )
        
        # Get the resampled games
        resampled_games = [games_batch[i] for i in bootstrap_indices]
        
        # Compute performance matrix from this bootstrap sample
        agent_performance = defaultdict(lambda: defaultdict(list))
        
        # Process all resampled games
        for game in resampled_games:
            agent1 = game.get('agent1')
            agent2 = game.get('agent2')
            
            # Skip games with missing agents
            if not agent1 or not agent2:
                continue
                
            # Ensure string type for both agents
            agent1 = str(agent1)
            agent2 = str(agent2)
            
            # Record agent1's performance against agent2
            if 'agent1_value' in game and game['agent1_value'] is not None:
                agent_performance[agent1][agent2].append(game['agent1_value'])
                
            # Record agent2's performance against agent1
            if 'agent2_value' in game and game['agent2_value'] is not None:
                agent_performance[agent2][agent1].append(game['agent2_value'])
        
        # Create performance matrix with all known agents
        performance_matrix = pd.DataFrame(index=all_agents, columns=all_agents)
        
        # Fill the matrix with average performance values
        for agent1 in all_agents:
            for agent2 in all_agents:
                values = agent_performance[agent1][agent2]
                if values:
                    performance_matrix.loc[agent1, agent2] = np.mean(values)
                else:
                    performance_matrix.loc[agent1, agent2] = np.nan
        
        # Convert matrix to numpy array for computation
        game_matrix_np = performance_matrix.to_numpy().astype(float)
        
        for i in range(game_matrix_np.shape[0]):
            for j in range(game_matrix_np.shape[1]):
                if np.isnan(game_matrix_np[i, j]):
                    col_slice = game_matrix_np[:, j].astype(float)
                    if np.any(~np.isnan(col_slice)):
                        col_mean = np.nanmean(col_slice)
                        game_matrix_np[i, j] = col_mean
                    else:
                        # Check if row has any non-NaN values before computing mean
                        row_slice = game_matrix_np[i, :].astype(float)
                        if np.any(~np.isnan(row_slice)):
                            # Fall back to row mean if column mean is not available
                            row_mean = np.nanmean(row_slice)
                            game_matrix_np[i, j] = row_mean
                        else:
                            # If both column and row are all NaN, set to 0
                            game_matrix_np[i, j] = 0
        
        # Compute best response matrix for this bootstrap sample
        br_matrix = compute_best_response_matrix(pd.DataFrame(game_matrix_np, index=all_agents, columns=all_agents))
        
        # Accumulate the best response matrix
        if accumulated_br_matrix is None:
            accumulated_br_matrix = br_matrix
        else:
            accumulated_br_matrix += br_matrix
        
        if b % 100 == 0 and b > 0:
            print(f"  - Completed {b}/{num_bootstrap} bootstrap iterations...")
    
    avg_br_matrix = accumulated_br_matrix / num_bootstrap
    
    print(f"Average BR matrix shape: {avg_br_matrix.shape}, sum: {np.sum(avg_br_matrix)}")
    
    return avg_br_matrix

def plot_frobenius_norm_evolution(batch_numbers, frobenius_norms, output_dir=None, filename="frobenius_norm_evolution.png"):
    """
    Plot the evolution of Frobenius norm between consecutive best response graphs.
    
    Args:
        batch_numbers: List of batch numbers
        frobenius_norms: List of Frobenius norms
        output_dir: Directory to save the plot
        filename: Name of the output file
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batch_numbers, frobenius_norms, marker='o', linestyle='-')
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Evolution of Best Response Graph (Frobenius Norm between Consecutive Batches)')
    ax.grid(True)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    
    return fig

 

def _infer_company(agent_name: str) -> str:
    """Infer provider/company from an agent string (lower-case heuristics).
    Falls back to 'Unknown'.
    This duplicates logic embedded later in process_all_games so that the
    helper can be reused when computing company-level meta-games.
    """
    lower = agent_name.lower()
    if any(lower.startswith(p) or p in lower for p in ("gem", "gemini")):
        return "Gemini"
    if "sonnet" in lower:
        return "Anthropic"
    if any(lower.startswith(p) or p in lower for p in ("4o", "o3", "gpt", "openai")):
        return "OpenAI"
    return "Unknown"


def _impute_nan(mat: np.ndarray) -> np.ndarray:
    """Impute NaNs in a 2-D numpy array using column mean, then row mean, then 0."""
    mat = mat.astype(float)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                # column mean
                col = mat[:, j]
                if np.any(~np.isnan(col)):
                    mat[i, j] = np.nanmean(col)
                else:
                    row = mat[i, :]
                    if np.any(~np.isnan(row)):
                        mat[i, j] = np.nanmean(row)
                    else:
                        mat[i, j] = 0.0
    return mat


def _agent_perf_to_company_perf(perf_df: pd.DataFrame, *, return_strategies: bool = False, use_average: bool = False):
    """Collapse an agent-level performance matrix to a company-level matrix using
    Nash equilibrium between the two companies' agent sets.

    Args:
        perf_df: DataFrame with agents as both index and columns, where each
                 entry is the mean payoff of the row agent against the column
                 agent (row player's utility).

    Returns:
        DataFrame indexed and columned by company names with expected row
        payoffs computed from NE of each sub-game.  Asymmetric nature is kept.
    """
    agents = perf_df.index.tolist()
    company_of = {a: _infer_company(a) for a in agents}
    company_agents = defaultdict(list)
    for a, c in company_of.items():
        company_agents[c].append(a)
    companies = sorted(company_agents.keys())
    n_comp = len(companies)

    company_perf = pd.DataFrame(np.nan, index=companies, columns=companies)
    # When requested, store the NE strategies for every ordered company pair.
    strategy_dict = {} if return_strategies else None

    for A in companies:
        for B in companies:
            SA = company_agents[A]
            SB = company_agents[B]
            if not SA or not SB:
                continue

            if use_average:
                # Simple average of all agent-level payoffs (ignoring NaNs)
                sub_matrix = perf_df.loc[SA, SB].to_numpy(dtype=float)
                sub_matrix = _impute_nan(sub_matrix.copy())
                expected_row_payoff = float(np.mean(sub_matrix))
                company_perf.loc[A, B] = expected_row_payoff
                if return_strategies:
                    strategy_dict[(A, B)] = {
                        "row_agents": SA,
                        "sigma": np.full(len(SA), 1.0/len(SA)),  # uniform for reporting
                        "col_agents": SB,
                        "tau": np.full(len(SB), 1.0/len(SB))
                    }
            else:
                # Nash-equilibrium collapse
                R = perf_df.loc[SA, SB].to_numpy(dtype=float)
                C = perf_df.loc[SB, SA].T.to_numpy(dtype=float)
                R = _impute_nan(R.copy())
                C = _impute_nan(C.copy())
                sigma, tau = _solve_bimatrix_ne(R, C)
                expected_row_payoff = float(sigma @ R @ tau)
                company_perf.loc[A, B] = expected_row_payoff

                if return_strategies:
                    strategy_dict[(A, B)] = {
                        "row_agents": SA,
                        "sigma": sigma.copy(),
                        "col_agents": SB,
                        "tau": tau.copy()
                    }

    if return_strategies:
        return company_perf, strategy_dict
    return company_perf

def nonparametric_bootstrap_best_response_with_matrix_weighting(all_game_results, all_known_agents, percent_data, num_bootstrap=1000, random_seed=42, use_company_meta_game=True, company_pooling_average=False):
    """
    Perform bootstrap-level 1/3 weighting for best response evolution analysis.
    Each cell is bootstrapped separately from 3 underlying game matrices, then combined with equal weights.
    
    Args:
        all_game_results: List of all game results (with 'source_matrix' field)
        all_known_agents: Complete list of all agents
        percent_data: Percentage of data to use (e.g., 0.1 for 10%, 0.2 for 20%)
        num_bootstrap: Number of bootstrap samples to generate
        random_seed: Fixed random seed for reproducibility
        
    Returns:
        tuple: (avg_br_matrix, me_nash_strategy, max_me_regret, avg_me_regret)
    """
    np.random.seed(random_seed)
    
    print(f"\n=== Bootstrap-level 1/N weighting for BR evolution ===")
    
    # Separate games by source matrix
    games_by_source = {
        'game_matrix_1a': [],
        'game_matrix_2a': [],
        'game_matrix_3a': [], 
        # 'game_matrix_1': [],
        # 'game_matrix_2': [],
        # 'game_matrix_3': []
    }
    
    for game in all_game_results:
        source = game.get('source_matrix')
        if source in games_by_source:
            games_by_source[source].append(game)
    
    # print(f"Games per matrix: gm1={len(games_by_source['game_matrix_1a'])}, "
    #       f"gm2={len(games_by_source['game_matrix_2a'])}, gm3={len(games_by_source['game_matrix_3a'])}")
    # print(f"Games per matrix: gm1={len(games_by_source['game_matrix_1'])}, "
    #       f"gm2={len(games_by_source['game_matrix_2'])}, gm3={len(games_by_source['game_matrix_3'])}")
    # Organize data by cell for each source matrix
    cell_data_by_source = {}
    
    for source in ['game_matrix_1a', 'game_matrix_2a', 'game_matrix_3a']: #, 'game_matrix_3a']:
        cell_data = defaultdict(list)
        
        for game in games_by_source[source]:
            agent1 = game.get('agent1')
            agent2 = game.get('agent2')
            
            if not agent1 or not agent2:
                continue
            if agent1 not in all_known_agents or agent2 not in all_known_agents:
                continue
            if 'agent1_value' not in game or 'agent2_value' not in game:
                continue
            if game['agent1_value'] is None or game['agent2_value'] is None:
                continue
                
            # Store both agent performance values together (from the same game)
            cell_data[(agent1, agent2)].append((game['agent1_value'], game['agent2_value']))
        
        cell_data_by_source[source] = cell_data
    
    # For each source and agent pair, take only the specified percentage of data
    cumulative_cell_data_by_source = {}

    for source in ['game_matrix_1a', 'game_matrix_2a', 'game_matrix_3a']: # 'game_matrix_3a']:
        cumulative_cell_data = {}
        
        for agent_pair, values in cell_data_by_source[source].items():
            if not values:
                continue
                
            agent1, agent2 = agent_pair
            
            n_values = max(1, int(len(values) * percent_data))
            
            selected_values = values[:n_values]
            
            for i in range(len(selected_values)):
                if (agent1, agent2) not in cumulative_cell_data:
                    cumulative_cell_data[(agent1, agent2)] = []
                cumulative_cell_data[(agent1, agent2)].append(selected_values[i][0])  # agent1's value
                
                
                if (agent2, agent1) not in cumulative_cell_data:
                    cumulative_cell_data[(agent2, agent1)] = []
                cumulative_cell_data[(agent2, agent1)].append(selected_values[i][1])  # agent2's value
        
        cumulative_cell_data_by_source[source] = cumulative_cell_data
    
    all_agent_pairs = set()
    for source_data in cumulative_cell_data_by_source.values():
        all_agent_pairs.update(source_data.keys())
    
    total_pairs = len(all_agent_pairs)
    print(f"Using {percent_data:.1%} of data across {total_pairs} unique agent pair cells")
    
    accumulated_br_matrix = None
    
    for b in range(num_bootstrap):
        performance_matrix = pd.DataFrame(index=all_known_agents, columns=all_known_agents)
        
        for agent1 in all_known_agents:
            for agent2 in all_known_agents:
                cell_key = (agent1, agent2)
                
                source_values = []
                source_count = 0

                for source in ['game_matrix_1a', 'game_matrix_2a', 'game_matrix_3a']: #bootstrap each matrix seperatley
                    source_cell_data = cumulative_cell_data_by_source[source]
                    
                    if cell_key in source_cell_data and source_cell_data[cell_key]:
                        values = source_cell_data[cell_key]
                        bootstrap_indices = np.random.choice(range(len(values)), size=len(values), replace=True)
                        resampled_values = [values[i] for i in bootstrap_indices]
                        source_mean = np.mean(resampled_values)
                        source_values.append(source_mean)
                        source_count += 1
                
                # Combine with equal weighting (1/3 each, or 1/N if fewer sources have data)
                if source_values:
                    # Equal weighting across available sources
                    combined_value = np.mean(source_values)
                    performance_matrix.loc[agent1, agent2] = combined_value
                else:
                    performance_matrix.loc[agent1, agent2] = np.nan
        
        game_matrix_np = performance_matrix.to_numpy().astype(float)
        
        for i in range(game_matrix_np.shape[0]):
            for j in range(game_matrix_np.shape[1]):
                if np.isnan(game_matrix_np[i, j]):
                    col_slice = game_matrix_np[:, j].astype(float)
                    if np.any(~np.isnan(col_slice)):
                        col_mean = np.nanmean(col_slice)
                        game_matrix_np[i, j] = col_mean
                    else:
                        row_slice = game_matrix_np[i, :].astype(float)
                        if np.any(~np.isnan(row_slice)):
                            row_mean = np.nanmean(row_slice)
                            game_matrix_np[i, j] = row_mean
                        else:
                      
                            game_matrix_np[i, j] = 0
        
        perf_df_for_br = pd.DataFrame(game_matrix_np, index=all_known_agents, columns=all_known_agents)
        if use_company_meta_game:
            perf_df_for_br, strategy_info = _agent_perf_to_company_perf(perf_df_for_br, return_strategies=True, use_average=company_pooling_average)
           

            if 'strategy_sums' not in locals():
                strategy_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
                company_names_cached = sorted(perf_df_for_br.index.tolist())

            for (row_comp, col_comp), info in strategy_info.items():
                for agent_name, prob in zip(info['row_agents'], info['sigma']):
                    strategy_sums[row_comp][col_comp][agent_name] += float(prob)

        br_matrix = compute_best_response_matrix(perf_df_for_br)
        #print(f"Best Response Matrix {br_matrix}")
        # Accumulate the best response matrix
        if accumulated_br_matrix is None:
            accumulated_br_matrix = br_matrix
        else:
            accumulated_br_matrix += br_matrix
        
       
    
    avg_br_matrix = accumulated_br_matrix / num_bootstrap
    print(f"avg best response matrix: {avg_br_matrix}")
    print(f"Bootstrap-level 1/3 weighting complete. BR matrix shape: {avg_br_matrix.shape}")
    print(f"Matrix sum: {np.sum(avg_br_matrix):.4f}")
    
    if use_company_meta_game and 'strategy_sums' in locals():
        print("\nAverage equilibrium mix each provider plays (row provider) against every opponent (column provider):")
        for row_comp in company_names_cached:
            for col_comp in company_names_cached:
                mix_dict = strategy_sums.get(row_comp, {}).get(col_comp, {})
                if not mix_dict:
                    continue
                mix_avg = {agent: prob / num_bootstrap for agent, prob in mix_dict.items()}
                sorted_mix = sorted(mix_avg.items(), key=lambda x: -x[1])
                mix_str = ", ".join([f"{agent}: {p:.2f}" for agent, p in sorted_mix])
                print(f"  {row_comp} vs {col_comp}: {mix_str}")
        print("-" * 60)
    
    return avg_br_matrix, 0, 0, 0  # Placeholders for ME Nash metrics

def compute_euclidean_distance(vector1, vector2):
    """
    Compute the Euclidean distance between two vectors.
    
    Args:
        vector1: First vector (numpy array)
        vector2: Second vector (numpy array)
        
    Returns:
        float: Euclidean distance
    """
    # Ensure vectors have the same shape
    if vector1.shape != vector2.shape:
        raise ValueError(f"Vectors must have the same shape. Got {vector1.shape} and {vector2.shape}")
    
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

def plot_metric_evolution(data_points, metric_values, metric_name, output_dir=None, filename=None):
    """
    Plot the evolution of a metric across data points
    Args:
        data_points: List of x-axis values (batch numbers or percentages)
        metric_values: List of y-axis values (metric values)
        metric_name: Name of the metric being plotted
        output_dir: Directory to save the plot
        filename: Name of the output file
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    import matplotlib.pyplot as plt
    
    if filename is None:
        filename = f"{metric_name.lower().replace(' ', '_')}_evolution.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data_points, metric_values, marker='o', linestyle='-')
    ax.set_xlabel('Percentage of Data')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Evolution of {metric_name} Across Data Percentages')
    ax.grid(True)
    
    max_idx = np.argmax(metric_values)
    ax.annotate(f'Max: {metric_values[max_idx]:.4f}',
                xy=(data_points[max_idx], metric_values[max_idx]),
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    min_idx = np.argmin(metric_values)
    ax.annotate(f'Min: {metric_values[min_idx]:.4f}',
                xy=(data_points[min_idx], metric_values[min_idx]),
                xytext=(10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    
    return fig

def _infer_discount_from_path(path: str, default_other: float = 0.98) -> float:
    """Return the discount factor based on the game-matrix identifier in *path*.

    • game_matrix_1 and game_matrix_1a  → 0.9
    • anything else                     → *default_other*
    """
    lowered = path.lower()
    if "game_matrix_1a" in lowered or "game_matrix_1" in lowered:
        return 0.9
    return default_other

def generate_walk_agent_data(num_games=100000, random_seed=42):
    """
    Generate synthetic walk agent data following the same statistical distributions
    as the existing game data. Walk agent always walks in round 1.
    Args:
        num_games: Number of synthetic games to generate
        random_seed: Random seed for reproducibility
    Returns:
        List of game result dictionaries compatible with analyze_single_game output
    """
    np.random.seed(random_seed)
    walk_results = []
    
    for _ in range(num_games):
        #items = np.random.poisson(4, 5)
        items = [7, 4, 1]
        
        p1_values = np.random.randint(1, 101, 3).tolist()
        p2_values = np.random.randint(1, 101, 3).tolist()
        
        total_value_p1 = int(np.ceil(np.dot(items, p1_values)))
        total_value_p2 = int(np.ceil(np.dot(items, p2_values)))
        
        p1_outside_offer = np.random.randint(1, total_value_p1)
        p2_outside_offer = np.random.randint(1, total_value_p2)
        
        walk_results.append({
            'agent1': 'walk_agent_circle_0',
            'agent2': 'walk_agent_circle_0',  
            'final_action': 'WALK',
            'final_actor_role': 'agent1',
            'final_round': 1,
            'discount_factor': 1.0,  # No discount for round 1
            'agent1_value': float(p1_outside_offer),
            'agent2_value': float(p2_outside_offer),
            'p1_values': p1_values,
            'p2_values': p2_values,
            'p1_items': None,  # No items received in walk
            'p2_items': None,
            'full_items': items,
            'num_items': 3, #5
            'p1_outside_offer': p1_outside_offer,
            'p2_outside_offer': p2_outside_offer,
            'is_walk_optimal': None,  # Could be computed but not essential
            'is_agree_optimal': None,
            'utilitarian_welfare': float(p1_outside_offer + p2_outside_offer),
            'nash_welfare': math.sqrt(float(p1_outside_offer) * float(p2_outside_offer)),
            'nash_welfare_adv': 0,
            'is_ef1': None  # Not applicable for walk
        })
    
    return walk_results

def process_all_games(crossplay_dir="crossplay/game_matrix_2", discount_factor=DEFAULT_DISCOUNT_FACTOR, 
                      batch_size=100, num_bootstraps=1000, track_br_evolution=False, random_seed=42, use_cell_based_bootstrap=False, full_game_mix=True):
    """
    Process all game data files in the specified directory and return compiled results,
    separated by the round the game ended.
    
    Args:
        crossplay_dir: Directory containing game data files
        discount_factor: Discount factor to apply to utilities (default: DEFAULT_DISCOUNT_FACTOR)
        batch_size: Number of games in first batch for BR evolution
        num_bootstraps: Number of bootstrap samples for BR evolution
        track_br_evolution: Whether to track the evolution of best response graphs
        random_seed: Fixed random seed for reproducibility
        use_cell_based_bootstrap: Whether to use cell-based bootstrapping for BR evolution
        full_game_mix: boolean flag whether mix over full game matrix
        
    Returns:
        tuple: (
            all_results,
            agent_performance_by_round,  # Dict: {round: {agent1: {agent2: mean_value}}}
            agent_final_rounds_by_round, # Dict: {round: {agent: mean_final_round}}
            agent_game_counts_by_round,  # Dict: {round: {agent1: {agent2: count}}}
            agent_final_rounds_self_play_by_round, # Dict: {round: {agent: mean_final_round}}
            br_evolution_data             # Optional, if track_br_evolution is True
        )
    """
    
    print("Reading all game data files...")
    all_game_results = []
    files_processed = 0
    if not full_game_mix:
        
        for root, dirs, files in os.walk(crossplay_dir):
            for file in files:
                if file.endswith('.json'):
                    
                    file_path = os.path.join(root, file)
                    try:
                        # if "game_matrix_1" in crossplay_dir:
                        #     discount_factor = .9
                        # elif "game_matrix_1a" in crossplay_dir:
                        #     discount_factor = .9
                        # else:
                        #     discount_factor = .98

                        discount_factor = _infer_discount_from_path(file_path)

                        print(f"Discount factor is {discount_factor}")
                        game_results = analyze_single_game(file_path, discount_factor) 
                        
                        filtered_results = []  
                        for result in game_results:
                            agent1 = result.get('agent1')
                            agent2 = result.get('agent2')

                            if (agent1 is not None and "gemini_2.0_flash_circle_6" in agent1) or \
                            (agent2 is not None and "gemini_2.0_flash_circle_6" in agent2):
                                continue 
                            # if (agent1 is not None and "gemini_2.0_flash_circle_2" in agent1) or \
                            # (agent2 is not None and "gemini_2.0_flash_circle_2" in agent2):
                            #     continue 
                            # if (agent1 is not None and "gemini_2.0_flash_circle_5" in agent1) or \
                            # (agent2 is not None and "gemini_2.0_flash_circle_5" in agent2):
                            #     continue

                            # if (agent1 is not None and "openai_4o_circle_4" in agent1) or \
                            # (agent2 is not None and "openai_4o_circle_4" in agent2):
                            #     continue
                            # if (agent1 is not None and "openai_4o_circle_5" in agent1) or \
                            # (agent2 is not None and "openai_4o_circle_5" in agent2):
                            #     continue
                            # if (agent1 is not None and "openai_4o_circle_6" in agent1) or \
                            # (agent2 is not None and "openai_4o_circle_6" in agent2):
                            #     continue
                            # if (agent1 is not None and "openai_4o_2024-08-06_circle_4" in agent1) or \
                            # (agent2 is not None and "openai_4o_2024-08-06_circle_4" in agent2):
                            #     continue
                            # if (agent1 is not None and "openai_4o_2024-08-06_circle_5" in agent1) or \
                            # (agent2 is not None and "openai_4o_2024-08-06_circle_5" in agent2):
                            #     continue
                            # if (agent1 is not None and "openai_4o_2024-08-06_circle_6" in agent1) or \
                            # (agent2 is not None and "openai_4o_2024-08-06_circle_6" in agent2):
                            #     continue
                             
                            # if (agent1 is not None and "anthropic_3.7_sonnet_circle_5" in agent1) or \
                            # (agent2 is not None and "anthropic_3.7_sonnet_circle_5" in agent2):
                            #     continue
                            # if (agent1 is not None and "anthropic_3.7_sonnet_circle_6" in agent1) or \
                            # (agent2 is not None and "anthropic_3.7_sonnet_circle_6" in agent2):
                            #     continue
                            # if (agent1 is not None and "anthropic_sonnet_3.7_reasoning_circle_0" in agent1) or \
                            # (agent2 is not None and "anthropic_sonnet_3.7_reasoning_circle_0" in agent2):
                            #     continue
                            # if (agent1 is not None and "anthropic_3.7_sonnet_2025-02-19_circle_5" in agent1) or \
                            # (agent2 is not None and "anthropic_3.7_sonnet_2025-02-19_circle_5" in agent2):
                            #     continue
                            # if (agent1 is not None and "anthropic_3.7_sonnet_2025-02-19_circle_6" in agent1) or \
                            # (agent2 is not None and "anthropic_3.7_sonnet_2025-02-19_circle_6" in agent2):
                            #     continue
                            # if (agent1 is not None and "anthropic_sonnet_3.7_reasoning_2025-02-19_circle_0" in agent1) or \
                            # (agent2 is not None and "anthropic_sonnet_3.7_reasoning_2025-02-19_circle_0" in agent2):
                            #     continue    
        
                            # if (agent1 is not None and "tough_agent_circle_0" in agent1) or \
                            # (agent2 is not None and "tough_agent_circle_0" in agent2):
                            #    continue 
                            # if (agent1 is not None and "soft_agent_circle_0" in agent1) or \
                            # (agent2 is not None and "soft_agent_circle_0" in agent2):
                            #    continue 
                            # if (agent1 is not None and "rnad_agent_circle_0" in agent1) or \
                            # (agent2 is not None and "rnad_agent_circle_0" in agent2):
                             #    continue
                            # if (agent1 is not None and "nfsp_agent_circle_0" in agent1) or \
                            # (agent2 is not None and "nfsp_agent_circle_0" in agent2):
                            #    continue
                            if (agent1 is not None and "tit_for_tat_agent_circle_0" in agent1) or \
                            (agent2 is not None and "tit_for_tat_agent_circle_0" in agent2):
                               continue
                            if (agent1 is not None and "nfsp_agent_both_circ_bg4_circle_0" in agent1) or \
                            (agent2 is not None and "nfsp_agent_both_circ_bg4_circle_0" in agent2):
                                 continue
                                
                             
                            

                            if (agent1 is not None and "nfsp_agent_2_circle_0" in agent1) or \
                            (agent2 is not None and "nfsp_agent_2_circle_0" in agent2):
                               continue


                            if agent1 is not None:
                                result['agent1'] = get_canonical_name(agent1)
                            
                            if agent2 is not None:
                                result['agent2'] = get_canonical_name(agent2)

                            if "game_matrix_1a" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_1a'
                            elif "game_matrix_2a" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_2a'
                            elif "game_matrix_3a" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_3a'
                            elif "game_matrix_1" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_1'
                            elif "game_matrix_2" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_2'
                            elif "game_matrix_3" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_3'

                            
                            filtered_results.append(result)
                        
                        all_game_results.extend(filtered_results) 
                        files_processed += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        print(f"Finished reading {files_processed} files.")
    else:
        print("AGGREGATING ALL GAME MATRICES INTO ONE MATRIX!")
        directories = [
            #"/Users/gabesmithline/Desktop/caif_negotiation/crossplay/game_matrix_1a",
            #"/Users/gabesmithline/Desktop/caif_negotiation/crossplay/game_matrix_2a",
            #"/Users/gabesmithline/Desktop/caif_negotiation/crossplay/game_matrix_3a",
            "/Users/gabesmithline/Desktop/caif_negotiation/crossplay/game_matrix_3",
            "/Users/gabesmithline/Desktop/caif_negotiation/crossplay/game_matrix_2",
            "/Users/gabesmithline/Desktop/caif_negotiation/crossplay/game_matrix_1",
        ]
        print(f"{directories}")
        for crossplay_dir in directories:
            print("Processesing directory")
            for root, dirs, files in os.walk(crossplay_dir):
             
                for file in files:
                    
                    if file.endswith('.json'):
                        
                        
                        file_path = os.path.join(root, file)
                        try:
                            if "game_matrix_1" in crossplay_dir:
                                discount_factor = .9
                            elif "game_matrix_1a" in crossplay_dir:
                                discount_factor = .9
                            else:
                                discount_factor = .98
                            game_results = analyze_single_game(file_path, discount_factor) 
                            
                            filtered_results = []
                            for result in game_results:
                                agent1 = result.get('agent1')
                                agent2 = result.get('agent2')

                                agent1 = get_canonical_name(agent1)
                                agent2 = get_canonical_name(agent2)


                                if (agent1 is not None and "gemini_2.0_flash_circle_6" in agent1) or \
                                (agent2 is not None and "gemini_2.0_flash_circle_6" in agent2):
                                    continue 
                                if (agent1 is not None and "gemini_2.0_flash_circle_6" in agent1) or \
                                (agent2 is not None and "gemini_2.0_flash_circle_6" in agent2):
                                    continue 
                                # if (agent1 is not None and "gemini_2.0_flash_circle_2" in agent1) or \
                                # (agent2 is not None and "gemini_2.0_flash_circle_2" in agent2):
                                #     continue 
                                # if (agent1 is not None and "gemini_2.0_flash_circle_5" in agent1) or \
                                # (agent2 is not None and "gemini_2.0_flash_circle_5" in agent2):
                                #     continue
                                # if (agent1 is not None and "openai_4o_circle_4" in agent1) or \
                                # (agent2 is not None and "openai_4o_circle_4" in agent2):
                                #     continue
                                # if (agent1 is not None and "openai_4o_circle_5" in agent1) or \
                                # (agent2 is not None and "openai_4o_circle_5" in agent2):
                                #     continue
                                # if (agent1 is not None and "openai_4o_circle_6" in agent1) or \
                                # (agent2 is not None and "openai_4o_circle_6" in agent2):
                                #     continue
                                # if (agent1 is not None and "openai_4o_2024-08-06_circle_4" in agent1) or \
                                # (agent2 is not None and "openai_4o_2024-08-06_circle_4" in agent2):
                                #     continue
                                # if (agent1 is not None and "openai_4o_2024-08-06_circle_5" in agent1) or \
                                # (agent2 is not None and "openai_4o_2024-08-06_circle_5" in agent2):
                                #     continue
                                # if (agent1 is not None and "openai_4o_2024-08-06_circle_6" in agent1) or \
                                # (agent2 is not None and "openai_4o_2024-08-06_circle_6" in agent2):
                                #     continue
                                # if (agent1 is not None and "anthropic_3.7_sonnet_circle_5" in agent1) or \
                                # (agent2 is not None and "anthropic_3.7_sonnet_circle_5" in agent2):
                                #     continue
                                # if (agent1 is not None and "anthropic_3.7_sonnet_circle_6" in agent1) or \
                                # (agent2 is not None and "anthropic_3.7_sonnet_circle_6" in agent2):
                                #     continue
                                # if (agent1 is not None and "anthropic_sonnet_3.7_reasoning_circle_0" in agent1) or \
                                # (agent2 is not None and "anthropic_sonnet_3.7_reasoning_circle_0" in agent2):
                                #     continue
                                # if (agent1 is not None and "anthropic_3.7_sonnet_2025-02-19_circle_5" in agent1) or \
                                # (agent2 is not None and "anthropic_3.7_sonnet_2025-02-19_circle_5" in agent2):
                                #     continue
                                # if (agent1 is not None and "anthropic_3.7_sonnet_2025-02-19_circle_6" in agent1) or \
                                # (agent2 is not None and "anthropic_3.7_sonnet_2025-02-19_circle_6" in agent2):
                                #     continue
                                # if (agent1 is not None and "anthropic_sonnet_3.7_reasoning_2025-02-19_circle_0" in agent1) or \
                                # (agent2 is not None and "anthropic_sonnet_3.7_reasoning_2025-02-19_circle_0" in agent2):
                                #     continue    

                                # if (agent1 is not None and "conceder_agent_circle_0" in agent1) or \
                                # (agent2 is not None and "conceder_agent_circle_0" in agent2):
                                #    continue
                                # if (agent1 is not None and "rnad_agent_circle_0" in agent1) or \
                                # (agent2 is not None and "rnad_agent_circle_0" in agent2):
                                #    continue
                                # if (agent1 is not None and "tough_agent_circle_0" in agent1) or \
                                # (agent2 is not None and "tough_agent_circle_0" in agent2):
                                    # continue 
                                # if (agent1 is not None and "soft_agent_circle_0" in agent1) or \
                                # (agent2 is not None and "soft_agent_circle_0" in agent2):
                                    # continue 

                                if (agent1 is not None and "nfsp_agent_circle_0" in agent1) or \
                                (agent2 is not None and "nfsp_agent_circle_0" in agent2):
                                    continue 
                                if (agent1 is not None and "nfsp_agent_both_circ_bg4_circle_0" in agent1) or \
                                (agent2 is not None and "nfsp_agent_both_circ_bg4_circle_0" in agent2):
                                    continue

                                if (agent1 is not None and "nfsp_agent_2_circle_0" in agent1) or \
                                (agent2 is not None and "nfsp_agent_2_circle_0" in agent2):
                                    continue 

                                # if (agent1 is not None and "tit_for_tat_agent_circle_0" in agent1) or \
                                # (agent2 is not None and "tit_for_tat_agent_circle_0" in agent2):
                                #     continue 
                                
                                
                                if agent1 is not None:
                                    result['agent1'] = get_canonical_name(agent1)
                                
                                if agent2 is not None:
                                    result['agent2'] = get_canonical_name(agent2)
                                
                                # Tag each game with its source matrix
                                if "game_matrix_1a" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_1a'
                                elif "game_matrix_2a" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_2a'
                                elif "game_matrix_3a" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_3a'
                                elif "game_matrix_1" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_1'
                                elif "game_matrix_2" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_2'
                                elif "game_matrix_3" in crossplay_dir:
                                    result['source_matrix'] = 'game_matrix_3'

                                filtered_results.append(result)
                            
                            all_game_results.extend(filtered_results) 
                            files_processed += 1
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                    print(f"Finished reading {files_processed} files.")

    group_by_company = False  # Preserve individual model names; company mapping handled later by meta-game helper
    all_agents = set()
    for game in all_game_results:
        if game['agent1'] is not None:
            all_agents.add(game['agent1'])
        if game['agent2'] is not None:
            all_agents.add(game['agent2']) #NOTE Seperate this module!
    all_known_agents = sorted(list(all_agents))
    print(f"Identified {len(all_known_agents)} unique agents.")
    if group_by_company:
        # Replace agent-level sets with company names
        try:
            from compute_subgame_equilibria import infer_company
        except ImportError:
            # Fallback definition (keep in sync with compute_subgame_equilibria)
            def infer_company(agent_name: str) -> str:
                lower = agent_name.lower()
                if any(lower.startswith(p) or p in lower for p in ("gem", "gemini")):
                    return "Gemini"
                if "sonnet" in lower:
                    return "Anthropic"
                if any(lower.startswith(p) or p in lower for p in ("4o", "o3", "gpt", "openai")):
                    return "OpenAI"
                return "Unknown"

        company_names = {infer_company(a) for a in all_agents if a is not None}
        # Optionally drop unknowns
        if "Unknown" in company_names and len(company_names) > 1:
            company_names.remove("Unknown")

        all_agents = company_names
        all_known_agents = sorted(list(company_names))
        print(f"Grouped into {len(all_known_agents)} companies: {all_known_agents}")
    

    
    print("Generating synthetic walk agent data...")
    print(all_known_agents)
    base_walk_data = generate_walk_agent_data(num_games=150000, random_seed=random_seed)
    # 
    walk_matchup_data = []
    # 
    if full_game_mix:
        for source_matrix in ['game_matrix_1a', 'game_matrix_2a', 'game_matrix_3a']:
            for base_game in base_walk_data:
                walk_self_play = base_game.copy()
                walk_self_play['source_matrix'] = source_matrix
                walk_matchup_data.append(walk_self_play)
    else:
        walk_matchup_data.extend(base_walk_data)
    # 
    for other_agent in all_known_agents:
        if other_agent != 'walk_agent_circle_0':  
            if full_game_mix:
                for source_matrix in ['game_matrix_1a', 'game_matrix_2a', 'game_matrix_3a']:
                    for base_game in base_walk_data:
                        walk_vs_other = base_game.copy()
                        walk_vs_other['agent1'] = 'walk_agent_circle_0'
                        walk_vs_other['agent2'] = other_agent
                        walk_vs_other['source_matrix'] = source_matrix
                        walk_matchup_data.append(walk_vs_other)
                    # 
                    for base_game in base_walk_data:
                        other_vs_walk = base_game.copy()
                        other_vs_walk['agent1'] = other_agent
                        other_vs_walk['agent2'] = 'walk_agent_circle_0'
                        other_vs_walk['source_matrix'] = source_matrix
                        walk_matchup_data.append(other_vs_walk)
            else:
                #For single matrix, add walk data without source tagging
                for base_game in base_walk_data:
                    walk_vs_other = base_game.copy()
                    walk_vs_other['agent1'] = 'walk_agent_circle_0'
                    walk_vs_other['agent2'] = other_agent
                    walk_matchup_data.append(walk_vs_other)
                # 
                for base_game in base_walk_data:
                    other_vs_walk = base_game.copy()
                    other_vs_walk['agent1'] = other_agent
                    other_vs_walk['agent2'] = 'walk_agent_circle_0'
                    walk_matchup_data.append(other_vs_walk)
    # 
    all_game_results.extend(walk_matchup_data)
    print(f"Note: All walk matchups use identical underlying game data (opponent-independent)")
    # 
    all_agents.add('walk_agent_circle_0')
    all_known_agents = sorted(list(all_agents))
    print(f"Updated agent count: {len(all_known_agents)} agents including walk agent.")
    # 
    np.random.seed(random_seed)
    np.random.shuffle(all_game_results)

    all_results = all_game_results
   
    
    
    rounds = [1, 2, 3, 4, 5, 'aggregate'] #TODO, should just extract number of game rounds. 
    raw_agent_performance_by_round = {r: defaultdict(lambda: defaultdict(list)) for r in rounds}
    raw_agent_final_rounds_by_round = {r: defaultdict(list) for r in rounds}
    agent_game_counts_by_round = {r: defaultdict(lambda: defaultdict(int)) for r in rounds}
    raw_agent_final_rounds_self_play_by_round = {r: defaultdict(list) for r in rounds}
    #TODO 
    print("Aggregating results by round...")
    games_processed = 0
    for result in all_results:
        agent1 = result.get('agent1')
        agent2 = result.get('agent2')
        final_round_num = result.get('final_round')

        if agent1 is None or agent2 is None:
            continue

        if final_round_num not in [1, 2, 3, 4, 5]:
             round_key = None
        else:
            round_key = final_round_num

        if result.get('agent1_value') is not None:
            raw_agent_performance_by_round['aggregate'][agent1][agent2].append(result['agent1_value'])
            agent_game_counts_by_round['aggregate'][agent1][agent2] += 1
        if result.get('agent2_value') is not None:
            raw_agent_performance_by_round['aggregate'][agent2][agent1].append(result['agent2_value'])
            agent_game_counts_by_round['aggregate'][agent2][agent1] += 1

        if final_round_num is not None: 
             raw_agent_final_rounds_by_round['aggregate'][agent1].append(final_round_num)
             raw_agent_final_rounds_by_round['aggregate'][agent2].append(final_round_num)
             if agent1 == agent2:
                 raw_agent_final_rounds_self_play_by_round['aggregate'][agent1].append(final_round_num)


        if round_key is not None:
            if result.get('agent1_value') is not None:
                raw_agent_performance_by_round[round_key][agent1][agent2].append(result['agent1_value'])
                agent_game_counts_by_round[round_key][agent1][agent2] += 1
            if result.get('agent2_value') is not None:
                raw_agent_performance_by_round[round_key][agent2][agent1].append(result['agent2_value'])
                agent_game_counts_by_round[round_key][agent2][agent1] += 1

            raw_agent_final_rounds_by_round[round_key][agent1].append(final_round_num)
            raw_agent_final_rounds_by_round[round_key][agent2].append(final_round_num)
            if agent1 == agent2: # Simple self-play check
                 raw_agent_final_rounds_self_play_by_round[round_key][agent1].append(final_round_num)

        games_processed += 1
    print(f"Processed {games_processed} games for aggregation.")

    agent_performance_by_round = {r: defaultdict(lambda: defaultdict(lambda: np.nan)) for r in rounds}
    agent_final_rounds_by_round = {r: defaultdict(lambda: np.nan) for r in rounds}
    agent_final_rounds_self_play_by_round = {r: defaultdict(lambda: np.nan) for r in rounds}

    for r in rounds:
        for agent1, opponents in raw_agent_performance_by_round[r].items():
            for agent2, values in opponents.items():
                if values:
                    agent_performance_by_round[r][agent1][agent2] = np.mean(values)

        for agent, final_rounds_list in raw_agent_final_rounds_by_round[r].items():
            if final_rounds_list:
                agent_final_rounds_by_round[r][agent] = np.mean(final_rounds_list)

        for agent, final_rounds_list in raw_agent_final_rounds_self_play_by_round[r].items():
             if final_rounds_list:
                 agent_final_rounds_self_play_by_round[r][agent] = np.mean(final_rounds_list)

    print("Finished calculating mean values by round.")

    br_evolution_data = None
    if track_br_evolution:
        print("\nStarting Best Response evolution tracking...")
        br_evolution_data = {
            "batch_numbers": [], "frobenius_norms": [], "avg_br_matrices": [],
            "me_nash_strategies": [], "max_me_regrets": [], "avg_me_regrets": [],
            "me_strategy_distances": [], "random_seed": random_seed,
            "all_known_agents": all_known_agents
        }
        if use_cell_based_bootstrap:
            print(f"\nProcessing cumulative percentages for BR evolution with fixed random seed {random_seed}...")
            print(f"Using cell-based bootstrap with consistent set of {len(all_known_agents)} agents")

            previous_avg_br_matrix = None
            previous_me_strategy = None
            #percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            percentages = [1.0]
            print(f"Total games available for cell bootstrap: {len(all_game_results)}")

            for percent in percentages:
                print(f"\n>>> Processing data at {percent:.1%} cumulative percentage...")

                # Use appropriate bootstrap function based on full_game_mix flag
                
                if full_game_mix:
                    # Use bootstrap-level 1/3 weighting when combining multiple game matrices
                    avg_br_matrix, me_strategy, max_me_regret, avg_me_regret = nonparametric_bootstrap_best_response_with_matrix_weighting(
                        all_game_results,
                        all_known_agents,
                        percent_data=percent,
                        num_bootstrap=num_bootstraps,
                        random_seed=random_seed,
                        use_company_meta_game=False,
                        company_pooling_average=False 
                    )
                else:
                    # Use original function for single matrix analysis
                    avg_br_matrix, me_strategy, max_me_regret, avg_me_regret = nonparametric_bootstrap_best_response_by_cell(
                        all_game_results,
                        all_known_agents,
                        percent_data=percent,
                        num_bootstrap=num_bootstraps,
                        random_seed=random_seed
                    )
                
                if avg_br_matrix is None or me_strategy is None:
                     print(f"  >>> WARNING: Bootstrapping for {percent:.1%} data returned None. Skipping this percentage.")
                     continue 

                if previous_avg_br_matrix is not None and previous_me_strategy is not None:
                    current_agents = avg_br_matrix.index.tolist()
                    prev_agents = previous_avg_br_matrix.index.tolist()
                    common_agents = sorted(list(set(current_agents) & set(prev_agents)))

                    if not common_agents:
                        print(f"  >>> WARNING: No common agents between {percent:.1%} and previous percentage. Skipping comparison.")
                        frobenius_norm = np.nan
                        me_strategy_distance = np.nan
                    else:
                        aligned_br_matrix = avg_br_matrix.loc[common_agents, common_agents]
                        aligned_prev_br_matrix = previous_avg_br_matrix.loc[common_agents, common_agents]

                        current_indices = [all_known_agents.index(agent) for agent in common_agents]
                        prev_indices = [all_known_agents.index(agent) for agent in common_agents] # Should be the same if all_known_agents is consistent

                        aligned_me_strategy = me_strategy[current_indices]
                        aligned_prev_me_strategy = previous_me_strategy[prev_indices]

                        if np.sum(aligned_me_strategy) > 1e-6:
                            aligned_me_strategy = aligned_me_strategy / np.sum(aligned_me_strategy)
                        if np.sum(aligned_prev_me_strategy) > 1e-6:
                            aligned_prev_me_strategy = aligned_prev_me_strategy / np.sum(aligned_prev_me_strategy)

                        frobenius_norm = compute_frobenius_norm(aligned_br_matrix, aligned_prev_br_matrix)
                        me_strategy_distance = compute_euclidean_distance(aligned_me_strategy, aligned_prev_me_strategy)


                    br_evolution_data["batch_numbers"].append(percent)
                    br_evolution_data["frobenius_norms"].append(frobenius_norm) # Store potentially NaN norm
                    br_evolution_data["avg_br_matrices"].append(avg_br_matrix.copy())
                    #br_evolution_data["me_nash_strategies"].append(me_strategy.copy())
                    br_evolution_data["max_me_regrets"].append(max_me_regret)
                    br_evolution_data["avg_me_regrets"].append(avg_me_regret)
                    #br_evolution_data["me_strategy_distances"].append(me_strategy_distance) # Store potentially NaN distance

                    print(f"  >>> Cumulative Percentage {percent:.1%}:")
                    print(f"Frobenius norm = {frobenius_norm:.4f}")
                    print(f"Max ME NE regret = {max_me_regret:.6f}")
                    print(f"Avg ME NE regret = {avg_me_regret:.6f}")
                    print(f"ME strategy distance = {me_strategy_distance:.6f}")
                else: # First batch
                    br_evolution_data["batch_numbers"].append(percent)
                    br_evolution_data["frobenius_norms"].append(0.0)
                    br_evolution_data["avg_br_matrices"].append(avg_br_matrix.copy())
                    #br_evolution_data["me_nash_strategies"].append(me_strategy.copy())
                    br_evolution_data["max_me_regrets"].append(max_me_regret)
                    br_evolution_data["avg_me_regrets"].append(avg_me_regret)
                    br_evolution_data["me_strategy_distances"].append(0.0)
                    print(f"Cumulative Percentage {percent:.1%}: First percentage processed")
                    print(f"Max ME NE regret = {max_me_regret:.6f}")
                    print(f"Avg ME NE regret = {avg_me_regret:.6f}")

                # Update the previous matrix and strategy IF the current ones are valid
                #previous_avg_br_matrix = avg_br_matrix.copy()
                #previous_me_strategy = me_strategy.copy()

        else: # Game-based bootstrap logic (unchanged, still needs ME NE implementation)
            # Use the existing game-based implementation
            print(f"\nProcessing cumulative batches for BR evolution with fixed random seed {random_seed}...")
            print(f"Using consistent set of {len(all_known_agents)} agents for all batches")
            
            previous_avg_br_matrix = None
            previous_me_strategy = None # Initialize here too
            total_games = len(all_game_results)
            
            max_batches = total_games // batch_size
            
            if max_batches < 2:
                print(f"WARNING: Not enough games ({total_games}) for meaningful BR evolution with batch size {batch_size}")
                print(f"Need at least {2*batch_size} games for 2 batches. Will analyze all available games.")
                max_batches = 1 if total_games >= batch_size else 0
            
            print(f"Will analyze {max_batches} cumulative batches from {total_games} total games")
            print(f"All known agents: {all_known_agents}")
            
            for batch_num in range(1, max_batches + 1):
                # Calculate size of this cumulative batch
                cumulative_size = min(batch_num * batch_size, total_games)
                cumulative_batch = all_game_results[:cumulative_size]
                
                print(f"\nProcessing cumulative batch {batch_num} with {cumulative_size} games...")
                

                print("NOTE: ME NE metric calculation not yet implemented for game-based batch evolution.")
                
                from meta_game_analysis.bootstrap_nonparametric import nonparametric_bootstrap_best_response # Import needed
                avg_br_matrix = nonparametric_bootstrap_best_response(
                    cumulative_batch,
                    num_bootstrap=num_bootstraps,
                    all_known_agents=all_known_agents,  
                    random_seed=random_seed
                )
                
                if avg_br_matrix is None:
                    print(f"  >>> WARNING: Game-based bootstrapping for batch {batch_num} returned None. Skipping.")
                    continue
                    
                me_strategy = None # Placeholder
                max_me_regret = None # Placeholder
                avg_me_regret = None # Placeholder

                if previous_avg_br_matrix is not None:
                    frobenius_norm = compute_frobenius_norm(avg_br_matrix, previous_avg_br_matrix)
                    # me_strategy_distance = compute_euclidean_distance(me_strategy, previous_me_strategy) # Cannot compute yet
                    
                    br_evolution_data["batch_numbers"].append(batch_num) 
                    br_evolution_data["frobenius_norms"].append(frobenius_norm)
                    br_evolution_data["avg_br_matrices"].append(avg_br_matrix.copy())
                    # Cannot store ME metrics yet
                    
                    print(f"  >>> Cumulative Batch {batch_num} ({cumulative_size} games): Frobenius norm = {frobenius_norm:.4f}")
                else: # First batch
                    br_evolution_data["batch_numbers"].append(batch_num) # Use batch_num here
                    br_evolution_data["frobenius_norms"].append(0.0)
                    br_evolution_data["avg_br_matrices"].append(avg_br_matrix.copy())
                    # Cannot store ME metrics yet
                    print(f"  >>> Cumulative Batch {batch_num} ({cumulative_size} games): First batch processed")
                
                previous_avg_br_matrix = avg_br_matrix.copy()
                # previous_me_strategy = me_strategy.copy() 

    if track_br_evolution and br_evolution_data is not None and len(br_evolution_data.get("batch_numbers", [])) > 1:
        print("\n>>> Final check: Found enough data points for evolution plots.")

    if track_br_evolution:
        return (all_results, agent_performance_by_round, agent_final_rounds_by_round,
                agent_game_counts_by_round, agent_final_rounds_self_play_by_round, br_evolution_data)
    else:
        return (all_results, agent_performance_by_round, agent_final_rounds_by_round,
                agent_game_counts_by_round, agent_final_rounds_self_play_by_round) 

def nonparametric_bootstrap_best_response_by_cell(all_game_results, all_known_agents, percent_data, num_bootstrap=1000, random_seed=42, use_company_meta_game=False):
    """
    Perform non-parametric bootstrap on game data organized by cell, using a specified percentage of the data.
    This function takes the first X% of data points for each cell in the performance matrix.
    
    Args:
        all_game_results: List of all game results
        all_known_agents: Complete list of all agents
        percent_data: Percentage of data to use (e.g., 0.1 for 10%, 0.2 for 20%)
        num_bootstrap: Number of bootstrap samples to generate
        random_seed: Fixed random seed for reproducibility
        
    Returns:
        tuple: (avg_br_matrix, me_nash_strategy, max_me_regret, avg_me_regret)
    """
    np.random.seed(random_seed)
    
    agent_pairs_data = defaultdict(list)
    
    for game in all_game_results:
        agent1 = game.get('agent1')
        agent2 = game.get('agent2')
        
        # Skip games with missing agents
        if not agent1 or not agent2:
            continue
            
        # Ensure string type for both agents
        agent1 = str(agent1)
        agent2 = str(agent2)
        
        # Only include known agents
        if agent1 not in all_known_agents or agent2 not in all_known_agents:
            continue
        
        # Skip games with missing values
        if 'agent1_value' not in game or 'agent2_value' not in game:
            continue
        if game['agent1_value'] is None or game['agent2_value'] is None:
            continue
            
        # Store both agent performance values together (from the same game)
        # This ensures cell[i,j] and cell[j,i] are kept together from the same games
        agent_pairs_data[(agent1, agent2)].append((game['agent1_value'], game['agent2_value']))
    
    # For each agent pair, take only the specified percentage of data
    cumulative_cell_data = {}
    total_games = 0
    
    for agent_pair, values in agent_pairs_data.items():
        agent1, agent2 = agent_pair
        
        # Calculate how many values to take (cumulative percentage)
        n_values = max(1, int(len(values) * percent_data))
        
        # Take the first n_values entries
        selected_values = values[:n_values]
        total_games += len(selected_values)
        
        # Create entries for both cells from these selected games
        for i in range(len(selected_values)):
            # For cell [agent1, agent2]
            if (agent1, agent2) not in cumulative_cell_data:
                cumulative_cell_data[(agent1, agent2)] = []
            cumulative_cell_data[(agent1, agent2)].append(selected_values[i][0])  # agent1's value
            
            # For cell [agent2, agent1]
            if (agent2, agent1) not in cumulative_cell_data:
                cumulative_cell_data[(agent2, agent1)] = []
            cumulative_cell_data[(agent2, agent1)].append(selected_values[i][1])  # agent2's value
    
    # print(f"Using {percent_data:.1%} of data ({total_games} total games from {len(agent_pairs_data)} agent pairs)")
    
    # Initialize arrays to accumulate results
    accumulated_br_matrix = None
    
    for b in range(num_bootstrap):
        performance_matrix = pd.DataFrame(index=all_known_agents, columns=all_known_agents)
        
        for (agent1, agent2), values in cumulative_cell_data.items():
            if not values:
                continue
                
            if len(values) > 0:
                bootstrap_indices = np.random.choice(range(len(values)), size=len(values), replace=True)
                resampled_values = [values[i] for i in bootstrap_indices]
                performance_matrix.loc[agent1, agent2] = np.mean(resampled_values)
            else:
                performance_matrix.loc[agent1, agent2] = np.nan
        
        game_matrix_np = performance_matrix.to_numpy().astype(float)
        
        for i in range(game_matrix_np.shape[0]):
            for j in range(game_matrix_np.shape[1]):
                if np.isnan(game_matrix_np[i, j]):
                    # Check if column has any non-NaN values
                    col_slice = game_matrix_np[:, j].astype(float)
                    if np.any(~np.isnan(col_slice)):
                        # Try column mean first
                        col_mean = np.nanmean(col_slice)
                        game_matrix_np[i, j] = col_mean
                    else:
                        # Check if row has any non-NaN values
                        row_slice = game_matrix_np[i, :].astype(float)
                        if np.any(~np.isnan(row_slice)):
                            row_mean = np.nanmean(row_slice)
                            game_matrix_np[i, j] = row_mean
                        else:
                            # If both column and row are all NaN, set to 0
                            game_matrix_np[i, j] = 0
        
        # Compute best response matrix for this bootstrap sample
        perf_df_for_br = pd.DataFrame(game_matrix_np, index=all_known_agents, columns=all_known_agents)
        if use_company_meta_game:
            perf_df_for_br = _agent_perf_to_company_perf(perf_df_for_br)
        
        br_matrix = compute_best_response_matrix(perf_df_for_br)
        
        # Accumulate the best response matrix
        if accumulated_br_matrix is None:
            accumulated_br_matrix = br_matrix
            print(f"BR Matrix for sample {b} {br_matrix}")

        else:
            accumulated_br_matrix += br_matrix
            #print(f"BR Matrix for sample {b} {br_matrix}")

        
        # Print progress for long bootstraps
        if b % 100 == 0 and b > 0:
            print(f"  - Completed {b}/{num_bootstrap} bootstrap iterations...")
    
    # Calculate averages from accumulated values
    avg_br_matrix = accumulated_br_matrix / num_bootstrap
    
    # Print a bit of info about the result
    print(f"Average BR matrix shape: {avg_br_matrix.shape}, sum: {np.sum(avg_br_matrix)}")
    
    # After building cumulative_cell_data and before printing stats, compute expected number of cells
    expected_pairs = len(all_known_agents) * len(all_known_agents)
    actual_pairs = len(cumulative_cell_data)
    
    # Provide a clearer diagnostic message
    print(f"Using {percent_data:.1%} of data across {actual_pairs} unique agent pair cells (expected {expected_pairs})")
    if actual_pairs < expected_pairs:
        missing = expected_pairs - actual_pairs
        print(f"WARNING: {missing} cells are missing data for the current bootstrap sample.")
    print(avg_br_matrix)
    return avg_br_matrix, 0, 0, 0  # Placeholders for ME Nash metrics 

# --- NE solver with fallback -------------------------------------------

def _solve_bimatrix_ne(R: np.ndarray, C: np.ndarray, max_jitter_attempts: int = 2):
    """Return one Nash equilibrium (sigma, tau) for the bimatrix game (R, C).

    Strategy:
    1.   Try nashpy.support_enumeration (fast, exact for small games).
    2.   If it yields no equilibrium (common in degenerate games), try the
         Lemke–Howson algorithm.
    3.   If that still fails, add a tiny iid noise (jitter) to break exact
         ties and go back to step 1 (do this *max_jitter_attempts* times).
    4.   As a last resort, return uniform mixed strategies of appropriate
         dimension.
    """
    def _uniform_mixed(k, l):
        return np.ones(k) / k, np.ones(l) / l

    # k, l = R.shape
    # # 1) Support enumeration ------------------------------------------------
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=RuntimeWarning, module="nashpy")
    #     try:
    #         g = nash.Game(R, C)
    #         sigma, tau = next(g.support_enumeration())
    #         return np.asarray(sigma, dtype=float), np.asarray(tau, dtype=float)
    #     except StopIteration:
    #         # Support enumeration found no equilibrium (degenerate game). Fall through to other solvers.
    #         pass
    #     except Exception:
    #         # Numerical issues or nashpy error; try other methods.
    #         pass
    
    # # # 2) Lemke–Howson -------------------------------------------------------
    try:
        g = nash.Game(R, C)
        sigma, tau = g.lemke_howson(initial_dropped_label=0)
        return np.asarray(sigma, dtype=float), np.asarray(tau, dtype=float)
    except Exception:
        pass

    # 3) Replicator-dynamics with multiple random restarts ------------------
    def _replicator_dynamics(R: np.ndarray, C: np.ndarray, restarts: int = 40,
                             steps: int = 6000, eta: float = 0.03, tol: float = 1e-7):
        """Run RD from many random seeds, return the attractor with largest basin."""
        k, l = R.shape
        rng = np.random.default_rng(0)
        attractors = {}
        hits = {}

        for _ in range(restarts):
            sigma = rng.dirichlet(np.ones(k))
            tau = rng.dirichlet(np.ones(l))

            prev_sigma, prev_tau = None, None
            for _ in range(steps):
                sigma_pay = R @ tau
                tau_pay = sigma @ C
                sigma_avg = sigma @ sigma_pay
                tau_avg = tau_pay @ tau

                sigma = sigma * np.exp(eta * (sigma_pay - sigma_avg))
                tau = tau * np.exp(eta * (tau_pay - tau_avg))

                sigma_sum = sigma.sum(); tau_sum = tau.sum()
                if sigma_sum == 0 or tau_sum == 0:
                    break
                sigma /= sigma_sum
                tau /= tau_sum

                if prev_sigma is not None and np.max(np.abs(sigma - prev_sigma)) < tol and \
                   np.max(np.abs(tau - prev_tau)) < tol:
                    break
                prev_sigma, prev_tau = sigma.copy(), tau.copy()

            key = tuple(np.round(np.concatenate([sigma, tau]), 5))
            attractors[key] = (sigma.copy(), tau.copy())
            hits[key] = hits.get(key, 0) + 1

        if hits:
            best_key = max(hits, key=hits.get)
            return attractors[best_key]
        return None

    # 2) Replicator dynamics -------------------------------------------------
    # rd_result = _replicator_dynamics(R, C)
    # if rd_result is not None:
    #     return rd_result

    # # 2.5) Max-entropy Nash via convex optimisation (general rectangular game)
    # try:
    #     k, l = R.shape
    #     sigma_var = cp.Variable(k)
    #     tau_var = cp.Variable(l)
    #     v = cp.Variable()  # row player's value
    #     w = cp.Variable()  # col player's value

    #     constraints = [
    #         sigma_var >= 0,
    #         tau_var >= 0,
    #         cp.sum(sigma_var) == 1,
    #         cp.sum(tau_var) == 1,
    #         R @ tau_var <= v,                # no profitable deviation rows
    #         sigma_var @ C <= w               # no profitable deviation columns
    #     ]

    #     entropy_obj = -cp.sum(cp.entr(sigma_var)) - cp.sum(cp.entr(tau_var))
    #     prob = cp.Problem(cp.Maximize(entropy_obj), constraints)
    #     prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-8, reltol=1e-8)

    #     if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    #         sigma = np.array(sigma_var.value).flatten()
    #         tau = np.array(tau_var.value).flatten()
    #         if sigma.sum() > 1e-8 and tau.sum() > 1e-8:
    #             sigma /= sigma.sum(); tau /= tau.sum()
    #             return sigma, tau
    # except Exception:
    #     # cvxpy not installed or solver failed; continue to other fallbacks
    #     pass
    
    # # 4) Final safety fallback: return uniform mixed strategies to avoid None.
    # return _uniform_mixed(R.shape[0], R.shape[1])

    

import numpy as np
from itertools import product
import sys

sys.path.append('../')

from game_runner import NegotitaionGame
from eval.game_evaluator import GameEvaluator
import agents.simple_agent as simple_agent
import agents.llm_agent as llm_agent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass, field
from math import prod, sqrt
sys.path.append('../caif_negotiation/')

import import_ipynb
from IPython import get_ipython
import runpy
import os

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('run', '../test_game_eval.ipynb')
else:
    py_file = os.path.join('..', 'test_game_eval.py')
    if os.path.isfile(py_file):
        runpy.run_path(py_file)
    else:
        print(
            "ERROR: Not in an IPython environment, and ../test_game_eval.ipynb "
            "hasn’t been converted to ../test_game_eval.py. Please convert it:\n"
            "  jupyter nbconvert --to python ../test_game_eval.ipynb\n"
            "…and then try again."
        )
from test_game_eval import *

import torch
from utils.offer import Offer

from prompts.make_prompt import make_prompt
from prompts.make_prompt_bargain import make_prompt_bargain
from metrics.visualizations import (
    plot_discounted_values,
    plot_offer_evolution,
    plot_negotiation_gap,
    plot_fairness
)
import concurrent.futures

pathology_results = pd.DataFrame()  
import itertools
envy_results_history = {}
from eval.metrics import *
import time
import pandas as pd
import torch
import numpy as np
from math import sqrt, prod


def calculate_discounted_value(offer, values, gamma, round_num):
    if offer is None:
        return 0
    base_value = sum(v * q for v, q in zip(values, offer))
    return base_value * (gamma ** (round_num - 1))

# ------------------------------------------------------------------------
# Helper function: Discounted value
# ------------------------------------------------------------------------
def calculate_discounted_value(offer, values, gamma, realization_round):
    """
    Returns the discounted value of `offer` for an agent with utility `values`,
    discount factor `gamma`, and the realization round number.
    """
    if offer is None:
        return 0
    base_value = sum(v * q for v, q in zip(values, offer))
    return base_value * (gamma ** (realization_round - 1))

# ------------------------------------------------------------------------
# Helper function: Detect pathology #4 (accepting worse than outside)
# ------------------------------------------------------------------------
def check_accepting_worse_than_outside(current_player, p1_kept, p2_kept, game):
    """
    PATHOLOGY 4: Accepting an offer worse than your outside offer.
    If the current player accepted, check if the portion they get
    is less than their outside offer.
    """
    accepting_worse = False
    action = game.players[current_player - 1].action
    if action == "ACCEPT":
        if current_player == 1 and p1_kept is not None:
            if np.dot(game.player_values[0], p1_kept) < game.outside_offer_values[0]:
                accepting_worse = True
        elif current_player == 2 and p2_kept is not None:
            if np.dot(game.player_values[1], p2_kept) < game.outside_offer_values[1]:
                accepting_worse = True
    return accepting_worse

# ------------------------------------------------------------------------
# Helper function: Detect pathology #5 (walking away from a better offer)
# ------------------------------------------------------------------------
def check_walking_away_from_better(current_player, p1_kept, p2_kept, game):
    """
    PATHOLOGY 5: Walking away from an offer better than your outside offer.
    """
    walking_away_better = False
    action = game.players[current_player - 1].action
    if ("WALK" in action) or (
        current_player == 2
        and action == "COUNTEROFFER"
        and game.current_round == game.max_rounds
    ):
        if current_player == 1 and p1_kept is not None:
            if np.dot(game.player_values[0], p1_kept) > game.outside_offer_values[0]:
                walking_away_better = True
        elif current_player == 2 and p2_kept is not None:
            if np.dot(game.player_values[1], p2_kept) > game.outside_offer_values[1]:
                walking_away_better = True
    return walking_away_better

# ------------------------------------------------------------------------
# Helper function: Determine validity of a WALK
# ------------------------------------------------------------------------
def determine_walk_away_type(current_player, game):
    """
    Checks if the current player's action is 'INVALID WALK' or 'WALK'
    and returns an appropriate walk_away_type. Otherwise returns None.
    """
    action = game.players[current_player - 1].action
    if "INVALID WALK" in action:
        return "INVALID"
    elif "WALK" in action:
        return "VALID"
    return None

# ------------------------------------------------------------------------
# Helper function: Update who-keeps-what (p1_kept, p2_kept)
# ------------------------------------------------------------------------
def update_kept_portions(current_player, game, p1_kept, p2_kept):
    """
    If there's a new COUNTEROFFER from the current player, update
    p1_kept and p2_kept accordingly.
    """
    action = game.players[current_player - 1].action
    if action == "COUNTEROFFER":
        if current_player == 1:
            # P1 is proposing, so P1's kept portion is whatever is left
            # and P2 is offered game.current_offer.offer
            p1_kept = game.items - np.array(game.current_offer.offer)
            p2_kept = np.array(game.current_offer.offer)
        else:  # current_player == 2
            # P2 is proposing, so P2's kept portion is whatever is left
            # and P1 is offered game.current_offer.offer
            p1_kept = np.array(game.current_offer.offer)
            p2_kept = game.items - np.array(game.current_offer.offer)
    return p1_kept, p2_kept

# ------------------------------------------------------------------------
# Helper function: Final round resolution
# ------------------------------------------------------------------------
def handle_final_round(
    game_num,
    current_round,
    current_player,
    game,
    prev_offer,
    p1_kept,
    p2_kept,
    p1_values,
    p2_values,
    p1_offers,
    accepting_an_offer_worse_than_outside_offer,
    pareto_front
):
    """
    Handle the final round's action, compute final metrics, and prepare data for recording.

    Args:
        game_num (int): The current game number.
        current_round (int): The current round number.
        current_player (int): The current player's number (1 or 2).
        game (NegotiationGame): The game instance.
        prev_offer (Offer): The previous offer made in the game.
        p1_kept (list): Player 1's kept allocation.
        p2_kept (list): Player 2's kept allocation.
        p1_values (list): List of Player 1's values across rounds.
        p2_values (list): List of Player 2's values across rounds.
        p1_offers (list): List of Player 1's offers.
        accepting_an_offer_worse_than_outside_offer (bool): Flag for pathology #4.
        pareto_front (list): The Pareto frontier for reference.

    Returns:
        dict: A dictionary containing all metrics to be recorded for the final step.
    """
    # Initialize metrics dictionary
    metrics = {
        "game_num": game_num,
        "step_num": 3.5,  # Assign a unique step number for the final round
        "round_num": current_round,
        "player": current_player,
        "action_played": None,
        "discount_rate": game.gamma ** (current_round - 1),
        "offer": list(game.current_offer.offer) if game.current_offer else [],
        "value": None,
        "undiscounted_value": None,
        "p1_valuations": list(game.player_values[0]),
        "p2_valuations": list(game.player_values[1]),
        "p1_kept_allocation": None,
        "p2_kept_allocation": None,
        "p1_final_value": None,
        "p2_final_value": None,
        "items": list(game.items),
        "your_side_of_current_offer": None,  # Adjust if applicable
        "outside_offer": None,  # Final round may not have outside_offer
        "outside_offer_undiscounted": None,  # Final round outside offers already considered
        "accepting_an_offer_worse_than_outside_offer": False,
        "making_an_offer_worse_for_you_than_your_outside_offer": False,  # Final round handled separately
        "walking_away_from_an_offer_better_than_your_outside_offer": False,  # Final round handled separately
        "offer_no_items_or_all_items": False,  # Final round handled separately
        "making_offer_worse_than_previous": False,  # Final round handled separately
        "nash_welfare": None,
        "proposal_proportion_player_1": None,
        "proposal_proportion_player_2": None,
        "concession_size": None,  # Final round no concessions
        "security_level_player_1": 0.0,
        "security_level_player_2": 0.0,
        "average_concession_size": None,  # To be computed post-game if needed
        "rawlsian_welfare": None,
        "gini_coefficient": None,
        "utilitarian_welfare": None,
        "jain_fairness_index": None,
        "on_pareto_frontier": False,
        "mean_absolute_difference": None,
        "walk_type": None
    }

    # Determine the final action and compute final values
    if game.current_offer and game.current_offer != prev_offer:
        # Player 2 made a final COUNTEROFFER
        print(f"\nPlayer {current_player}'s final action: COUNTEROFFER {game.current_offer.offer}")
        p1_value = game.outside_offer_values[0] * (game.gamma ** (current_round - 1))
        p2_value = game.outside_offer_values[1] * (game.gamma ** (current_round - 1))
        print("\nGame ended after max rounds - both players get outside offers")

        # Assign values
        metrics["action_played"] = "COUNTEROFFER"
        metrics["value"] = p1_value if current_player == 1 else p2_value
        metrics["undiscounted_value"] = (
            p1_value / (game.gamma ** (current_round - 1)) if current_player == 1 else
            p2_value / (game.gamma ** (current_round - 1))
        )
        metrics["p1_final_value"] = p1_value
        metrics["p2_final_value"] = p2_value

        # No allocations kept
        p1_kept = [0] * game.num_items
        p2_kept = [0] * game.num_items
        metrics["p1_kept_allocation"] = p1_kept
        metrics["p2_kept_allocation"] = p2_kept

    elif game.current_offer == prev_offer:
        # Player 2 ACCEPTED the final offer
        print("\nPlayer {current_player}'s final action: ACCEPT")
        # Player 2 accepted Player 1's final offer
        p1_kept = game.items - np.array(game.current_offer.offer)
        p1_value = calculate_discounted_value(
            p1_kept, game.player_values[0], game.gamma, current_round
        )
        p2_value = calculate_discounted_value(
            game.current_offer.offer, game.player_values[1], game.gamma, current_round
        )
        print(f"\nRound {current_round} Final Values:")
        print(f"Player 1: {p1_value:.2f}")
        print(f"Player 2: {p2_value:.2f}")

        # Assign values
        metrics["action_played"] = "ACCEPT"
        metrics["value"] = p1_value if current_player == 1 else p2_value
        metrics["undiscounted_value"] = (
            p1_value / (game.gamma ** (current_round - 1)) if current_player == 1 else
            p2_value / (game.gamma ** (current_round - 1))
        )
        metrics["p1_final_value"] = p1_value
        metrics["p2_final_value"] = p2_value

        # Assign allocations
        metrics["p1_kept_allocation"] = list(p1_kept)
        metrics["p2_kept_allocation"] = list(game.current_offer.offer)

        # Check pathology #4
        if game.outside_offer_values[1] > np.dot(game.player_values[1], game.current_offer.offer):
            accepting_an_offer_worse_than_outside_offer = True
            metrics["accepting_an_offer_worse_than_outside_offer"] = True

    else:
        # Player 2 WALKED AWAY
        print("\nPlayer {current_player}'s final action: WALK")
        p1_value = game.outside_offer_values[0] * (game.gamma ** (current_round - 1))
        p2_value = game.outside_offer_values[1] * (game.gamma ** (current_round - 1))
        print("\nGame ended after max rounds - both players get outside offers")

        # Assign values
        metrics["action_played"] = "WALK"
        metrics["value"] = None  # No specific value since walked away
        metrics["undiscounted_value"] = None
        metrics["p1_final_value"] = p1_value
        metrics["p2_final_value"] = p2_value

        # No allocations kept
        p1_kept = [0] * game.num_items
        p2_kept = [0] * game.num_items
        metrics["p1_kept_allocation"] = p1_kept
        metrics["p2_kept_allocation"] = p2_kept

    # Compute additional metrics only if action is ACCEPT or COUNTEROFFER
    if metrics["action_played"] in ("ACCEPT", "COUNTEROFFER"):
        # Compute Nash Welfare
        nash_welfare = sqrt(prod([
            p1_value,
            p2_value
        ]))
        metrics["nash_welfare"] = nash_welfare

        # Compute Utilitarian Welfare
        utilitarian_welfare = p1_value + p2_value
        metrics["utilitarian_welfare"] = utilitarian_welfare

        # Compute Rawlsian Welfare
        rawlsian_welfare = min(p1_value, p2_value)
        metrics["rawlsian_welfare"] = rawlsian_welfare

        # Compute Gini Coefficient
        if utilitarian_welfare > 0:
            gini_coefficient = abs(p1_value - p2_value) / (4.0 * utilitarian_welfare)
        else:
            gini_coefficient = 0.0
        metrics["gini_coefficient"] = gini_coefficient

        # Compute Mean Absolute Difference
        if p1_value == 0.0 and p2_value == 0.0:
            mean_absolute_difference = 0.0
        else:
            mean_absolute_difference = abs(p1_value - p2_value) / 2.0
        metrics["mean_absolute_difference"] = mean_absolute_difference

        # Compute Jain's Fairness Index
        if utilitarian_welfare > 0:
            mean_utility = utilitarian_welfare / 2.0
            variance = (p1_value**2 + p2_value**2) / 2.0 - mean_utility**2
            variance = max(variance, 0.0)  # Correct for negative variance due to precision
            coefficient_of_variation = (
                np.sqrt(variance) / mean_utility if mean_utility != 0 else 0.0
            )
            jain_fairness_index = 1 / (1 + coefficient_of_variation ** 2)
        else:
            jain_fairness_index = 0.0
        metrics["jain_fairness_index"] = jain_fairness_index

        # Compute Security Levels
        security_level_player_1 = max(0.0, game.outside_offer_values[0] - p1_value)
        security_level_player_2 = max(0.0, game.outside_offer_values[1] - p2_value)
        metrics["security_level_player_1"] = security_level_player_1
        metrics["security_level_player_2"] = security_level_player_2

        # Determine walk_type based on final action
        walk_type = None
        if metrics["action_played"] == "WALK":
            walk_type = "Player2_WALK"
        elif metrics["action_played"] == "COUNTEROFFER":
            walk_type = "Final_COUNTEROFFER"
        elif metrics["action_played"] == "ACCEPT":
            walk_type = "Final_ACCEPT"
        metrics["walk_type"] = walk_type

        # Check if on Pareto Frontier
        on_pareto_frontier = False
        for vals in pareto_front:
            if vals["type"] == "outside_offer" and game.current_offer is None:
                on_pareto_frontier = True
                break
            elif vals["type"] == "allocation":
                if (np.array_equal(vals["agent1"], p1_kept) and
                        np.array_equal(vals["agent2"], p2_kept)):
                    on_pareto_frontier = True
                    break
        metrics["on_pareto_frontier"] = on_pareto_frontier

    else:
        # For WALK action, compute welfare metrics based on outside offers
        nash_welfare = sqrt(prod([
            p1_value,
            p2_value
        ]))
        metrics["nash_welfare"] = nash_welfare

        utilitarian_welfare = p1_value + p2_value
        metrics["utilitarian_welfare"] = utilitarian_welfare

        rawlsian_welfare = min(p1_value, p2_value)
        metrics["rawlsian_welfare"] = rawlsian_welfare

        if utilitarian_welfare > 0:
            gini_coefficient = abs(p1_value - p2_value) / (4.0 * utilitarian_welfare)
        else:
            gini_coefficient = 0.0
        metrics["gini_coefficient"] = gini_coefficient

        if p1_value == 0.0 and p2_value == 0.0:
            mean_absolute_difference = 0.0
        else:
            mean_absolute_difference = abs(p1_value - p2_value) / 2.0
        metrics["mean_absolute_difference"] = mean_absolute_difference

        if utilitarian_welfare > 0:
            mean_utility = utilitarian_welfare / 2.0
            variance = (p1_value**2 + p2_value**2) / 2.0 - mean_utility**2
            variance = max(variance, 0.0)  # Correct for negative variance due to precision
            coefficient_of_variation = (
                np.sqrt(variance) / mean_utility if mean_utility != 0 else 0.0
            )
            jain_fairness_index = 1 / (1 + coefficient_of_variation ** 2)
        else:
            jain_fairness_index = 0.0
        metrics["jain_fairness_index"] = jain_fairness_index

        # Security levels already set to 0.0
        metrics["security_level_player_1"] = 0.0
        metrics["security_level_player_2"] = 0.0

        # Determine walk_type based on final action
        walk_type = None
        if metrics["action_played"] == "WALK":
            walk_type = "Player2_WALK"
        elif metrics["action_played"] == "COUNTEROFFER":
            walk_type = "Final_COUNTEROFFER"
        elif metrics["action_played"] == "ACCEPT":
            walk_type = "Final_ACCEPT"
        metrics["walk_type"] = walk_type

        # Check if on Pareto Frontier
        on_pareto_frontier = False
        for vals in pareto_front:
            if vals["type"] == "outside_offer" and game.current_offer is None:
                on_pareto_frontier = True
                break
            elif vals["type"] == "allocation":
                if (np.array_equal(vals["agent1"], p1_kept) and
                        np.array_equal(vals["agent2"], p2_kept)):
                    on_pareto_frontier = True
                    break
        metrics["on_pareto_frontier"] = on_pareto_frontier

    # Mark game as ended
    game.in_progress = False

    return metrics



def find_allocation_less_than_outside_offer_dp(items, player_valuations, outside_offer, player_num):
    """
    Finds the allocation that yields the highest utility strictly less than the outside_offer.
    Using dynamic programming to find the best allocation.
    """
    num_items = len(items)
    best_utility = -1.0
    best_combo = None

    quantity_ranges = [range(int(items[i]) + 1) for i in range(num_items)]
    
    for combo in product(*quantity_ranges):
        
        total_utility = 0.0
        for i in range(num_items):
            total_utility += player_valuations[i] * combo[i]

        if total_utility < outside_offer and total_utility > best_utility:
            best_utility = total_utility
            best_combo = combo

    if best_combo is None:
        return None
    allocation = {}
    for i in range(num_items):
        allocation[i] = best_combo[i]

    return allocation
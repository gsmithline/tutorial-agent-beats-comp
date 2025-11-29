
import numpy as np
# import utils # Not used currently
import random
import re # Needed for parsing walk action
from utils.offer import Offer
import ast
from prompts.make_prompt import make_prompt


class SoftNegotiator:
  def __init__(self, game, player_id, num_items=3, max_rounds=3): # Accept game object, player_id, num_items
    self.game = game
    self.player_id = player_id
    self.num_items = num_items
    self.num_actions = self.game.num_distinct_actions()
    self.max_quantity_for_encoding = 10 # Ensure this matches game param used by ToughNegotiator/LLM
    # Book-keeping attributes
    self.action = "SOFT_ACTION" # Default action type string
    self.total_quantities = None
    self.current_counter_offer_give_format = None
    self.items = None
    self.player_num = player_id 
    self.max_rounds = max_rounds
    self.valuation_vector = None
    self.walk_away_value = None
    self.prompt = None
  def process_observation(self, observation, num_items=3):
        """
        Process the OpenSpiel observation tensor and convert it to the format needed for prompts.
        
        Args:
            observation: The observation tensor from OpenSpiel
            num_items: Number of items in the negotiation (default: 5)
            
        Returns:
            dict: A dictionary containing all parameters needed for prompt generation
        """
        current_player_idx = self.player_id # Use self.player_id
        is_response_state = observation[3] == 0.0 if len(observation) > 3 else False # Corrected: Use 0.0 based on wrapper
        is_terminal = observation[4] == 1.0 if len(observation) > 4 else False
        agreement_reached = observation[5] == 1.0 if len(observation) > 5 else False
        
        round_number = int(observation[6]) + 1 if len(observation) > 6 else 1  # Convert from 0-indexed to 1-indexed
        discount_factor = observation[7] if len(observation) > 7 else 0.9
        current_discount = observation[8] if len(observation) > 8 else 1.0
        
        item_quantities = observation[9:9+num_items] if len(observation) > 9+num_items else [0] * num_items
        self.items = item_quantities
        
        player_values_start = 9+num_items
        player_values_end = player_values_start + num_items
        player_values = observation[player_values_start:player_values_end] if len(observation) > player_values_end else [0] * num_items
        self.valuation_vector = player_values
        walk_away_value_idx = 9+2*num_items
        walk_away_value = observation[walk_away_value_idx] if len(observation) > walk_away_value_idx else 0
        self.walk_away_value = walk_away_value
        history_start_idx = 9+2*num_items+1
        
        total_utility = np.dot(player_values, item_quantities)
        
        print(f"[DEBUG] Current player index: {current_player_idx}")
        print(f"[DEBUG] Round number: {round_number}")
        print(f"[DEBUG] Item quantities: {item_quantities}")
        self.total_quantities = item_quantities
        print(f"[DEBUG] Player values: {player_values}")

        print(f"[DEBUG] Walk away value: {walk_away_value}")

        print(f"[DEBUG] Total utility: {total_utility}")

        max_history_entries = (len(observation) - history_start_idx) // num_items
        
        max_rounds = max_history_entries // 2 if max_history_entries > 1 else 0
        
        history = {0: [], 1: []}
        current_offer = None
        
        print(f"[DEBUG] Raw history from observation tensor:")
        for i in range(max_history_entries):
            start_idx = history_start_idx + i * num_items
            if start_idx + num_items <= len(observation):
                raw_entry = observation[start_idx:start_idx + num_items]
                player = i % 2  
                print(f"  Entry {i} (Player {player+1}): {raw_entry}")
                
                if not all(x == -1 for x in raw_entry):
                    keep_for_self = [int(x) for x in raw_entry]
                    give_to_other = []
                    
                    for j in range(num_items):
                        quantity = int(item_quantities[j]) if j < len(item_quantities) else 0
                        give_amount = max(0, quantity - keep_for_self[j])
                        give_to_other.append(give_amount)
                    
                    offer_obj = Offer(player=player, offer=give_to_other)
                    history[player].append(offer_obj)
                    
                    if player != current_player_idx and is_response_state:
                        if (not current_offer) or (i > start_idx):
                            current_offer = offer_obj
                            print(f"  Setting as current offer for player {current_player_idx+1}")
        
        # Check for missing valid offers and apply corrections
        if not history[0] and not history[1] and round_number > 1:
            print("[WARNING] No valid offers found in history despite non-first round")
            # Use last_offer from agent's memory if available
            if self.last_offer:
                print(f"[DEBUG] Using last_offer from memory: {self.last_offer.offer}")
                history[self.player_num].append(self.last_offer)
            
        # Print reconstructed history for debugging
        print("[DEBUG] Reconstructed history (in 'give to other' format):")
        for p in range(2):
            player_offers = history[p]
            for i, offer in enumerate(player_offers):
                print(f"  Round {i+1}, Player {p+1}: {offer.offer if hasattr(offer, 'offer') else offer}")
        
        if is_response_state and not current_offer:
            other_player = 1 - current_player_idx
            other_player_offers = history.get(other_player, [])
            if other_player_offers:
                current_offer = other_player_offers[-1]
                print(f"[DEBUG] Using last offer from Player {other_player+1}: {current_offer.offer if hasattr(current_offer, 'offer') else current_offer}")
            else:
                
                if len(observation) > 20 and round_number > 1:
                    # Try common patterns for stored offers
                    possible_offer_indices = [
                        20, 
                        history_start_idx,
                        history_start_idx - num_items
                    ]
                    
                    for start_idx in possible_offer_indices:
                        if start_idx + num_items <= len(observation):
                            possible_offer = observation[start_idx:start_idx+num_items]
                            if not all(x == -1 for x in possible_offer):
                                # Convert from "keep for self" to "give to other" format
                                keep_for_self = [int(x) for x in possible_offer]
                                give_to_other = []
                                
                                # Calculate "give to other" by subtracting "keep for self" from total quantities
                                for j in range(num_items):
                                    quantity = int(item_quantities[j]) if j < len(item_quantities) else 0
                                    give_amount = max(0, quantity - keep_for_self[j])
                                    give_to_other.append(give_amount)
                                
                                print(f"[DEBUG] Found possible offer at index {start_idx}:")
                                print(f"  OpenSpiel keep format: {keep_for_self}")
                                print(f"  Converted give format: {give_to_other}")
                                
                                current_offer = Offer(player=other_player, offer=give_to_other)
                                break
                                
        if current_offer:
            print(f"[DEBUG] Current offer on table (give format): {current_offer.offer if hasattr(current_offer, 'offer') else current_offer}")
        else:
            print("[DEBUG] No current offer on table")
        
        # Prepare parameters for make_prompt
        params = {
            "T": num_items,
            "quantities": list(map(int, item_quantities)),
            "V": 101,  
            "values": list(map(int, player_values)),
            "W1": int(total_utility),
            "W2": 1,  
            "w": int(walk_away_value),
            "R": max_rounds,
            "g": discount_factor,
            "r": round_number,
            "history": history,
            "current_offer": current_offer,
            "player_num": current_player_idx,
            "p1_outside_offer": [1, int(total_utility) if current_player_idx == 0 else 0],  # Placeholder for p1 outside offer
            "p2_outside_offer": [1, int(total_utility) if current_player_idx == 1 else 0],  # Placeholder for p2 outside offer
            "circle": 0 
        }

        self.prompt = make_prompt(**params)
        
        

        
  # action_probabilities remains the same logic (uniform random over legal)
  def action_probabilities(self, state):
    """Calculates uniform probabilities over legal actions."""
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    if not legal_actions:
        print("[WARNING] SoftNegotiator found no legal actions.")
        return {0: 1.0} 

    accept_action = self._create_accept_action_id(state)
    if len(legal_actions) == 1 and legal_actions[0] == accept_action:
      return {accept_action: 1.0}

    return {a: 1/len(legal_actions) for a in legal_actions}

  def step(self, state):
    """
    Selects an action based on the soft strategy:
    1. If Accept is legal, choose Accept.
    2. Otherwise, choose randomly from legal actions.
    Updates self.action and self.current_counter_offer_give_format.
    """
    self.current_counter_offer_give_format = None
    self.process_observation(state.observation_tensor(self.player_id), self.num_items)

    cur_player = state.current_player()
    if cur_player != self.player_id:
        print(f"[ERROR] Soft P{self.player_id} asked to step for P{cur_player}. Returning action 0.")
        self.action = "ERROR_WRONG_PLAYER"
        return 0

    legal_actions = state.legal_actions()
    if not legal_actions:
        print(f"[ERROR] Soft P{self.player_id} has no legal actions. Returning action 0.")
        self.action = "ERROR_NO_LEGAL_ACTIONS"
        return 0

    accept_action_id = self._create_accept_action_id(state)

    chosen_action_id = None
    if accept_action_id is not None and accept_action_id in legal_actions:
        chosen_action_id = accept_action_id
        self.action = "ACCEPT"
        print(f"[INFO] Soft P{self.player_id} choosing ACCEPT action: {chosen_action_id}")
    else:
        chosen_action_id = random.choice(legal_actions)
        print(f"[INFO] Soft P{self.player_id} choosing randomly from legal actions: {chosen_action_id}")
        walk_action_id = self._create_walk_action_id(state)
        if chosen_action_id == walk_action_id:
            self.action = "WALK"
        else:
            self.action = "COUNTEROFFER"
            offer_to_self = self._create_counteroffer_action(action_id=chosen_action_id, state=state)
  
            if offer_to_self and isinstance(offer_to_self, str) and "Proposal:" in offer_to_self:
                try:
                    array_str = offer_to_self.split("Proposal:")[1].strip()
                    array_str = ast.literal_eval(array_str)
                    self.current_counter_offer_give_format = np.array(self.items) - np.array(array_str)
                    self.current_counter_offer_give_format = self.current_counter_offer_give_format.tolist()
                except Exception as e:
                    print(f"[ERROR] Failed to parse offer: {e}")
                    self.current_counter_offer_give_format = None
            else:
                self.current_counter_offer_give_format = None
    return chosen_action_id

  def _create_accept_action_id(self, state):
        """
        Create the action format for ACCEPT in OpenSpiel.
        In OpenSpiel negotiate game, typically ACCEPT is represented by action 0,
        but we'll look for the action with "Agreement" in its description.
        
        Args:
            state: The current game state
            
        Returns:
            int: The action ID for ACCEPT
        """
        if state is None:
            raise Exception("No State ACCEPT was given")
        self.action = "ACCEPT"
        for action in state.legal_actions():
            action_str = state.action_to_string(state.current_player(), action)
            if "Agreement" in action_str:
                print(f"[DEBUG] Found Agreement action: {action} with description: {action_str}")
                return action  
        return None

  def _create_walk_action_id(self, state):
        """
        Create the action format for WALK in OpenSpiel.
        [4.0, 4.0, 6.0, 6.0, 2.0]
        [0, 1, 1, 0, 0]
        Args:
            state: The current game state
            
        Returns:
            action: The formatted action for OpenSpiel
        """
        try:
            legal_actions = state.legal_actions()
            
            for a in legal_actions:
                action_str = state.action_to_string(state.current_player(), a)
                if "walk away" in action_str.lower():
                    print(f"[INFO] Walking away with action: {a}")
                    return a
                    
            if legal_actions:
                walk_action = max(legal_actions)
                print(f"[INFO] Walking away with highest action ID: {walk_action}")
                return walk_action
                
            print(f"[WARNING] No legal actions found when trying to walk away")
            return None
            
        except Exception as e:
            print(f"[ERROR] Exception in _create_walk_action: {e}")
            return None

  def _create_counteroffer_action(self, action_id, state):
        """
        Create the action format for COUNTEROFFER in OpenSpiel.
        
        This method directly encodes the offer vector to an action ID
        using OpenSpiel's encoding scheme, rather than trying to match
        from legal actions' string representations.
        
        Args:
            offer: List of item quantities to offer (what you give to other player)
            state: The current game state
            
        Returns:
            action: The formatted action for OpenSpiel
        """
        action = state.action_to_string(self.player_id, action_id)
        return action
            
       

# Removed the example usage block as it's not part of the class definition
# game_str = "bargaining(instances_file=./bargaining6796.txt)"
# game = pyspiel.load_game(game_str)
# print(utils.compute_game_scores(game, [[SoftNegotiator(game)], [
#       SoftNegotiator(game)]], weights=[[1.0], [1.0]], is_joint=False))
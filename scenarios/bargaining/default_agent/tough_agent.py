import numpy as np
import random
import re
from utils.offer import Offer
from prompts.make_prompt import make_prompt
# Remove Offer import if not used elsewhere
# from utils.offer import Offer


class ToughNegotiator:
 
  def __init__(self, game, valuation_vector, num_items, player_id, max_rounds=3):
    self.game = game
    self.valuation_vector = np.array(valuation_vector)

    if self.valuation_vector.shape != (num_items,):
      raise ValueError(f"Valuation vector shape mismatch. Expected ({num_items},), got {self.valuation_vector.shape}")
    self.num_items = num_items
    self.player_id = player_id
    self.player_num = player_id
    self.num_actions = self.game.num_distinct_actions()
    self.action = "TOUGH_ACTION" # Default action type string
    self.max_quantity_for_encoding = 10 # Ensure this matches game param
    self.walk_away_value = None
    self.total_quantities = None
    self.items = None
    self.current_counter_offer_give_format = None # Initialize storage
    self.max_rounds = max_rounds
    self.prompt = None
    # Removed action_to_offer_map logic

  def process_observation(self, observation, num_items=5):
        """
        Process the OpenSpiel observation tensor and convert it to the format needed for prompts.
        
        Args:
            observation: The observation tensor from OpenSpiel
            num_items: Number of items in the negotiation (default: 5)
            
        Returns:
            dict: A dictionary containing all parameters needed for prompt generation
        """
        current_player_idx = self.player_id # Use self.player_id directly
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
        
        walk_away_value_idx = 9+2*num_items
        walk_away_value = observation[walk_away_value_idx] if len(observation) > walk_away_value_idx else 0
        
        history_start_idx = 9+2*num_items+1
        
        total_utility = np.dot(player_values, item_quantities)
        
        # Debug prints
        print(f"[DEBUG] Current player index: {current_player_idx}")
        print(f"[DEBUG] Round number: {round_number}")
        print(f"[DEBUG] Item quantities: {item_quantities}")
        print(f"[DEBUG] Player values: {player_values}")
        self.valuation_vector = np.array(player_values)
        print(f"[DEBUG] Walk away value: {walk_away_value}")
        self.walk_away_value = walk_away_value
        self.total_quantities = np.array(item_quantities, dtype=int) # Ensure numpy array
        print(f"[DEBUG] Total utility: {total_utility}")

          # Extract proposal history directly from the observation tensor
        max_history_entries = (len(observation) - history_start_idx) // num_items
        
        # Calculate how many complete player pairs are in the history
        max_rounds = max_history_entries // 2 if max_history_entries > 1 else 0
        
        # Initialize history
        history = {0: [], 1: []}
        current_offer = None
        
        # Print raw history from observation tensor for debugging
        print(f"[DEBUG] Raw history from observation tensor:")
        for i in range(max_history_entries):
            start_idx = history_start_idx + i * num_items
            if start_idx + num_items <= len(observation):
                raw_entry = observation[start_idx:start_idx + num_items]
                player = i % 2  # Assuming alternating players in history
                print(f"  Entry {i} (Player {player+1}): {raw_entry}")
                
                # Check if this entry is not all -1 (indicating a valid offer)
                if not all(x == -1 for x in raw_entry):
                    keep_for_self = [int(x) for x in raw_entry]
                    give_to_other = []
                    
                    # Calculate "give to other" by subtracting "keep for self" from total quantities
                    for j in range(num_items):
                        quantity = int(item_quantities[j]) if j < len(item_quantities) else 0
                        give_amount = max(0, quantity - keep_for_self[j])
                        give_to_other.append(give_amount)
                    
    
                    
                    # Create offer object using "give to other" format since that's what our LLM expects
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
        
        # If in response state but we don't have a current offer, try to deduce it
        if is_response_state and not current_offer:
            other_player = 1 - current_player_idx
            # Check for any offers from other player
            other_player_offers = history.get(other_player, [])
            if other_player_offers:
                current_offer = other_player_offers[-1]
                print(f"[DEBUG] Using last offer from Player {other_player+1}: {current_offer.offer if hasattr(current_offer, 'offer') else current_offer}")
            else:
                
                if len(observation) > 20 and round_number > 1:
                    # Try common patterns for stored offers
                    possible_offer_indices = [
                        20, # Common location in OpenSpiel tensor
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
            "V": 101,  # Maximum value per item (assumed 100)
            "values": list(map(int, player_values)),
            "W1": int(total_utility),
            "W2": 1,  # Constant value as per original code
            "w": int(walk_away_value),
            "R": max_rounds,
            "g": discount_factor,
            "r": round_number,
            "history": history,
            "current_offer": current_offer,
            "player_num": current_player_idx,
            "p1_outside_offer": [1, int(total_utility) if current_player_idx == 0 else 0],  # Placeholder for p1 outside offer
            "p2_outside_offer": [1, int(total_utility) if current_player_idx == 1 else 0],  # Placeholder for p2 outside offer
            "circle": 0 # Use the stored example
        }

        self.prompt = make_prompt(**params)
        

  def _update_state_info(self, state):
    observation = state.observation_tensor(self.player_id)
    obs_len = len(observation)

    quantities_start_idx = 9
    quantities_end_idx = quantities_start_idx + self.num_items
    values_start_idx = quantities_end_idx
    values_end_idx = values_start_idx + self.num_items
    walk_value_idx = values_end_idx
    history_start_idx = walk_value_idx + 1

    if obs_len >= quantities_end_idx:
      self.total_quantities = np.array(observation[quantities_start_idx:quantities_end_idx], dtype=int)
    else:
      print(f"[ERROR] Tough P{self.player_id} could not parse quantities. Obs len: {obs_len}")
      self.total_quantities = np.zeros(self.num_items, dtype=int)

    if obs_len > walk_value_idx:
      self.walk_away_value = observation[walk_value_idx]
    else:
      print(f"[ERROR] Tough P{self.player_id} could not parse walk value. Obs len: {obs_len}")
      self.walk_away_value = 0

    self.current_opponent_offer_keep_format = None
    is_response_state = obs_len > 3 and observation[3] == 1.0
    if is_response_state:
      num_history_entries = (obs_len - history_start_idx) // self.num_items
      if num_history_entries > 0:
        last_entry_idx = history_start_idx + (num_history_entries - 1) * self.num_items
        if last_entry_idx + self.num_items <= obs_len:
          opponent_offer_keep_format = np.array(observation[last_entry_idx : last_entry_idx + self.num_items], dtype=int)
          if not np.all(opponent_offer_keep_format == -1):
            if self.total_quantities is not None:
              offer_for_us = self.total_quantities - opponent_offer_keep_format
              if np.all(offer_for_us >= 0):
                self.current_opponent_offer_keep_format = offer_for_us
              else:
                print(f"[WARNING] Tough P{self.player_id} Calculated negative quantity in opponent offer. Their Keep: {opponent_offer_keep_format}, Total: {self.total_quantities}")
            else:
              print(f"[WARNING] Tough P{self.player_id} Cannot calculate opponent offer value without total quantities.")

  def _decode_action_to_proposal(self, action_id):
    encoding_base = self.max_quantity_for_encoding + 1
    max_possible_encoded_offer = (encoding_base ** self.num_items) - 1

    if action_id < 0 or action_id > max_possible_encoded_offer:
      return None

    proposal = np.zeros(self.num_items, dtype=int)
    temp_value = action_id
    valid_decoding = True
    for i in range(self.num_items - 1, -1, -1):
      power_of_base = encoding_base ** i
      digit = temp_value // power_of_base
      if digit > self.max_quantity_for_encoding:
        print(f"[WARNING] Decoding error: digit {digit} > max_quantity {self.max_quantity_for_encoding} for action {action_id}")
        valid_decoding = False
        break
      proposal[self.num_items - 1 - i] = digit
      temp_value %= power_of_base

    if not valid_decoding:
      return None

    return proposal

  def _encode_proposal_to_action(self, proposal_keep_format):
      """Encodes a proposal (agent's keep format) into an action ID."""
      if proposal_keep_format is None or len(proposal_keep_format) != self.num_items:
          print(f"[ERROR] Invalid proposal format for encoding: {proposal_keep_format}")
          return None
      if np.any(proposal_keep_format < 0) or np.any(proposal_keep_format > self.max_quantity_for_encoding):
          print(f"[ERROR] Proposal quantities out of bounds for encoding: {proposal_keep_format}")
          return None

      encoding_base = self.max_quantity_for_encoding + 1
      action_id = 0
      for i in range(self.num_items):
          action_id += proposal_keep_format[i] * (encoding_base ** (self.num_items - 1 - i))

      # Validate against max possible action ID derived from encoding base
      max_possible_encoded_offer = (encoding_base ** self.num_items) -1
      if action_id < 0 or action_id > max_possible_encoded_offer:
          print(f"[ERROR] Calculated action_id {action_id} is out of theoretical bounds.")
          
          return None

      return int(action_id) # Ensure it's an integer

  def _calculate_utility(self, proposal_keep_format):
    if proposal_keep_format is None or len(proposal_keep_format) != self.num_items:
      return -np.inf
    return np.dot(proposal_keep_format, self.valuation_vector)
  
  def step(self, state):
    """
    Selects an action based on the tough (greedy) strategy.
    Args:
        state: The current game state
    Returns:
        action: The action ID to take
    """
    self.process_observation(state.observation_tensor(self.player_id))
    if self.walk_away_value > np.dot(self.valuation_vector, self.total_quantities):
        self.action = "WALK"
        action = self._create_walk_action(state)
        return action
    # Reset the stored offer for this step
    self.current_counter_offer_give_format = None

    probs = self.action_probabilities(state)

    action = None
    if not probs:
        print(f"[INFO] Tough P{self.player_id} found no greedy offers. Choosing to WALK.")
        self.action = "WALK"
        action = self._create_walk_action(state)
        if action is None:
             print(f"[ERROR] Tough P{self.player_id} could not determine walk action. Defaulting to action 0 (Accept?).")
             action = 0
    else:
        action = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
        self.action = "COUNTEROFFER" # Assume chosen action is a counteroffer
        print(f"[INFO] Tough P{self.player_id} chose greedy action {action}")

        proposal_keep_format = self._decode_action_to_proposal(action)
        if proposal_keep_format is not None and self.total_quantities is not None:
            if len(proposal_keep_format) == len(self.total_quantities):
                offer_give_format = self.total_quantities - proposal_keep_format
                if np.all(offer_give_format >= 0):
                    self.current_counter_offer_give_format = offer_give_format.tolist() # Store as list
                    print(f"[DEBUG] Tough P{self.player_id} calculated give offer: {self.current_counter_offer_give_format}")
                else:
                    print(f"[ERROR] Tough P{self.player_id} calculated negative 'give' quantity. Keep: {proposal_keep_format}, Total: {self.total_quantities}")
            else:
                 print(f"[ERROR] Tough P{self.player_id} length mismatch between decoded proposal and total quantities.")
        elif self.total_quantities is None:
            print(f"[ERROR] Tough P{self.player_id} cannot calculate 'give' offer because total_quantities is None.")
        else: 
            print(f"[ERROR] Tough P{self.player_id} cannot calculate 'give' offer because decoding action {action} failed.")
        # ----------------------------------------------------------

    legal_actions = state.legal_actions()
    if action not in legal_actions:
        print(f"[ERROR] Tough P{self.player_id} chosen action {action} is NOT LEGAL. Legal: {legal_actions}. Defaulting to WALK.")
        self.action = "WALK"
        self.current_counter_offer_give_format = None 
        walk_action = self._create_walk_action(state)
        action = walk_action if walk_action is not None else (legal_actions[0] if legal_actions else 0)

    return action

  def action_probabilities(self, state):
    """Calculates probabilities for the best greedy offers."""
    cur_player = state.current_player()
    if cur_player != self.player_id:
         print(f"[ERROR] Tough P{self.player_id} asked to calculate probs for P{cur_player}")
         return {}

    legal_actions = state.legal_actions()
    # Update state info (quantities, walk value) before evaluating
    self.process_observation(state.observation_tensor(self.player_id), self.num_items)

    if self.total_quantities is None:
        print(f"[ERROR] Tough P{self.player_id} cannot determine total quantities. Cannot evaluate offers.")
        return {} # Return empty dict if critical info missing

    offer_values = {}
    valid_offer_actions = []

    for action in legal_actions:
        action_str = state.action_to_string(cur_player, action)

        # Skip Accept and Walk actions
        if "Agreement" in action_str or "walk away" in action_str.lower() or ("get" in action_str and "points" in action_str):
            continue

        # Decode the action to see what the agent would keep
        proposal_keep_format = self._decode_action_to_proposal(action)

        if proposal_keep_format is None:
            # print(f"[DEBUG] Tough P{self.player_id} could not decode action {action} ({action_str}) into proposal.")
            continue

        # Sanity check decoded proposal against total quantities
        if np.any(proposal_keep_format < 0) or np.any(proposal_keep_format > self.total_quantities):
             print(f"[WARNING] Tough P{self.player_id} decoded invalid proposal {proposal_keep_format} for action {action} (Total: {self.total_quantities}). Skipping.")
             continue

        value = self._calculate_utility(proposal_keep_format)
        offer_values[action] = value
        valid_offer_actions.append(action)
        # print(f"[DEBUG] Tough P{self.player_id} evaluated Action {action} ({action_str}) -> Keep {proposal_keep_format} -> Value {value:.2f}")


    if not valid_offer_actions:
        # If no valid offers found (e.g., only Accept/Walk legal, or parsing failed)
        print(f"[WARNING] Tough P{self.player_id} found no valid offers to evaluate among legal actions.")
        return {} # Return empty dict, step method should handle this (e.g., walk)

    # Find the maximum value among valid offers
    max_value = -np.inf
    for action in valid_offer_actions:
        if offer_values[action] > max_value:
            max_value = offer_values[action]

    greedy_actions = [a for a in valid_offer_actions if offer_values[a] == max_value]

    if not greedy_actions:
         print(f"[ERROR] Tough P{self.player_id}: No greedy actions found despite valid offers. This shouldn't happen.")
         return {} 

    print(f"[INFO] Tough P{self.player_id} identified {len(greedy_actions)} greedy actions with max value {max_value:.2f}")

    # --- New Strategy: Try to give away 1 of least valued item --- 
    first_greedy_action = greedy_actions[0]
    best_keep_proposal = self._decode_action_to_proposal(first_greedy_action)

    if best_keep_proposal is not None:
        min_positive_value = np.inf
        least_valued_item_index = -1
        # Find the index of the item with the smallest positive valuation
        for i in range(self.num_items):
            if self.valuation_vector[i] > 0 and self.valuation_vector[i] < min_positive_value:
                min_positive_value = self.valuation_vector[i]
                least_valued_item_index = i

        if least_valued_item_index != -1: 
            # Create the modified proposal
            modified_keep_proposal = np.copy(best_keep_proposal)
            modified_keep_proposal[least_valued_item_index] -= 1

            # Check if the modified proposal is valid (quantities >= 0)
            if modified_keep_proposal[least_valued_item_index] >= 0:
                # Encode the modified proposal back to an action ID
                modified_action = self._encode_proposal_to_action(modified_keep_proposal)
                
                if modified_action is not None and modified_action in legal_actions:
                    print(f"[INFO] Tough P{self.player_id} Applying modified strategy: giving away 1 of item {least_valued_item_index}. Action: {modified_action}")
                    return {modified_action: 1.0} # Play the modified action
                else:
                    print(f"[INFO] Tough P{self.player_id} Modified action {modified_action} is not legal or encoding failed. Falling back to original greedy.")
            else:
                print(f"[INFO] Tough P{self.player_id} Cannot give away least valued item {least_valued_item_index} (quantity is 0). Falling back to original greedy.")
        else:
            print(f"[INFO] Tough P{self.player_id} No positively valued item found to give away. Falling back to original greedy.")
    else:
        print(f"[WARNING] Tough P{self.player_id} Could not decode best greedy action {first_greedy_action}. Falling back to original greedy.")
    # --- End New Strategy ---

    # Fallback to original strategy: uniform probability over all greedy actions
    return {a: 1/len(greedy_actions) for a in greedy_actions}

  def _create_accept_action(self, state):
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
                
        print("[WARNING] No Agreement action found in legal actions, defaulting to WALK")
        return 
    
  def _create_walk_action(self, state):
        """
        Create the action format for WALK in OpenSpiel.
        
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

    
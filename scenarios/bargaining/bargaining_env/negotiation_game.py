import json
import os
import pickle
import sys
import time

import torch

import numpy as np
import pyspiel

from game_runner import NegotitaionGame
import agents.llm_agent as llm_agent
import agents.llm_agent_openspiel_wrapper as os_wrapper
import agents.soft_agent as soft_agent
import agents.tough_agent as tough_agent
import agents.nfsp_agent_wrapper as nfsp_wrapper
import agents.rnad_agent_wrapper as rnad_wrapper
from agents.tit_for_tat_agent import TitForTatNegotiator as tit_for_tat_agent
from agents.conecession_schedule_agent import ConcessionScheduleNegotiator as conceder_agent
# import agents.alpha_zero_agent_wrapper as az_wrapper
from eval.metrics import compute_pareto_frontier
from eval.game_data import GameData
from .helpers import find_allocation_less_than_outside_offer_dp

sys.path.append("../caif_negotiation/")
from test_game_eval import GameHistory  # noqa: E402

MAX_QUANTITY = 10


def run_game(circle1: int, circle2: int, games: int, max_rounds: int, date: str, game_title: str, llm_model_p1: str, llm_model_p2: str, discount: float, is_openspiel: bool = False, num_items: int = 3):
    """
    Runs a series of negotiation games for a specific circle, tracking comprehensive metrics.

    Args:
        circle1 (int): The circle parameter influencing allocation strategies for player 1.
        circle2 (int): The circle parameter influencing allocation strategies for player 2.
        games (int): Number of games to simulate.
        max_rounds (int): Maximum number of rounds per game.
        date (str): Date identifier for result files.
        game_title (str): Title identifier for the game series.
        llm_model_p1 (str): Type of LLM agent being used (e.g., "openai_4o").
        llm_model_p2 (str): Type of LLM agent being used (e.g., "openai_o3_mini").
        discount (float): Discount factor for future rounds.
        is_openspiel (bool): Whether to use OpenSpiel for the game environment.
        num_items (int): Number of items in the negotiation.
    """

    def _make_openspiel_agent(model_name: str, circle: int, player_id: int, game) -> object:
        """Factory for OpenSpiel agents to avoid duplicated instantiation logic."""
        if model_name == "soft_agent":
            return soft_agent.SoftNegotiator(game=game, player_id=player_id, max_rounds=max_rounds)
        if model_name == "tough_agent":
            return tough_agent.ToughNegotiator(
                game=game,
                valuation_vector=[0] * num_items,
                num_items=num_items,
                player_id=player_id,
                max_rounds=max_rounds,
            )
        if model_name in ("nfsp_agent", "nfsp_agent_2", "nfsp_agent_both_circ_bg4"):
            return nfsp_wrapper.NFSPAgentWrapper(game=game, player_id=player_id, checkpoint_dir="agents")
        if model_name == "rnad_agent":
            ckpt = "/Users/gabesmithline/Desktop/rnad_bg6_step_748000.pkl"
            return rnad_wrapper.RNaDAgentWrapper(game=game, player_id=player_id, checkpoint_path=ckpt)
        if model_name == "tit_for_tat_agent":
            return tit_for_tat_agent(
                game=game,
                player_id=player_id,
                num_items=num_items,
                valuation_vector=[0] * num_items,
                max_rounds=max_rounds,
            )
        if model_name == "conceder_agent":
            return conceder_agent(
                game=game,
                valuation_vector=[0] * num_items,
                num_items=num_items,
                player_id=player_id,
                max_rounds=max_rounds,
                discount_factor=discount,
                acceptance_slack=0.05,
            )
        # Default: LLM wrapper
        return os_wrapper.LLMAgentOpenSpielWrapper(
            llm_type=model_name,
            model=model_name,
            player_num=player_id,
            circle=circle,
            num_items=num_items,
        )

    all_game_data = []
    completed_games = 0
    attempts = 0
    max_attempts = games
    
    if not is_openspiel:
        while completed_games < games and attempts < max_attempts:
            attempts += 1
            if (attempts) % 10 == 0:
                print(f"Game attempt {attempts}, completed {completed_games} of {games}")
                sleep_duration = 2 * np.random.randint(55, 60)  # Sleep for ~2 minutes
                print(f"Sleeping for {sleep_duration} seconds to respect rate limits.")
                time.sleep(sleep_duration)
            player1_agent = llm_agent.LLMAgent(llm_type=llm_model_p1, model=llm_model_p1, player_num=0)
            player2_agent = llm_agent.LLMAgent(llm_type=llm_model_p2, model=llm_model_p2, player_num=1)
            game = NegotitaionGame(
                player1_agent=player1_agent,
                player2_agent=player2_agent,
                num_items=num_items,
                item_value_range=[1, 101], #excludes 101 range 1-100
                gamma=discount,
                max_rounds=max_rounds,
                circle1=circle1,
                circle2=circle2 
            )
        
            pareto_front = compute_pareto_frontier(
                game.player_values[0],
                game.player_values[1],
                game.num_items,
                game.items,
                game.outside_offer_values
            )

            allocations_less_than_outside_offer = None
            if circle1 in (5, 6) or circle2 in (5, 6):
                allocations_less_than_outside_offer = []

                allocation_p1 = find_allocation_less_than_outside_offer_dp(
                    items=game.items,
                    player_valuations=game.player_values[0],
                    outside_offer=game.outside_offer_values[0],
                    player_num=1
                )
                if allocation_p1:
                    allocations_less_than_outside_offer.append({
                        'player': 1,
                        'allocation': list(allocation_p1.values())
                    })
                else:
                    allocations_less_than_outside_offer.append({
                        'player': 1,
                        'allocation': [0] * game.num_items
                    })
                    print(f"[INFO] No feasible < outside_offer allocation for Player 1 in Game attempt {attempts}.")

                # Find allocations where Player 2's utility is less than their outside offer
                allocation_p2 = find_allocation_less_than_outside_offer_dp(
                    items=game.items,
                    player_valuations=game.player_values[1],
                    outside_offer=game.outside_offer_values[1],
                    player_num=2
                )
                if allocation_p2:
                    allocations_less_than_outside_offer.append({
                        'player': 2,
                        'allocation': list(allocation_p2.values())
                    })
                else:
                    allocations_less_than_outside_offer.append({
                        'player': 2,
                        'allocation': [0] * game.num_items
                    })
                    print(f"[INFO] No feasible < outside_offer allocation for Player 2 in Game attempt {attempts}.")

                print(f"[DEBUG] Game attempt {attempts} allocations_less_than_outside_offer: {allocations_less_than_outside_offer}")

            print(f"[DEBUG] game.items: {game.items}")
            print(f"[DEBUG] allocations_less_than_outside_offer: {allocations_less_than_outside_offer}")

            game_history = GameHistory(
                agent_1_name="Agent1",
                agent_2_name="Agent2",
                num_items=game.num_items,
                items=torch.tensor(game.items),
                agent_1_values=torch.tensor(game.player_values[0]),
                agent_2_values=torch.tensor(game.player_values[1]),
                agent_1_outside_value=game.outside_offer_values[0],
                agent_2_outside_value=game.outside_offer_values[1]
            )
            game_history.agent_1_offers = []
            game_history.agent_2_offers = []

        
            game_data = GameData(
                circle=(circle1, circle2),
                date=date,
                agent1=f"Agent1_{llm_model_p1}",
                agent2=f"Agent2_{llm_model_p2}"
            )

            print(f"[INFO] Starting Game attempt {attempts} (completed {completed_games} of {games}) for Circle {circle1 if game.current_player == 0 else circle2}.")

            # Flag to track if this game had an API failure
            had_api_failure = False

            while game.in_progress and not had_api_failure:
                # Sleep to simulate thinking time and rate-limit API calls
                sleep_duration = circle1 if game.current_player == 0 else circle2 + .5  # Adjust based on desired rate-limiting
                print(f"[DEBUG] Sleeping for {sleep_duration} seconds before next step.")
                sleep_duration = np.random.randint(sleep_duration, sleep_duration + 10)
                time.sleep(sleep_duration)

                # Determine current step, round, and player
                current_step = len(game.history[0]) + len(game.history[1]) + 1
                current_round = (current_step - 1) // 2 + 1
                current_player = 1 if current_step % 2 == 1 else 2
                game.current_round = current_round

                print("\n" + "=" * 80)
                print(f"Game attempt {attempts}, Round {current_round}, Player {current_player}'s turn (Step {current_step})")
                print("=" * 80)

                current_allocation_example = None
                if circle1 in (5, 6) or circle2 in (5, 6) and allocations_less_than_outside_offer is not None:
                    if current_player == 1:
                        current_allocation_example = allocations_less_than_outside_offer[0]['allocation']
                    elif current_player == 2:
                        current_allocation_example = allocations_less_than_outside_offer[1]['allocation']

                print(f"[DEBUG] Current allocation example type: {type(current_allocation_example)}")

                game.step(example_offer_less_than_outside_offer_self=current_allocation_example)
                
                current_agent = game.players[current_player - 1]
                if hasattr(current_agent, 'api_failure') and current_agent.api_failure:
                    print(f"[WARNING] API failure detected in step {current_step}. Skipping this game.")
                    had_api_failure = True
                    break
                    
                action_played = game.players[current_player - 1].action.upper()

                game_data.add_round_data(
                        prompt=game.players[current_player - 1].current_prompt,
                        response=game.players[current_player - 1].current_response,  
                        action=action_played
                    )

                if "WALK" in action_played or "ACCEPT" in action_played:
                    game.in_progress = False

            if not had_api_failure:
                all_game_data.append(game_data)
                completed_games += 1
                print(f"[INFO] Game {completed_games} completed successfully.")
            else:
                print(f"[INFO] Game attempt {attempts} skipped due to API failure.")

        if completed_games < games:
            print(f"[WARNING] Only completed {completed_games} of {games} requested games after {attempts} attempts due to API failures.")
        
       
        all_data = {
            "date": date,
            "circle_p1": circle1,
            "circle_p2": circle2,
            "all_game_data": [gd.to_dict() for gd in all_game_data]
        }
        all_games_filename = f'all_game_data_{date}_{completed_games}_{game_title}_circle_p1_{circle1}_circle_p2_{circle2}.json'
        with open(all_games_filename, "w") as f:
            json.dump(all_data, f, indent=4)
            #json.pickle(all_data, f)
        print(f"[INFO] Saved all GameData to JSON file: {all_games_filename}.")

        #save to pickle optinally
        all_games_filename_pkl = f'all_game_data_{date}_{completed_games}_{game_title}_circle_p1_{circle1}_circle_p2_{circle2}.pkl'
        with open(all_games_filename_pkl, "wb") as pf:
            pickle.dump(all_data, pf)
        print(f"[INFO] Saved all GameData as a pickle to {all_games_filename_pkl}.")
    else:
        try:
            if 'pyspiel' not in sys.modules:
                raise ImportError("OpenSpiel not installed. To use OpenSpiel functionality, install it with: pip install open_spiel")
            
            print("[INFO] Using OpenSpiel for game environment")
            
            for game_num in range(games):
                print(f"[INFO] Starting game {game_num+1}/{games}")
                
                params = {
                    "enable_proposals": True,      # Whether proposals are allowed
                    "enable_utterances": False,    # Whether utterances are allowed (set to False to simplify)
                    "num_items": 3,               # Number of items to negotiate over
                    "discount": discount, 
                    'min_value': 1,
                    'max_value': 100,
                    'max_rounds': max_rounds,
                    'max_quantity': 10,
                    "item_quantities": "7,4,1" 
                }
                game = pyspiel.load_game("negotiation", params)
                
                # --- Agent Instantiation ---
                player1_agent = _make_openspiel_agent(llm_model_p1, circle1, 0, game)
                player2_agent = _make_openspiel_agent(llm_model_p2, circle2, 1, game)
                    
                game_start_time = time.time()
                
                state = game.new_initial_state()
                
                game_data = GameData(
                    circle=(circle1, circle2),
                    date=date,
                    agent1=f"Agent1_{llm_model_p1}",
                    agent2=f"Agent2_{llm_model_p2}"
                )
                
                had_api_failure = False
                print(f"[DEBUG] Game parameters: max_rounds={max_rounds}, num_items={num_items}, discount={discount}")
                print(f"[DEBUG] Initial state string: {state}")
                try:
                    print(f"[DEBUG] Game type: {game.get_type()}")
                    print(f"[DEBUG] Game parameters: {game.get_parameters()}")
                    print(f"[DEBUG] Num players: {game.num_players()}")
                    print(f"[DEBUG] Initial state current player: {state.current_player()}")
                    print(f"[DEBUG] Initial legal actions: {state.legal_actions()}")

                except Exception as e:
                    print(f"[ERROR] Error during game initialization: {e}")
                    had_api_failure = True
                allocations_less_than_outside_offer = None
                
                    
                
                current_round = 1
                game_over = False
                #startign game 
                while not state.is_terminal() and not had_api_failure and not game_over and current_round <= max_rounds:
                    try:
                        if state.is_chance_node():
                            print("[DEBUG] Processing mid-game chance node...")
                            chance_outcomes = state.chance_outcomes()
                            
                            if not chance_outcomes:
                                print("[ERROR] No chance outcomes available")
                                had_api_failure = True
                                break
                                
                            action, prob = chance_outcomes[0]
                            print(f"[DEBUG] Selected chance action: {action} with probability {prob}")
                            
                            state.apply_action(action)
                            if circle1 in (5, 6) or circle2 in (5, 6):
                                allocations_less_than_outside_offer = []

                                if isinstance(player1_agent, os_wrapper.LLMAgentOpenSpielWrapper): #
                                    player1_agent.process_observation(observation=state.observation_tensor(player1_agent.player_num))
                                    try:
                                        allocation_p1 = find_allocation_less_than_outside_offer_dp(
                                            items=player1_agent.items,
                                            player_valuations=player1_agent.valuation,
                                            outside_offer=player1_agent.walk_away_value,
                                            player_num=1
                                        )
                                        
                                        if allocation_p1:
                                            player1_agent.example_offer_less_than_outside_offer_self = list(allocation_p1.values())
                                            allocations_less_than_outside_offer.append({
                                                'player': 1,
                                                'allocation': list(allocation_p1.values())
                                            })
                                        else:
                                            player1_agent.example_offer_less_than_outside_offer_self = [0] * num_items
                                            allocations_less_than_outside_offer.append({
                                                'player': 1,
                                                'allocation': [0] * num_items
                                            })
                                            print(f"[INFO] No feasible < outside_offer allocation for Player 1.")
                                    except Exception as e:
                                        print(f"[WARNING] Error computing allocations for Player 1: {e}")
                                        allocations_less_than_outside_offer.append({
                                            'player': 1,
                                            'allocation': [0] * num_items
                                        })
                                if isinstance(player2_agent, os_wrapper.LLMAgentOpenSpielWrapper): #
                                    player2_agent.process_observation(observation=state.observation_tensor(player1_agent.player_num))
                                    try:
                                        allocation_p2 = find_allocation_less_than_outside_offer_dp(
                                            items=player2_agent.items,
                                            player_valuations=player2_agent.valuation,
                                            outside_offer=player2_agent.walk_away_value,
                                            player_num=2
                                        )
                                        
                                        if allocation_p2:
                                            player2_agent.example_offer_less_than_outside_offer_self = list(allocation_p2.values())
                                            allocations_less_than_outside_offer.append({
                                                'player': 2,
                                                'allocation': list(allocation_p2.values())
                                            })
                                        else:
                                            player2_agent.example_offer_less_than_outside_offer_self = [0] * num_items
                                            allocations_less_than_outside_offer.append({
                                                'player': 2,
                                                'allocation': [0] * num_items
                                            })
                                            print(f"[INFO] No feasible < outside_offer allocation for Player 1.")
                                    except Exception as e:
                                        import traceback
                                        print(f"[WARNING] Error computing allocations for Player 2: {e}")
                                        print(f"[WARNING] Stack trace: {traceback.format_exc()}")
                                        
                                        allocations_less_than_outside_offer.append({
                                            'player': 2,
                                            'allocation': [0] * num_items
                                        })
                            continue
                            
                        current_player = state.current_player()
                        print(f"[DEBUG] Current player: {current_player}")
                        
                        print(f"[DEBUG] State string: {state}")

                        print(f"[DEBUG] State is terminal: {state.is_terminal()}")
                        print(f"[DEBUG] State is chance node: {state.is_chance_node()}")
                        
                        if current_player < 0:
                            print(f"[ERROR] Invalid player index: {current_player}")
                            break
                        legal_actions = state.legal_actions()
                        other_player = 1 - current_player
                        try:
                            other_legal_actions = state.legal_actions(other_player)
                        except Exception as e:
                            print(f"[DEBUG] Cannot get legal actions for other player: {e}")
                        
                        observation = state.observation_tensor(current_player)
                        try:
                            other_observation = state.observation_tensor(other_player)
                        except Exception as e:
                            print(f"[DEBUG] Cannot get observation for other player: {e}")
                            
                        
                        
                        current_agent = player1_agent if current_player == 0 else player2_agent
                        
                        if len(observation) > 6:
                            observed_round = int(observation[6]) + 1
                        else:
                            observed_round = current_round
                        
                        print("\n" + "=" * 80)
                        print(f"Game {game_num+1}, Round {observed_round}, Player {current_player+1}'s turn")
                        print("=" * 80)
                    
                    
                        
                        #sleep_duration = circle1 if current_player == 0 else circle2 + 0.5
                        #sleep_duration = np.random.randint(int(sleep_duration), int(sleep_duration) + 10)
                        #print(f"[DEBUG] Sleeping for {sleep_duration} seconds before next step.")
                        #time.sleep(sleep_duration)
                        
                        turn_start_time = time.time()
                        
                        action = current_agent.step(state=state)
                        
                        turn_time = time.time() - turn_start_time
                        print(f"[INFO] Player {current_player+1}'s turn took {turn_time:.2f} seconds")
                        if isinstance(current_agent, os_wrapper.LLMAgentOpenSpielWrapper):
                            game_data.add_round_data(
                                prompt=current_agent.current_prompt,
                                response=current_agent.current_response,
                                action=current_agent.action
                            )
                        else:
                            # Handle non-LLM agents (Soft, Tough, NFSP)
                            response_data = {"action": current_agent.action}
                            # Check if it's the last round and player 2's turn
                            if current_round == max_rounds and hasattr(current_agent, 'player_id') and current_agent.player_id == 1:
                                response_data = {"action": "WALK"}
                                current_agent.action = "WALK"
                            elif current_agent.action == "COUNTEROFFER" and hasattr(current_agent, 'current_counter_offer_give_format'):
                                response_data["offer"] = current_agent.current_counter_offer_give_format
                            
                            response_str = str(response_data)
                            
                            # Prepare prompt string with agent's valuation and items
                            '''
                            if hasattr(current_agent, 'valuation_vector') and current_agent.valuation_vector is not None:
                                val_vector = current_agent.valuation_vector
                                val_str = val_vector.tolist() if isinstance(val_vector, np.ndarray) else val_vector
                                prompt_str += f"valuation vector: {val_str}"
                            
                            if hasattr(current_agent, 'items') and current_agent.items is not None:
                                prompt_str += f", item counts: {current_agent.items}"
                            
                            if hasattr(current_agent, 'walk_away_value') and current_agent.walk_away_value is not None:
                                prompt_str += f", outside_offer: {current_agent.walk_away_value}"

                            # If we don't have the specific attributes, fall back to a generic prompt
                            if not prompt_str:
                                prompt_str = f"{current_agent.__class__.__name__} state info"
                            '''
                            game_data.add_round_data(
                                prompt=current_agent.prompt,
                                response=response_str,
                                action=current_agent.action
                            )

                        
                        if hasattr(current_agent, 'api_failure') and current_agent.api_failure:
                            print(f"[WARNING] API failure detected. Skipping this game.")
                            had_api_failure = True
                            break
                        
                        print(f"[INFO] Player {current_player+1} action: {action}")
                        
                 
                        if current_round == max_rounds and current_player == 1:
                            
                            if current_agent.action == "COUNTEROFFER":
                                print(f"[INFO] Last round counteroffer by Player 2 treated as WALK")
                                for legal_action in legal_actions:
                                    action_str = state.action_to_string(current_player, legal_action)
                                    if "get" in action_str and "points" in action_str:
                                        action = legal_action
                                        break
                                else:
                                    action = 1
                        
                        try:
                            action_value = int(action)
                            print(f"[DEBUG] Applying action value: {action_value} for action_type: {current_agent.action}")
                            state.apply_action(action_value)
                        except (ValueError, TypeError) as e:
                            print(f"[ERROR] Failed to convert action to integer: {action}, {e}")
                            safe_action = None
                            for legal_action in legal_actions:
                                action_str = state.action_to_string(current_player, legal_action)
                                if current_agent.action == "ACCEPT" and "Agreement" in action_str:
                                    safe_action = legal_action
                                    break
                                elif current_agent.action == "WALK" and "get" in action_str and "points" in action_str:
                                    safe_action = legal_action
                                    break
                            
                            if safe_action is not None:
                                print(f"[DEBUG] Using safe alternative action: {safe_action}")
                                state.apply_action(safe_action)
                            else:
                                print(f"[ERROR] Could not find safe alternative action. Using first legal action.")
                                if legal_actions:
                                    state.apply_action(legal_actions[0])
                                else:
                                    had_api_failure = True
                                    break
                        
                        if current_agent.action == "ACCEPT":
                            print(f"[INFO] Player {current_player+1} accepted the offer. Game is over.")
                            game_over = True
                            break
                        
                        if current_agent.action == "WALK":
                            print(f"[INFO] Player {current_player+1} walked away. Game is over.")
                            game_over = True
                            break
                            
                        if current_player == 1:  
                            current_round += 1
                            
                        if current_round > max_rounds:
                            print(f"[INFO] Maximum rounds ({max_rounds}) reached. Game ending.")
                            game_over = True
                            break
                        
                    except Exception as e:
                        print(f"[ERROR] Exception during agent step: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        had_api_failure = True
                        break
                
                if not had_api_failure:
                    returns = state.returns()
                    print(f"[INFO] Game complete. Returns: {returns}")
                    
                    all_game_data.append(game_data)
                    completed_games += 1
                    print(f"[INFO] Game {completed_games} completed successfully.")
                else:
                    print(f"[INFO] Game {game_num+1} skipped due to API failure or error.")
            
            if completed_games > 0:
                print("SAVING GAME DATA")
                all_data = {
                    "date": date,
                    "circle_p1": circle1,
                    "circle_p2": circle2,
                    "all_game_data": [gd.to_dict() for gd in all_game_data]
                }
                
                all_games_filename = f'openspiel_game_data_{date}_{completed_games}_{game_title}_circle_p1_{circle1}_circle_p2_{circle2}.json'
                with open(all_games_filename, "w") as f:
                    json.dump(all_data, f, indent=4)
                print(f"[INFO] Saved all OpenSpiel GameData to JSON file: {all_games_filename}.")
                
                all_games_filename_pkl = f'openspiel_game_data_{date}_{completed_games}_{game_title}_circle_p1_{circle1}_circle_p2_{circle2}.pkl'
                with open(all_games_filename_pkl, "wb") as pf:
                    pickle.dump(all_data, pf)
                print(f"[INFO] Saved all OpenSpiel GameData as a pickle to {all_games_filename_pkl}.")
                
        except ImportError as e:
            print(f"[ERROR] Could not use OpenSpiel: {e}")
        except Exception as e:
            print(f"[ERROR] Error running OpenSpiel games: {e}")

import json
import pickle


class GameData:
	def __init__(self, circle, date, agent1, agent2):
		"""
		circle: An integer or string indicating which circle/stage of the experiment
		date: String or datetime indicating the date/time of the game
		agent1: Identifier for agent 1
		agent2: Identifier for agent 2
		"""
		self.circle = circle
		self.date = date
		self.agent1 = agent1
		self.agent2 = agent2

		# Store a list of rounds with prompts/responses/actions
		# Each entry can be a dict with keys "prompt", "response", "action"
		self.round_data = []

		# Store the final outcome separately
		self.outcome = None

	def add_round_data(self, prompt, response, action):
		"""
		Store data from a single round.

		prompt: The prompt text shown to the agent
		response: The raw text (or structured data) from the agent's response
		action: A string or structured data describing the action (ACCEPT, WALK, COUNTEROFFER, etc.)
		metrics: A dictionary of metrics for the round
		"""
		self.round_data.append({
			"prompt": prompt,
			"response": response,
			"action": action,
		})

	def set_outcome(self, outcome):
		"""
		Set or update the final outcome of the game.
		"""
		self.outcome = outcome

	@classmethod
	def from_dict(cls, data):
		"""
		Create a GameData instance from a dictionary, typically loaded from JSON.
		"""
		game_data = cls(
			circle=data["circle"],
			date=data["date"],
			agent1=data["agent1"],
			agent2=data["agent2"]
		)
		game_data.round_data = data.get("round_data", [])
		game_data.outcome = data.get("outcome")
		return game_data

	def save_to_json(self, filename):
		"""
		Save the GameData as JSON to a specified file.
		"""
		with open(filename, "w") as f:
			json.dump(self.to_dict(), f)

	def to_dict(self):
		"""
		Convert the GameData instance into a dictionary with all data
		in JSON-serializable formats.
		"""
		data = {
			"circle": self.circle,
			"date": self.date,
			"agent1": self.agent1,
			"agent2": self.agent2,
			"round_data": self.round_data  # Assuming round_data is a list of dicts
		}
		return data

	@classmethod
	def load_from_json(cls, filename):
		"""
		Load the GameData from a JSON file.
		"""
		with open(filename, "r") as f:
			data = json.load(f)
		return cls.from_dict(data)

	def save_pickle(self, filename):
		"""
		Pickle the GameData object to a specified file.
		"""
		with open(filename, "wb") as f:
			pickle.dump(self, f)

	@classmethod
	def load_pickle(cls, filename):
		"""
		Load a pickled GameData object from a file.
		"""
		with open(filename, "rb") as f:
			return pickle.load(f)



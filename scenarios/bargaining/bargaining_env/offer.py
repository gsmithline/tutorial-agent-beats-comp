from dataclasses import dataclass

@dataclass
class Offer:
    player: int
    offer: list[int] | bool

    def to_dict(self):
        """Serializes Offer into a dictionary"""
        return {
            "player": self.player,
            "offer": self.offer
        }

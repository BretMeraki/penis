# forest_app/modules/memory.py

import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MemorySystem:
    """
    MemorySystem manages symbolic "Memory Echoes" that capture key moments in the user's journey.

    Each Memory Echo is stored as a dictionary with the following keys:
      - label: A brief title or identifier for the moment.
      - description: A detailed description of the moment.
      - timestamp: The UTC timestamp when the echo was recorded (ISO format).
      - emotional_imprint (optional): A value (numeric or descriptive) indicating the emotional intensity.
      - symbolic_echo (optional): A symbolic representation (e.g., "Spark", "Ember") of the moment.
      - archetype_influence (optional): A dictionary (or string) summarizing the active archetype's influence at that moment.

    Methods:
      - store_moment: Saves a new Memory Echo along with any additional details.
      - get_recent_echoes: Retrieves a specified number of the most recent echoes.
      - to_dict / update_from_dict: Methods for serialization and rehydration.
    """

    def __init__(self):
        self.echoes = []  # List of memory echoes.

    def store_moment(
        self,
        label: str,
        description: str,
        emotional_imprint: float = None,
        symbolic_echo: str = None,
        archetype_influence: dict = None,
    ) -> dict:
        """
        Stores a new Memory Echo with detailed information.
        """
        echo = {
            "label": label,
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if emotional_imprint is not None:
            echo["emotional_imprint"] = emotional_imprint
        if symbolic_echo is not None:
            echo["symbolic_echo"] = symbolic_echo
        if archetype_influence is not None:
            echo["archetype_influence"] = archetype_influence

        self.echoes.append(echo)
        logger.info("Stored Memory Echo: %s", echo)
        return echo

    def get_recent_echoes(self, count: int = 2) -> list:
        """
        Returns the most recent `count` memory echoes.
        """
        return self.echoes[-count:] if self.echoes else []

    def to_dict(self) -> dict:
        """
        Serializes the MemorySystem into a dictionary for persistence.
        """
        return {"recent_echoes": self.echoes}

    def update_from_dict(self, data: dict):
        """
        Updates the MemorySystem state from a dictionary.
        """
        self.echoes = data.get("recent_echoes")

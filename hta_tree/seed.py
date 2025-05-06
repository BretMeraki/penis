# forest_app/modules/seed.py

import json
import logging
import uuid
from datetime import datetime, timezone # Added timezone
# --- FIX: Added Union import ---
from typing import List, Optional, Dict, Any, Union
# --- END FIX ---

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- HTA Imports (Direct) ---
# Ensure this import path is correct for your project structure.
from forest_app.hta_tree.hta_tree import HTATree, HTANode
# ─────────────────────────────


class Seed:
    """
    Represents a symbolic Seed within the Forest system, encapsulating a goal
    or intention and its associated HTA plan.
    """
    def __init__(
        self,
        seed_name: str,
        seed_domain: str,
        seed_form: Optional[str] = "",
        description: Optional[str] = "",
        emotional_root_tags: Optional[List[str]] = None,
        shadow_trigger: Optional[str] = "",
        associated_archetypes: Optional[List[str]] = None,
        status: str = "active",
        seed_id: Optional[str] = None,
        # --- FIX: Applying User's Proposed Change (Union + Convert on Init) ---
        created_at: Optional[Union[str, datetime]] = None,
        # --- END FIX ---
        # Store HTA tree as a dictionary for serialization
        hta_tree: Optional[Dict[str, Any]] = None,
    ):
        self.seed_id = seed_id or str(uuid.uuid4())
        self.seed_name = seed_name
        self.seed_domain = seed_domain
        self.seed_form = seed_form or ""
        self.description = description or ""
        self.emotional_root_tags = emotional_root_tags or []
        self.shadow_trigger = shadow_trigger or ""
        self.associated_archetypes = associated_archetypes or []
        self.status = status

        # --- FIX: Applying User's Proposed Change (Union + Convert on Init) ---
        # Store created_at internally as an ISO string
        if isinstance(created_at, datetime):
            # Ensure timezone awareness before formatting
            dt_obj = created_at
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            self.created_at = dt_obj.isoformat()
        elif isinstance(created_at, str):
             # Optionally validate ISO format here if desired
             self.created_at = created_at
        else:
            # Default to current time as ISO string
            self.created_at = datetime.now(timezone.utc).isoformat()
        # --- END FIX ---

        # Initialize hta_tree as an empty dict if None is provided
        self.hta_tree = hta_tree or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Seed object to a dictionary."""
        # self.created_at is already an ISO string due to __init__ logic
        return {
            "seed_id": self.seed_id,
            "seed_name": self.seed_name,
            "seed_domain": self.seed_domain,
            "seed_form": self.seed_form,
            "description": self.description,
            "emotional_root_tags": self.emotional_root_tags,
            "shadow_trigger": self.shadow_trigger,
            "associated_archetypes": self.associated_archetypes,
            "status": self.status,
            "created_at": self.created_at, # Already a string
            "hta_tree": self.hta_tree,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Seed":
        """Creates a Seed object from a dictionary."""
        hta_tree_data = data.get("hta_tree")
        if hta_tree_data is not None and not isinstance(hta_tree_data, dict):
             logger.warning("hta_tree data in Seed.from_dict is not a dict. Attempting to load anyway.")

        return cls(
            seed_name=data.get("seed_name", ""),
            seed_domain=data.get("seed_domain", ""),
            seed_form=data.get("seed_form", ""),
            description=data.get("description", ""),
            emotional_root_tags=data.get("emotional_root_tags", []),
            shadow_trigger=data.get("shadow_trigger", ""),
            associated_archetypes=data.get("associated_archetypes", []),
            status=data.get("status", "active"),
            seed_id=data.get("seed_id"),
            # Pass the string directly to __init__
            created_at=data.get("created_at"),
            hta_tree=hta_tree_data,
        )

    def update_status(self, new_status: str):
        """Updates the status of the seed."""
        old = self.status
        self.status = new_status
        logger.info("Seed '%s' status: '%s' → '%s'.", self.seed_name, old, new_status)

    def update_description(self, new_description: str):
        """Updates the description of the seed."""
        self.description = new_description
        logger.info("Seed '%s' description updated.", self.seed_name)

    def __str__(self) -> str:
        """Returns a JSON string representation of the Seed."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class SeedManager:
    """
    Manages a collection of Seeds in the Forest system's snapshot state.
    """
    def __init__(self):
        self.seeds: Dict[str, Seed] = {}

    def add_seed(self, seed: Seed) -> None:
        """Adds a seed to the manager, using its ID as the key."""
        if not seed or not hasattr(seed, 'seed_id'):
             logger.error("Attempted to add invalid seed object.")
             return
        if seed.seed_id in self.seeds:
            logger.warning("Seed with ID %s already exists in manager. Skipping.", seed.seed_id)
            return
        self.seeds[seed.seed_id] = seed
        logger.info("Added seed '%s' (ID: %s) to SeedManager.", seed.seed_name, seed.seed_id)

    def remove_seed_by_id(self, seed_id: str) -> bool:
        """Removes a seed from the manager by its ID."""
        if seed_id in self.seeds:
            del self.seeds[seed_id]
            logger.info("Removed seed ID %s from SeedManager.", seed_id)
            return True
        logger.warning("Seed ID %s not found for removal in SeedManager.", seed_id)
        return False

    def get_seed_by_id(self, seed_id: str) -> Optional[Seed]:
        """Retrieves a seed by its ID."""
        return self.seeds.get(seed_id)

    def find_seed(self, seed_id: str) -> Optional[Seed]:
        """Alias for get_seed_by_id for potential compatibility."""
        logger.debug("Using find_seed alias for get_seed_by_id: %s", seed_id)
        return self.get_seed_by_id(seed_id)

    def get_all_seeds(self) -> List[Seed]:
        """Returns a list of all seeds currently managed."""
        return list(self.seeds.values())

    def update_seed(self, seed_id: str, **kwargs) -> bool:
        """Updates attributes of an existing seed."""
        seed = self.get_seed_by_id(seed_id)
        if not seed:
            logger.warning("Cannot update: Seed ID %s not found.", seed_id)
            return False
        fields_updated = []
        for key, val in kwargs.items():
             # Prevent direct modification of created_at via this method if desired,
             # or handle it carefully. Let's prevent for now.
            if key == 'created_at':
                 logger.warning("Attempted to update 'created_at' via update_seed. This is generally discouraged. Ignoring.")
                 continue
            if hasattr(seed, key):
                setattr(seed, key, val)
                fields_updated.append(key)
            else:
                logger.warning("No attribute '%s' on Seed. Ignored during update.", key)
        if fields_updated:
            logger.info("Updated seed %s fields: %s.", seed_id, ", ".join(fields_updated))
            return True
        return False

    def plant_seed(
        self,
        raw_intention: str,
        seed_domain: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Seed:
        """
        Creates a new Seed instance, initializes a basic HTA tree for it,
        and adds it to the manager.
        """
        context = additional_context or {}
        seed_name = f"Seed of {raw_intention[:20].strip().capitalize()}"
        seed_form = context.get("seed_form", "A newly planted seedling.")
        emotional_root_tags = context.get("emotional_root_tags", [])
        shadow_trigger = context.get("shadow_trigger", "")
        associated_archetypes = context.get("associated_archetypes", [])

        # __init__ handles created_at default (will be string)
        new_seed = Seed(
            seed_name=seed_name,
            seed_domain=seed_domain,
            seed_form=seed_form,
            description=raw_intention,
            emotional_root_tags=emotional_root_tags,
            shadow_trigger=shadow_trigger,
            associated_archetypes=associated_archetypes,
            status="active",
            hta_tree=None
        )

        try:
            root_node = HTANode(
                id=f"root_{new_seed.seed_id}",
                title=new_seed.seed_name,
                description=new_seed.description,
                priority=1.0,
            )
            tree = HTATree(root=root_node)
            new_seed.hta_tree = tree.to_dict()
            logger.info("Initialized basic HTA tree for seed '%s'.", new_seed.seed_name)
        except Exception as e:
             logger.error("Failed to initialize HTATree/HTANode for seed '%s': %s. Seed HTA will be empty.", new_seed.seed_name, e)
             new_seed.hta_tree = {}

        self.add_seed(new_seed)
        return new_seed

    def evolve_seed(
        self, seed_id: str, evolution_type: str, new_intention: Optional[str] = None
    ) -> bool:
        """
        Applies an evolution to a seed (e.g., reframe, expansion, transformation).
        Currently modifies description/status; HTA modification needs implementation.
        """
        seed = self.get_seed_by_id(seed_id)
        if not seed:
            logger.warning("Cannot evolve: Seed ID %s not found.", seed_id)
            return False

        et = evolution_type.lower()
        evolved = False
        if et == "reframe":
            if new_intention:
                seed.update_description(new_intention)
                logger.info("Seed '%s' reframed.", seed.seed_name)
                evolved = True
            else: logger.warning("Reframe evolution requires 'new_intention'."); return False
        elif et == "expansion":
            if new_intention:
                seed.description += f"\nExpanded: {new_intention}"
                logger.info("Seed '%s' expanded.", seed.seed_name)
                evolved = True
            else: logger.warning("Expansion evolution requires 'new_intention'."); return False
        elif et == "transformation":
            seed.update_status("evolved")
            logger.info("Seed '%s' transformed (status set to evolved).", seed.seed_name)
            evolved = True
        else: logger.warning("Unknown evolution type '%s' for seed %s.", evolution_type, seed_id); return False

        if evolved:
            try:
                tree = HTATree.from_dict(seed.hta_tree)
                if tree and tree.root:
                     logger.info("Placeholder: HTA tree potentially modified for seed '%s' due to %s.", seed.seed_name, et)
                     seed.hta_tree = tree.to_dict()
                else: logger.warning("Could not load HTA tree for seed %s during evolution.", seed_id)
            except Exception as e: logger.error("Error updating HTA tree during seed evolution for %s: %s", seed_id, e)
        return evolved

    def get_seed_summary(self) -> str:
        """Returns a brief summary string of active seeds."""
        active_seeds = [s for s in self.seeds.values() if s.status.lower() == "active"]
        if not active_seeds: return "No active seeds."
        return " • ".join(f"{s.seed_name} ({s.seed_domain})" for s in active_seeds)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the SeedManager state to a dictionary."""
        return {"seeds": {sid: seed.to_dict() for sid, seed in self.seeds.items()}}

    def update_from_dict(self, data: Dict[str, Any]):
        """Updates the SeedManager state from a dictionary."""
        self.seeds.clear()
        seeds_data = data.get("seeds", {})
        if isinstance(seeds_data, dict):
             for seed_id, seed_dict in seeds_data.items():
                 try:
                     seed = Seed.from_dict(seed_dict)
                     if seed.seed_id != seed_id:
                          logger.warning("Seed ID mismatch during load: key '%s' vs data '%s'. Using key.", seed_id, seed.seed_id)
                          seed.seed_id = seed_id
                     self.seeds[seed_id] = seed
                 except Exception as e: logger.error("Failed to load seed with ID %s from dict: %s", seed_id, e)
        else: logger.warning("Seed data in update_from_dict is not a dictionary. Cannot load seeds.")
        logger.info("SeedManager updated from dict. Loaded %d seeds.", len(self.seeds))

    def __str__(self) -> str:
        """Returns a JSON string representation of the SeedManager."""
        return json.dumps(self.to_dict(), indent=2, default=str)

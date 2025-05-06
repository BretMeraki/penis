# forest_app/core/services/hta_service.py

import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
from logging.handlers import RotatingFileHandler

# --- Core & Module Imports ---
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.hta_tree.hta_tree import HTATree, HTANode # Core HTA classes
from forest_app.hta_tree.seed import SeedManager, Seed # To interact with Seed HTA storage
from forest_app.config.settings import settings # Import settings for model configuration
# Import LLM Client and specific response models needed for evolution
from forest_app.integrations.llm import (
    LLMClient,
    HTAEvolveResponse,
    DistilledReflectionResponse, # If distillation happens here
    LLMError,
    LLMValidationError
)
# Import Pydantic models for validation if needed directly
from forest_app.hta_tree.hta_models import HTANodeModel, HTAResponseModel as HTAValidationModel

# --- Feature Flags (Optional - if evolution logic depends on flags) ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    def is_enabled(feature): return True # Default assumption if flags module missing

logger = logging.getLogger(__name__)

# Add at the top of the file
HTA_TREE_TOTAL_OPS = 0
HTA_TREE_FAILURE_COUNT = 0

def increment_hta_failure(reason: str = "unknown"):
    global HTA_TREE_FAILURE_COUNT
    HTA_TREE_FAILURE_COUNT += 1
    logger.error(f"HTA Tree Failure (Service): {reason} | Failure count: {HTA_TREE_FAILURE_COUNT} | Total ops: {HTA_TREE_TOTAL_OPS}")

class HTAService:
    """Human Task Arbiter service."""

    def __init__(self, llm_client=None, seed_manager=None):
        self.tasks = {}
        self.llm_client = llm_client
        self.seed_manager = seed_manager
        self.last_error = None
        self.last_operation_success = True

    def get_health_status(self) -> dict:
        """Returns health and failure status for monitoring."""
        return {
            "failure_count": HTA_TREE_FAILURE_COUNT,
            "total_ops": HTA_TREE_TOTAL_OPS,
            "last_error": self.last_error,
            "last_operation_success": self.last_operation_success,
        }

    async def create_task(self, task_data: Dict[str, Any]) -> str:
        """Create a new task."""
        task_id = str(len(self.tasks) + 1)
        self.tasks[task_id] = {
            "id": task_id,
            "data": task_data,
            "status": "pending",
            "created_at": datetime.now()
        }
        return task_id

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    async def update_task(self, task_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing task."""
        if task_id in self.tasks:
            self.tasks[task_id].update(update_data)
            return self.tasks[task_id]
        return None

    async def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks."""
        return list(self.tasks.values())

    async def _get_active_seed(self, snapshot: MemorySnapshot) -> Optional[Seed]:
        """Helper to get the primary active seed from the snapshot or SeedManager."""
        # This logic might vary based on how active seed is tracked
        # Option 1: Check snapshot first
        active_seed_id = snapshot.component_state.get("seed_manager", {}).get("active_seed_id")
        if active_seed_id and hasattr(self.seed_manager, 'get_seed_by_id'):
            try:
                # Assume get_seed_by_id might be async if it involves DB lookups
                seed = await self.seed_manager.get_seed_by_id(active_seed_id)
                if seed: return seed
            except Exception as e:
                logger.warning(f"Could not get seed by ID {active_seed_id} from snapshot: {e}")

        # Option 2: Fallback to getting the first active seed from SeedManager
        if hasattr(self.seed_manager, 'get_primary_active_seed'):
             # Assume get_primary_active_seed might be async
            return await self.seed_manager.get_primary_active_seed()

        logger.warning("Could not determine active seed.")
        return None


    async def load_tree(self, snapshot: MemorySnapshot) -> Optional[HTATree]:
        """
        Loads the HTA tree, prioritizing the version stored in the active Seed,
        falling back to the snapshot's core_state if necessary.
        Returns an HTATree object or None if not found/invalid.
        """
        global HTA_TREE_TOTAL_OPS
        HTA_TREE_TOTAL_OPS += 1
        self.last_error = None
        self.last_operation_success = True
        logger.debug("Attempting to load HTA tree...")
        current_hta_dict: Optional[Dict] = None
        try:
            active_seed = await self._get_active_seed(snapshot)
        except Exception as e:
            logger.warning(f"Could not get active seed: {e}")
            self.last_error = str(e)
            self.last_operation_success = False
            increment_hta_failure(f"load_tree error: {e}")
            return None
        # Priority 1: Load from active seed
        if active_seed and hasattr(active_seed, 'hta_tree') and isinstance(active_seed.hta_tree, dict) and active_seed.hta_tree.get('root'):
            logger.debug(f"Loading HTA tree from active seed ID: {getattr(active_seed, 'seed_id', 'N/A')}")
            current_hta_dict = active_seed.hta_tree
        # Priority 2: Load from snapshot core_state
        elif isinstance(snapshot.core_state, dict) and snapshot.core_state.get('hta_tree'):
            logger.warning("Loading HTA tree from snapshot core_state (fallback).") # Log as warning - Seed should ideally be source of truth
            current_hta_dict = snapshot.core_state.get('hta_tree')
        else:
            logger.warning("Could not find HTA tree dictionary in active seed or snapshot core_state.")
            self.last_error = "No HTA tree found in seed or snapshot."
            self.last_operation_success = False
            return None

        # Parse the dictionary into an HTATree object
        if current_hta_dict and isinstance(current_hta_dict, dict):
            try:
                tree = HTATree.from_dict(current_hta_dict)
                if tree and tree.root:
                    logger.info(f"Successfully loaded HTA tree with root: {tree.root.id} - '{tree.root.title}'")
                    return tree
                else:
                    logger.error("Failed to parse valid HTATree object from loaded dictionary (root missing?).")
                    self.last_error = "Parsed HTATree missing root."
                    self.last_operation_success = False
                    return None
            except ValueError as ve:
                logger.error(f"ValueError parsing HTA tree dictionary: {ve}")
                self.last_error = str(ve)
                self.last_operation_success = False
                increment_hta_failure(f"load_tree error: {ve}")
                return None
            except Exception as e:
                logger.exception(f"Unexpected error parsing HTA tree dictionary: {e}")
                self.last_error = str(e)
                self.last_operation_success = False
                increment_hta_failure(f"load_tree error: {e}")
                return None
        else:
            logger.error("Loaded HTA data is not a valid dictionary.")
            self.last_error = "Loaded HTA data is not a valid dictionary."
            self.last_operation_success = False
            return None


    async def save_tree(self, snapshot: MemorySnapshot, tree: HTATree) -> bool:
        """
        Saves the current state of the HTATree object back to the active Seed
        and the snapshot's core_state.

        Args:
            snapshot: The MemorySnapshot object (its core_state will be updated).
            tree: The HTATree object to save.

        Returns:
            True if saving was successful (at least to the snapshot), False otherwise.
        """
        global HTA_TREE_TOTAL_OPS
        HTA_TREE_TOTAL_OPS += 1
        self.last_error = None
        self.last_operation_success = True
        if not tree or not tree.root:
            logger.error("Cannot save HTA tree: Tree object or root node is missing.")
            self.last_error = "Tree object or root node is missing."
            self.last_operation_success = False
            increment_hta_failure("save_tree error: Tree object or root node is missing.")
            return False

        try:
            final_hta_dict_to_save = tree.to_dict()
        except Exception as e:
            logger.exception(f"Failed to serialize HTATree object to dictionary: {e}")
            self.last_error = str(e)
            self.last_operation_success = False
            increment_hta_failure(f"save_tree error: {e}")
            return False

        if not final_hta_dict_to_save or not final_hta_dict_to_save.get('root'):
             logger.error("Failed to serialize HTA tree or root node is missing in dict.")
             self.last_error = "Failed to serialize HTA tree or root node is missing in dict."
             self.last_operation_success = False
             increment_hta_failure("save_tree error: Failed to serialize HTA tree or root node is missing in dict.")
             return False

        # 1. Update Snapshot Core State (Primary target)
        try:
            if not hasattr(snapshot, 'core_state') or not isinstance(snapshot.core_state, dict):
                snapshot.core_state = {}
            snapshot.core_state['hta_tree'] = final_hta_dict_to_save
            logger.info("Updated HTA tree in snapshot core_state.")
            snapshot_save_ok = True
        except Exception as e:
            logger.exception(f"Failed to update HTA tree in snapshot core_state: {e}")
            self.last_error = str(e)
            self.last_operation_success = False
            increment_hta_failure(f"save_tree error: {e}")
            snapshot_save_ok = False # Still try to save to seed

        # 2. Update Active Seed (Secondary, Source of Truth)
        seed_save_ok = False
        active_seed = await self._get_active_seed(snapshot)
        if active_seed and hasattr(self.seed_manager, 'update_seed'):
            try:
                # Assume update_seed is async if it interacts with DB
                success = await self.seed_manager.update_seed(
                    active_seed.seed_id,
                    hta_tree=final_hta_dict_to_save
                )
                if success:
                    logger.info(f"Successfully updated HTA tree in active seed ID: {active_seed.seed_id}")
                    seed_save_ok = True
                else:
                    logger.error(f"SeedManager failed to update HTA tree for seed ID: {active_seed.seed_id}")
                    self.last_error = f"SeedManager failed to update HTA tree for seed ID: {active_seed.seed_id}"
                    self.last_operation_success = False
                    increment_hta_failure(f"SeedManager failed to update HTA tree for seed ID: {active_seed.seed_id}")
            except Exception as seed_update_err:
                logger.exception(f"Failed to update seed {active_seed.seed_id} with final HTA: {seed_update_err}")
                self.last_error = str(seed_update_err)
                self.last_operation_success = False
                increment_hta_failure(f"save_tree error: {seed_update_err}")
        elif not active_seed:
            logger.error("Cannot save HTA to seed: Active seed not found.")
            self.last_error = "Active seed not found."
            self.last_operation_success = False
            increment_hta_failure("save_tree error: Cannot save HTA to seed: Active seed not found.")
        else: # SeedManager missing method
             logger.error("Cannot save HTA to seed: Injected SeedManager lacks update_seed method.")
             self.last_error = "Injected SeedManager lacks update_seed method."
             self.last_operation_success = False
             increment_hta_failure("save_tree error: Cannot save HTA to seed: Injected SeedManager lacks update_seed method.")

        # Return overall success (prioritize snapshot save)
        return snapshot_save_ok


    async def update_node_status(self, tree: HTATree, node_id: str, new_status: str) -> bool:
        """
        Updates the status of a specific node within the tree object and triggers propagation.
        Note: This modifies the tree object in place. Saving must be done separately.

        Args:
            tree: The HTATree object to modify.
            node_id: The ID of the node to update.
            new_status: The new status string (e.g., "completed", "pending").

        Returns:
            True if the node was found and status potentially updated, False otherwise.
        """
        global HTA_TREE_TOTAL_OPS
        HTA_TREE_TOTAL_OPS += 1
        self.last_error = None
        self.last_operation_success = True
        if not tree or not tree.root:
            logger.error("Cannot update node status: HTATree object is invalid.")
            self.last_error = "HTATree object is invalid."
            self.last_operation_success = False
            increment_hta_failure("update_node_status error: HTATree object is invalid.")
            return False

        node = tree.find_node_by_id(node_id)
        if node:
            logger.info(f"Updating status for node '{node.title}' ({node_id}) to '{new_status}'.")
            tree.update_node_status(node_id, new_status) # This handles propagation
            return True
        else:
            logger.warning(f"Cannot update status: Node with id '{node_id}' not found in tree.")
            self.last_error = f"Node with id '{node_id}' not found in tree."
            self.last_operation_success = False
            increment_hta_failure(f"update_node_status error: Node with id '{node_id}' not found in tree.")
            return False


    async def evolve_tree(self, tree: HTATree, reflections: List[str], snapshot: Optional[MemorySnapshot] = None, goal: str = None, context: str = None, user_id: int = None) -> Optional[HTATree]:
        """
        Handles the HTA evolution process using the LLM client.

        Args:
            tree: The current HTATree object.
            reflections: List of user reflections from the completed batch.
            snapshot: The MemorySnapshot object (optional, for homeostatic recovery).
            goal: The evolution goal (optional, for homeostatic recovery).
            context: The context for onboarding HTA generation (optional, for homeostatic recovery).
            user_id: The user ID for homeostatic recovery (optional, for homeostatic recovery).

        Returns:
            A new HTATree object with the evolved structure if successful and valid,
            otherwise None.
        """
        global HTA_TREE_TOTAL_OPS
        HTA_TREE_TOTAL_OPS += 1
        self.last_error = None
        self.last_operation_success = True
        if not tree or not tree.root:
            logger.error("Cannot evolve HTA: Initial tree is invalid.")
            self.last_error = "Initial tree is invalid."
            self.last_operation_success = False
            increment_hta_failure("evolve_tree error: Initial tree is invalid.")
            return None
        if self.llm_client is None:
            logger.error("Cannot evolve HTA: LLMClient is None.")
            self.last_error = "LLMClient is None."
            self.last_operation_success = False
            increment_hta_failure("evolve_tree error: LLMClient is None.")
            return None

        evolution_goal = "Previous task batch complete. Re-evaluate the plan and suggest next steps." # Default goal

        # 1. (Optional) Distill reflections
        if reflections and hasattr(self.llm_client, 'distill_reflections'):
            logger.info(f"Distilling {len(reflections)} reflections for evolution goal...")
            try:
                distilled_response: Optional[DistilledReflectionResponse] = await self.llm_client.distill_reflections(
                    reflections=reflections,
                    use_advanced_model=True  # Use advanced model for reflection distillation
                )
                if distilled_response and distilled_response.distilled_text:
                    evolution_goal = distilled_response.distilled_text
                    logger.info(f"Using distilled reflection as evolution goal: '{evolution_goal[:100]}...'")
                else:
                    logger.warning("Reflection distillation failed or returned empty text. Using default goal.")
            except Exception as distill_err:
                logger.exception(f"Error during reflection distillation: {distill_err}. Using default goal.")
                self.last_error = str(distill_err)
                self.last_operation_success = False
                increment_hta_failure(f"evolve_tree error: {distill_err}")
                return None
        elif not reflections:
            logger.info("No reflections provided for evolution. Using default goal.")

        # 2. Call LLM for evolution
        try:
            current_hta_json = json.dumps(tree.to_dict()) # Serialize current tree
            logger.debug(f"Calling request_hta_evolution. Goal: '{evolution_goal[:100]}...'")

            evolved_hta_response: Optional[HTAEvolveResponse] = await self.llm_client.request_hta_evolution(
                current_hta_json=current_hta_json,
                evolution_goal=evolution_goal,
                use_advanced_model=True  # Use advanced model for HTA evolution
            )

            # 3. Validate and Process LLM Response
            if not isinstance(evolved_hta_response, HTAEvolveResponse) or not evolved_hta_response.hta_root:
                 log_response = evolved_hta_response if len(str(evolved_hta_response)) < 500 else str(type(evolved_hta_response))
                 logger.error(f"Failed to get valid evolved HTA from LLM. Type: {type(evolved_hta_response)}. Response: {log_response}")
                 self.last_error = f"Failed to get valid evolved HTA from LLM. Type: {type(evolved_hta_response)}. Response: {log_response}"
                 self.last_operation_success = False
                 increment_hta_failure(f"evolve_tree error: {log_response}")
                 raise Exception("LLM evolution failed")

            # Check root ID match (critical!)
            llm_root_id = getattr(evolved_hta_response.hta_root, 'id', 'LLM_MISSING_ID')
            original_root_id = getattr(tree.root, 'id', 'ORIGINAL_MISSING_ID')
            if llm_root_id != original_root_id:
                 logger.error(f"LLM HTA evolution root ID mismatch ('{llm_root_id}' vs '{original_root_id}'). Discarding evolved tree.")
                 self.last_error = f"LLM HTA evolution root ID mismatch ('{llm_root_id}' vs '{original_root_id}'). Discarding evolved tree."
                 self.last_operation_success = False
                 increment_hta_failure(f"evolve_tree error: LLM HTA evolution root ID mismatch ('{llm_root_id}' vs '{original_root_id}'). Discarding evolved tree.")
                 raise Exception("LLM evolution root ID mismatch")

            # Convert Pydantic model back to dictionary for HTATree parsing
            evolved_hta_root_dict = evolved_hta_response.hta_root.model_dump(mode='json')
            evolved_hta_dict = {'root': evolved_hta_root_dict}

            # Attempt to parse the evolved dictionary back into an HTATree object
            try:
                new_tree = HTATree.from_dict(evolved_hta_dict)
                if new_tree and new_tree.root:
                    logger.info("Successfully received and parsed evolved HTA tree.")
                    return new_tree # Return the new, evolved tree object
                else:
                    logger.error("Failed to re-parse evolved HTA dictionary into valid HTATree object.")
                    self.last_error = "Failed to re-parse evolved HTA dictionary into valid HTATree object."
                    self.last_operation_success = False
                    increment_hta_failure("evolve_tree error: Failed to re-parse evolved HTA dictionary into valid HTATree object.")
                    raise Exception("Failed to re-parse evolved HTA dictionary")
            except ValueError as ve:
                 logger.error(f"ValueError parsing evolved HTA dictionary: {ve}")
                 self.last_error = str(ve)
                 self.last_operation_success = False
                 increment_hta_failure(f"evolve_tree error: {ve}")
                 raise
            except Exception as parse_err:
                 logger.exception(f"Unexpected error parsing evolved HTA dictionary: {parse_err}")
                 self.last_error = str(parse_err)
                 self.last_operation_success = False
                 increment_hta_failure(f"evolve_tree error: {parse_err}")
                 raise

        except (LLMError, LLMValidationError) as llm_evolve_err:
            logger.error(f"LLM/Validation Error during HTA evolution request: {llm_evolve_err}")
            self.last_error = str(llm_evolve_err)
            self.last_operation_success = False
            increment_hta_failure(f"evolve_tree error: {llm_evolve_err}")
            # --- Homeostatic Recovery: Schedule background retry ---
            if snapshot is not None and goal is not None and context is not None and user_id is not None:
                async def background_recover_evolve():
                    await asyncio.sleep(30)
                    try:
                        logger.info(f"[Homeostasis] Retrying HTA evolution for user {user_id} in background after fallback.")
                        # Regenerate onboarding HTA as a new base, then evolve again
                        hta_model_dict, _ = await self.generate_onboarding_hta(goal, context, user_id, snapshot)
                        if not hta_model_dict.get('_fallback_used'):
                            base_tree = HTATree.from_dict(hta_model_dict)
                            evolved_tree = await self.evolve_tree(base_tree, reflections, snapshot, goal, context, user_id)
                            if evolved_tree:
                                await self.save_tree(snapshot, evolved_tree)
                                logger.info(f"[Homeostasis] Successfully recovered and replaced fallback evolved HTA for user {user_id}.")
                            else:
                                logger.warning(f"[Homeostasis] Retry still resulted in fallback for user {user_id} (evolution step).")
                        else:
                            logger.warning(f"[Homeostasis] Retry still resulted in fallback for user {user_id} (onboarding step).")
                    except Exception as e:
                        logger.error(f"[Homeostasis] Background recovery for evolution failed for user {user_id}: {e}")
                asyncio.create_task(background_recover_evolve())
            return None # Evolution failed
        except Exception as evolve_err:
            logger.exception(f"Unexpected error during HTA evolution process: {evolve_err}")
            self.last_error = str(evolve_err)
            self.last_operation_success = False
            increment_hta_failure(f"evolve_tree error: {evolve_err}")
            # --- Homeostatic Recovery: Schedule background retry ---
            if snapshot is not None and goal is not None and context is not None and user_id is not None:
                async def background_recover_evolve():
                    await asyncio.sleep(30)
                    try:
                        logger.info(f"[Homeostasis] Retrying HTA evolution for user {user_id} in background after fallback.")
                        hta_model_dict, _ = await self.generate_onboarding_hta(goal, context, user_id, snapshot)
                        if not hta_model_dict.get('_fallback_used'):
                            base_tree = HTATree.from_dict(hta_model_dict)
                            evolved_tree = await self.evolve_tree(base_tree, reflections, snapshot, goal, context, user_id)
                            if evolved_tree:
                                await self.save_tree(snapshot, evolved_tree)
                                logger.info(f"[Homeostasis] Successfully recovered and replaced fallback evolved HTA for user {user_id}.")
                            else:
                                logger.warning(f"[Homeostasis] Retry still resulted in fallback for user {user_id} (evolution step).")
                        else:
                            logger.warning(f"[Homeostasis] Retry still resulted in fallback for user {user_id} (onboarding step).")
                    except Exception as e:
                        logger.error(f"[Homeostasis] Background recovery for evolution failed for user {user_id}: {e}")
                asyncio.create_task(background_recover_evolve())
            return None # Evolution failed

    async def generate_onboarding_hta(self, goal: str, context: str, user_id: int, snapshot: Optional[MemorySnapshot] = None) -> (dict, str):
        """
        Generate an HTA for onboarding using the LLM client, with fallback logic.
        Returns (hta_model_dict, seed_desc).
        """
        import uuid
        from datetime import datetime, timezone
        from forest_app.hta_tree.seed import Seed
        from forest_app.config import constants
        logger = logging.getLogger(__name__)
        root_node_id = f"root_{str(uuid.uuid4())[:8]}"
        hta_prompt = (
            f"[INST] Create a Hierarchical Task Analysis (HTA) representing a plan based on the user's goal and context.\n"
            f"Goal: {goal}\nContext: {context}\n\n"
            f"**RESPONSE FORMAT REQUIREMENTS:**\n"
            f"1. Respond ONLY with a single, valid JSON object.\n"
            f"2. This JSON object MUST contain ONLY ONE top-level key: 'hta_root'.\n"
            f"3. The value of 'hta_root' MUST be a JSON object representing the root node.\n"
            f"4. The entire response MUST strictly adhere to the `HTAResponseModel` and `HTANodeModel` structure.\n\n"
            f"**NODE ATTRIBUTE REQUIREMENTS (Apply recursively):**\n"
            f"* `id`: (String) Unique ID. Root MUST use: '{root_node_id}'.\n"
            f"* `title`: (String) Concise title.\n"
            f"* `description`: (String) Detailed description.\n"
            f"* `priority`: (Float) Value STRICTLY between 0.0 and 1.0 (inclusive).\n"
            f"* `estimated_energy`: (String) MUST be one of: \"low\", \"medium\", or \"high\".\n"
            f"* `estimated_time`: (String) MUST be one of: \"low\", \"medium\", or \"high\".\n"
            f"* `depends_on`: (List[String]) List of node IDs. Empty list `[]` if none.\n"
            f"* `children`: (List[Object]) List of child node objects. Empty list `[]` for leaf nodes.\n"
            f"* `linked_tasks`: (List[String]) Optional. Default `[]`.\n"
            f"* `is_milestone`: (Boolean) Default `false`.\n"
            f"* `rationale`: (String) Brief explanation.\n"
            f"* `status_suggestion`: (String) Initial status, e.g., \"pending\".\n\n"
            f"Ensure the generated JSON is well-formed and strictly follows all requirements. Output ONLY the JSON object starting with `{{ \"hta_root\": {{ ... }} }}`."
            f"[/INST]"
        )
        hta_model_dict = None
        seed_desc = None
        try:
            hta_response_obj = await self.llm_client.generate(
                prompt_parts=[hta_prompt],
                response_model=HTAValidationModel,
                use_advanced_model=True
            )
            if not isinstance(hta_response_obj, HTAValidationModel) or not hta_response_obj.hta_root:
                logger.error(f"LLMClient response unexpected/missing hta_root user {user_id}. Type: {type(hta_response_obj)}. Resp: {str(hta_response_obj)[:500]}")
                increment_hta_failure(f"generate_onboarding_hta error: LLMClient response unexpected/missing hta_root user {user_id}. Type: {type(hta_response_obj)}. Resp: {str(hta_response_obj)[:500]}")
                raise Exception("AI response format unexpected or missing hta_root.")
            hta_root_from_llm = hta_response_obj.hta_root
            if not hasattr(hta_root_from_llm, 'id') or hta_root_from_llm.id != root_node_id:
                logger.error(f"Generated HTA root ID mismatch user {user_id}. Expected: {root_node_id}. Got: {str(hta_response_obj.model_dump_json(indent=2))[:500]}")
                increment_hta_failure(f"generate_onboarding_hta error: Generated HTA root ID mismatch user {user_id}. Expected: {root_node_id}. Got: {str(hta_response_obj.model_dump_json(indent=2))[:500]}")
                raise Exception("AI failed to generate valid plan structure with correct root ID.")
            hta_model_dict = hta_response_obj.model_dump(mode='json')
            logger.info(f"Successfully generated HTA via LLMClient for user {user_id}.")
            fallback_used = False
            fallback_reason = None
        except Exception as hta_gen_err:
            logger.warning(f"CRITICAL: LLMClient HTA generation failed user {user_id}. Error: {hta_gen_err}. Generating fallback.", exc_info=True)
            increment_hta_failure(f"generate_onboarding_hta error: {hta_gen_err}")
            fallback_hta_root = {"id": root_node_id, "title": str(goal or "Primary Goal")[:150], "description": f"Initial plan for: {str(goal or 'Primary Goal')}. (Details pending)", "priority": 0.5, "depends_on": [], "estimated_energy": "medium", "estimated_time": "medium", "linked_tasks": [], "is_milestone": True, "rationale": f"Fallback HTA generated due to LLM failure: {hta_gen_err}", "status_suggestion": "pending", "children": []}
            hta_model_dict = {"hta_root": fallback_hta_root}
            logger.info(f"Using fallback HTA structure for user {user_id}.")
            fallback_used = True
            fallback_reason = str(hta_gen_err)
            # --- Homeostatic Recovery: Schedule background retry ---
            if snapshot is not None:
                async def background_recover_hta():
                    await asyncio.sleep(30)  # Wait before retrying (could use exponential backoff)
                    try:
                        logger.info(f"[Homeostasis] Retrying HTA LLM generation for user {user_id} in background after fallback.")
                        recovered_hta_model_dict, _ = await self.generate_onboarding_hta(goal, context, user_id, snapshot)
                        if not recovered_hta_model_dict.get('_fallback_used'):
                            tree = HTATree.from_dict(recovered_hta_model_dict)
                            await self.save_tree(snapshot, tree)
                            logger.info(f"[Homeostasis] Successfully recovered and replaced fallback HTA for user {user_id}.")
                        else:
                            logger.warning(f"[Homeostasis] Retry still resulted in fallback for user {user_id}.")
                    except Exception as e:
                        logger.error(f"[Homeostasis] Background recovery failed for user {user_id}: {e}")
                asyncio.create_task(background_recover_hta())
        # Compose seed description
        root_node_data = hta_model_dict.get("hta_root")
        seed_desc = str(root_node_data.get("description", f"Overall goal: {root_node_data.get('title', goal or 'Primary Goal')}"))
        # Attach fallback info for downstream use
        hta_model_dict["_fallback_used"] = fallback_used
        hta_model_dict["_fallback_reason"] = fallback_reason
        return hta_model_dict, seed_desc

# Set up rotating error log handler (max 1MB per file, keep 3 backups)
rotating_handler = RotatingFileHandler('error.log', maxBytes=1_000_000, backupCount=3)
rotating_handler.setLevel(logging.WARNING)
rotating_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    logger.addHandler(rotating_handler)

# forest_app/modules/task_engine.py
# =============================================================================
# Task Engine - Selects the next GRANULAR task(s) based on HTA, context, and patterns
# MODIFIED: Ensures priority and magnitude always have default float values.
#           Implements batch size limit (max 5) based on priority (desc)
#           and magnitude (desc) as a secondary sort key.
# =============================================================================

import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

# --- Feature Flag Import ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    # Fallback if flags cannot be imported - assume features are off
    def is_enabled(feature): return False
    class Feature:
        CORE_HTA = "FEATURE_ENABLE_CORE_HTA"
        PATTERN_ID = "FEATURE_ENABLE_PATTERN_ID"
        TASK_RESOURCE_FILTER = "FEATURE_ENABLE_TASK_RESOURCE_FILTER" # For resource checking

# --- Module Imports ---
# Assume HTANode has attributes like id, title, description, children, priority, magnitude, etc.
from forest_app.hta_tree.hta_tree import HTATree, HTANode # For type hinting and tree operations
from forest_app.modules.cognitive.pattern_id import PatternIdentificationEngine # For scoring

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_FALLBACK_TASK_MAGNITUDE = 3.0
DEFAULT_TASK_MAGNITUDE = 5.0 # Default if HTA node lacks magnitude
DEFAULT_TASK_PRIORITY = 0.5 # Default if HTA node lacks priority
MAX_FRONTIER_BATCH_SIZE = 5
# Scoring weights might be less critical now if we select based on depth, but kept for potential future use
BASE_PRIORITY_WEIGHT = 1.0
PATTERN_SCORE_WEIGHT = 0.5
CAPACITY_WEIGHT = 0.2
WITHERING_WEIGHT = -0.3

# --- Helper Functions ---
# [_calculate_node_score remains unchanged]
def _calculate_node_score(
    node: HTANode,
    snapshot: Dict[str, Any],
    pattern_score: float = 0.0,
) -> float:
    """Calculates a weighted score for an HTA node."""
    try:
        # Use the constant default priority here as well
        base_priority = float(getattr(node, 'priority', DEFAULT_TASK_PRIORITY))
    except (ValueError, TypeError):
        logger.warning(f"Could not convert priority '{getattr(node, 'priority', None)}' to float for node {getattr(node, 'id', 'N/A')}. Defaulting to {DEFAULT_TASK_PRIORITY}.")
        base_priority = DEFAULT_TASK_PRIORITY

    capacity = snapshot.get('capacity', 0.5)
    withering = snapshot.get('withering_level', 0.0)

    score = (
        (BASE_PRIORITY_WEIGHT * base_priority) +
        (PATTERN_SCORE_WEIGHT * pattern_score) +
        (CAPACITY_WEIGHT * capacity * base_priority) +
        (WITHERING_WEIGHT * withering * (1 - base_priority))
    )
    return max(0, score)

# Add at the top of the file
FALLBACK_TASK_COUNT = 0

def increment_fallback_count(reason: str = "unknown"):
    global FALLBACK_TASK_COUNT
    FALLBACK_TASK_COUNT += 1
    logger.warning(f"Fallback triggered in TaskEngine due to: {reason} | Fallback count: {FALLBACK_TASK_COUNT}")

class TaskEngine:
    """
    Selects the next set of granular, actionable task(s) based on HTA structure
    and status, potentially influenced by pattern matching and resource availability.
    Limits the output to a defined batch size based on priority (desc) and
    magnitude (desc). Ensures generated tasks always have valid priority/magnitude.
    """
    # [__init__ method remains unchanged]
    def __init__(
        self,
        pattern_engine: PatternIdentificationEngine,
        task_templates: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the TaskEngine.

        Args:
            pattern_engine: An instance of PatternIdentificationEngine.
            task_templates: Optional dictionary containing task templates.
        """
        self.is_real_pattern_engine = isinstance(pattern_engine, PatternIdentificationEngine)
        if not self.is_real_pattern_engine:
             if type(pattern_engine).__name__ == 'DummyService':
                 logger.warning("TaskEngine received a dummy PatternIdentificationEngine. Pattern scoring will be skipped.")
             else:
                 logger.error(f"TaskEngine received an invalid type for pattern_engine: {type(pattern_engine).__name__}. Pattern scoring disabled.")
        self.pattern_engine = pattern_engine

        self.task_templates = task_templates if task_templates else self._load_default_templates()
        logger.debug(f"TaskEngine initialized with injected {'real PatternIdentificationEngine' if self.is_real_pattern_engine else type(pattern_engine).__name__}.")

    # [_load_default_templates method remains unchanged]
    def _load_default_templates(self) -> Dict[str, Any]:
        """Loads default fallback task templates."""
        return {
            "default_reflection": {
                "id_prefix": "reflect",
                "tier": "Bud",
                "title": "Deep Reflection Session: Uncovering Insights",
                "description": "A guided session to explore your recent progress and current state.",
                "magnitude": DEFAULT_FALLBACK_TASK_MAGNITUDE,
                "metadata": {"fallback": True},
                "introspective_prompt": "What feels most alive or challenging in your journey right now?"
            }
        }

    # [_get_fallback_task method remains unchanged]
    def _get_fallback_task(self, template_key: str = "default_reflection") -> Dict[str, Any]:
        """Generates a fallback task using a template."""
        template = self.task_templates.get(template_key)
        if not template:
            increment_fallback_count("template_missing")
            return {
                "id": f"fallback_{uuid.uuid4().hex[:8]}",
                "tier": "Bud",
                "title": "Review Progress",
                "description": "Take a moment to review your current situation.",
                "magnitude": DEFAULT_FALLBACK_TASK_MAGNITUDE,
                "metadata": {"fallback": True, "error": "Template missing"},
                "hta_node_id": None,
            }
        task = template.copy()
        task["id"] = f"{template.get('id_prefix', 'task')}_{uuid.uuid4().hex[:8]}"
        task["created_at"] = datetime.now(timezone.utc).isoformat()
        task.pop("id_prefix", None)
        task["hta_node_id"] = None
        increment_fallback_count("normal_fallback")
        return task

    # [_check_dependencies method remains unchanged]
    def _check_dependencies(self, node: HTANode, tree: HTATree) -> bool:
        """
        Checks if all dependencies for a node are met.
        
        Args:
            node: The node to check dependencies for
            tree: The HTA tree containing all nodes
            
        Returns:
            bool: True if all dependencies are met, False otherwise
        """
        try:
            # Validate inputs
            if not isinstance(node, HTANode):
                logger.error(f"Invalid node type: {type(node)}")
                return False
                
            if not isinstance(tree, HTATree):
                logger.error(f"Invalid tree type: {type(tree)}")
                return False
                
            # Check if node has dependencies
            if not hasattr(node, 'depends_on') or not node.depends_on:
                return True
                
            # Get node map
            try:
                node_map = tree.get_node_map()
            except Exception as e:
                logger.error(f"Error getting node map: {e}")
                return False
                
            # Check each dependency
            for dep_id in node.depends_on:
                if not isinstance(dep_id, str):
                    logger.warning(f"Invalid dependency ID type: {type(dep_id)}")
                    continue
                    
                dep_node = node_map.get(dep_id)
                if not dep_node:
                    logger.warning(f"Dependency node ID '{dep_id}' not found in tree map for node '{getattr(node, 'id', 'N/A')}'. Assuming dependency not met.")
                    return False
                    
                dep_status = getattr(dep_node, 'status', 'pending')
                if not isinstance(dep_status, str):
                    logger.warning(f"Invalid dependency status type: {type(dep_status)}")
                    return False
                    
                if dep_status.lower() != "completed":
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False

    # [_check_resources method remains unchanged]
    def _check_resources(self, node: HTANode, snapshot: Dict[str, Any]) -> bool:
        """Checks resource requirements (if flag enabled)."""
        if not is_enabled(Feature.TASK_RESOURCE_FILTER):
            return True
        required_energy = getattr(node, 'estimated_energy', 'low').lower()
        capacity = snapshot.get('capacity', 0.5)
        energy_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        passes_energy = capacity >= energy_map.get(required_energy, 0.0)
        if not passes_energy:
            logger.debug(f"-> Node {getattr(node, 'id', 'N/A')} rejected: Insufficient energy (requires {required_energy}, capacity {capacity:.2f}).")
            return False
        return True

    # [_filter_candidate_nodes method remains unchanged]
    def _filter_candidate_nodes(self, flat_nodes: List[HTANode], tree: HTATree, snapshot: Dict[str, Any]) -> List[HTANode]:
        """Filters flattened HTA nodes to find viable candidates."""
        candidates = []
        logger.debug(f"Filtering {len(flat_nodes)} flattened nodes...")
        for node in flat_nodes:
            node_id = getattr(node, 'id', 'N/A')
            status = getattr(node, 'status', 'pending')
            if status not in ['pending', 'suggested']:
                continue
            if not self._check_dependencies(node, tree):
                continue
            if not self._check_resources(node, snapshot):
                continue
            candidates.append(node)
        logger.info(f"Found {len(candidates)} candidate HTA nodes after filtering.")
        return candidates

    # --- Method called by Orchestrator ---
    # [get_next_step method remains unchanged]
    def get_next_step(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines the next best step(s) based on the snapshot, prioritizing
        the most granular actionable HTA nodes (the "frontier") up to a
        defined batch size, sorted by priority (desc) then magnitude (desc).

        Args:
            snapshot: The current MemorySnapshot dictionary.

        Returns:
            A dictionary representing the next task bundle. Contains 'tasks' (a list
            of task dicts up to MAX_FRONTIER_BATCH_SIZE) if HTA tasks are found,
            otherwise contains a single 'fallback_task'.
        """
        tasks_list: List[Dict[str, Any]] = []
        fallback_task: Optional[Dict[str, Any]] = None
        hta_tree_obj: Optional[HTATree] = None

        # --- 1. Attempt HTA-based Task Selection (if Core HTA enabled) ---
        if is_enabled(Feature.CORE_HTA):
            logger.debug("Attempting HTA-based task selection (CORE_HTA enabled).")
            try:
                hta_data = snapshot.get("core_state", {}).get("hta_tree")
                if not hta_data or not isinstance(hta_data, dict) or "root" not in hta_data:
                    logger.warning("No valid HTA tree found in snapshot core_state.")
                else:
                    hta_tree_obj = HTATree.from_dict(hta_data)
                    if not hta_tree_obj.root:
                        logger.error("Failed to load HTA tree root from data.")
                        hta_tree_obj = None
                    elif hasattr(hta_tree_obj, 'flatten_tree'):
                        logger.info(f"Loaded HTA Tree with root: {hta_tree_obj.root.id} - '{hta_tree_obj.root.title}'")
                        flat_nodes = hta_tree_obj.flatten_tree()
                        candidate_nodes = self._filter_candidate_nodes(flat_nodes, hta_tree_obj, snapshot)

                        # --- Filter out the root/final node from candidates ---
                        root_id = getattr(hta_tree_obj.root, 'id', None) if hta_tree_obj and hta_tree_obj.root else None
                        root_title = getattr(hta_tree_obj.root, 'title', None) if hta_tree_obj and hta_tree_obj.root else None
                        root_children = getattr(hta_tree_obj.root, 'children', []) if hta_tree_obj and hta_tree_obj.root else []

                        # --- NEW: Personalized first batch based on user context ---
                        user_context = None
                        # Try to extract context from snapshot (common locations)
                        if 'activated_state' in snapshot and isinstance(snapshot['activated_state'], dict):
                            user_context = snapshot['activated_state'].get('context')
                        if not user_context:
                            user_context = snapshot.get('context')
                        # If context is a dict, flatten to string for matching
                        if isinstance(user_context, dict):
                            user_context_str = ' '.join(str(v) for v in user_context.values())
                        else:
                            user_context_str = str(user_context) if user_context else ''
                        # Only personalize if context is non-empty
                        personalized_start_nodes = []
                        if user_context_str:
                            for node in flat_nodes:
                                title = getattr(node, 'title', '').lower()
                                desc = getattr(node, 'description', '').lower()
                                tags = [str(t).lower() for t in getattr(node, 'tags', [])] if hasattr(node, 'tags') else []
                                ctx = user_context_str.lower()
                                if ctx in title or ctx in desc or any(ctx in tag for tag in tags):
                                    personalized_start_nodes.append(node)
                        # If we found personalized nodes, use their children as the batch
                        if personalized_start_nodes:
                            candidate_nodes = []
                            for n in personalized_start_nodes:
                                candidate_nodes.extend([child for child in getattr(n, 'children', []) if getattr(child, 'status', 'pending') in ['pending', 'suggested']])
                        # If no personalized match, use root's children if all are pending/suggested
                        elif root_children and all(getattr(child, 'status', 'pending') in ['pending', 'suggested'] for child in root_children):
                            candidate_nodes = [child for child in root_children if getattr(child, 'status', 'pending') or getattr(child, 'status', 'suggested')]
                        else:
                            candidate_nodes = [n for n in candidate_nodes if getattr(n, 'id', None) != root_id and getattr(n, 'title', None) != root_title]

                        if candidate_nodes:
                            # Find Frontier Nodes by Max Depth
                            max_depth = -1
                            nodes_with_depth = []
                            for node in candidate_nodes:
                                node_id = getattr(node, 'id', None)
                                if node_id and hasattr(hta_tree_obj, 'get_node_depth'):
                                    depth = hta_tree_obj.get_node_depth(node_id)
                                    if depth >= 0: # Ensure node was found
                                         nodes_with_depth.append((node, depth))
                                         max_depth = max(max_depth, depth)
                                    else:
                                         logger.warning(f"Could not get depth for candidate node {node_id}. Skipping.")
                                else:
                                     logger.warning(f"Could not get ID or get_node_depth method missing for candidate node {node_id}.")

                            if max_depth >= 0:
                                frontier_nodes_at_depth = [node for node, depth in nodes_with_depth if depth == max_depth]
                                logger.info(f"Identified {len(frontier_nodes_at_depth)} frontier nodes at depth {max_depth}.")

                                # Sort by priority (desc) then magnitude (desc)
                                def get_priority(node: HTANode) -> float:
                                    try: return float(getattr(node, 'priority', DEFAULT_TASK_PRIORITY))
                                    except (ValueError, TypeError): return DEFAULT_TASK_PRIORITY

                                def get_magnitude(node: HTANode) -> float:
                                    try: return float(getattr(node, 'magnitude', DEFAULT_TASK_MAGNITUDE))
                                    except (ValueError, TypeError): return DEFAULT_TASK_MAGNITUDE

                                frontier_nodes_sorted = sorted(
                                    frontier_nodes_at_depth,
                                    key=lambda node: (-get_priority(node), -get_magnitude(node))
                                )
                                logger.debug(f"Frontier nodes sorted by (-priority, -magnitude): {[getattr(n, 'id', 'N/A') for n in frontier_nodes_sorted]}")

                                # Limit to the batch size
                                final_frontier_nodes = frontier_nodes_sorted[:MAX_FRONTIER_BATCH_SIZE]
                                logger.info(f"Selected top {len(final_frontier_nodes)} nodes based on priority/magnitude (Max Batch: {MAX_FRONTIER_BATCH_SIZE}).")

                                # Convert selected frontier nodes to tasks
                                for node in final_frontier_nodes: # Use the limited list
                                    task = self._create_task_from_hta_node(snapshot, node, hta_tree_obj)
                                    tasks_list.append(task)

                                if tasks_list:
                                     logger.info(f"Generated {len(tasks_list)} tasks for the batch.")

                            else:
                                logger.warning("Could not determine max depth or find nodes at max depth.")
                        else:
                            logger.warning("No candidate HTA nodes found after filtering.")
                    else:
                        logger.error("HTATree object loaded, but lacks 'flatten_tree' method.")
                        hta_tree_obj = None

            except Exception as e:
                logger.exception(f"Error during HTA processing in TaskEngine: {e}")
                hta_tree_obj = None
        else:
            logger.debug("Skipping HTA-based task selection (CORE_HTA disabled).")

        # --- 2. Generate Fallback Task if no HTA tasks found ---
        if not tasks_list:
            logger.warning("No HTA tasks generated. Generating fallback.")
            fallback_task = self._get_fallback_task()

        # --- 3. Prepare and Return Bundle ---
        task_bundle = {
            "tasks": tasks_list, # List of HTA tasks (max MAX_FRONTIER_BATCH_SIZE)
            "fallback_task": fallback_task, # Single fallback task (None if HTA tasks exist)
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return task_bundle

    # --- MODIFIED: _create_task_from_hta_node with robust magnitude ---
    def _create_task_from_hta_node(self, snapshot: Dict[str, Any], hta_node: HTANode, tree: Optional[HTATree]) -> Dict[str, Any]:
        """Creates a task dictionary from a single HTA node, ensuring priority and magnitude."""
        task_id = f"hta_{getattr(hta_node, 'id', uuid.uuid4().hex[:8])}"

        # Robust Priority Handling
        try:
             priority_raw = float(getattr(hta_node, 'priority', DEFAULT_TASK_PRIORITY))
        except (ValueError, TypeError):
             logger.warning(f"Could not convert priority '{getattr(hta_node, 'priority', None)}' to float for node {getattr(hta_node, 'id', 'N/A')}. Defaulting to {DEFAULT_TASK_PRIORITY}.")
             priority_raw = DEFAULT_TASK_PRIORITY

        # --- MODIFIED: Robust Magnitude Handling ---
        try:
             magnitude_val = float(getattr(hta_node, 'magnitude', DEFAULT_TASK_MAGNITUDE))
        except (ValueError, TypeError):
             logger.warning(f"Could not convert magnitude '{getattr(hta_node, 'magnitude', None)}' to float for node {getattr(hta_node, 'id', 'N/A')}. Defaulting to {DEFAULT_TASK_MAGNITUDE}.")
             magnitude_val = DEFAULT_TASK_MAGNITUDE
        # --- END MODIFIED ---

        # Depth calculation remains the same
        hta_depth = 0
        if hasattr(hta_node, "depth"): # Check if depth was pre-calculated
             hta_depth = getattr(hta_node, "depth", 0)
        elif tree and hasattr(hta_node, 'id') and hasattr(tree, 'get_node_depth'):
             try:
                  node_id = getattr(hta_node, 'id', None)
                  if node_id:
                       depth_result = tree.get_node_depth(node_id)
                       hta_depth = depth_result if depth_result >= 0 else 0
                  else:
                       logger.warning("Cannot calculate depth: hta_node is missing 'id'.")
             except Exception as depth_err:
                  logger.error(f"Error calculating node depth for {getattr(hta_node, 'id', 'N/A')}: {depth_err}")
                  hta_depth = 0
        else:
             logger.debug(f"Could not calculate depth for node {getattr(hta_node, 'id', 'N/A')}: Tree or method missing.")


        task = {
            "id": task_id,
            "tier": "Node",
            "title": getattr(hta_node, 'title', 'Untitled HTA Task'),
            "description": getattr(hta_node, 'description', 'Execute this step from the plan.'),
            "magnitude": magnitude_val, # Use the validated magnitude
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hta_node_id": getattr(hta_node, 'id', None),
            "metadata": {
                "is_milestone": getattr(hta_node, 'is_milestone', False),
                "priority_raw": priority_raw, # Use the validated priority
                "hta_depth": hta_depth,
            },
             "estimated_time": getattr(hta_node, 'estimated_time', None),
             "estimated_energy": getattr(hta_node, 'estimated_energy', None),
        }
        # Clean up None values for cleaner output
        task = {k: v for k, v in task.items() if v is not None}
        if "metadata" in task:
             task["metadata"] = {k: v for k, v in task.get("metadata", {}).items() if v is not None}
        else:
             task["metadata"] = {}

        return task
    # --- END MODIFIED ---


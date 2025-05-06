# forest_app/modules/hta_tree.py
# MODIFIED: Added robust default value assignment for priority and magnitude in HTANode.from_dict

import logging
from typing import List, Optional, Dict, Any, Set, Tuple # Added Set, Tuple

logger = logging.getLogger(__name__)
# Ensure logger level is set appropriately elsewhere in your logging setup
# logger.setLevel(logging.INFO) # Example: Set level if not configured globally

# --- MODIFIED: Added Default Constants ---
DEFAULT_TASK_MAGNITUDE = 5.0 # Default if HTA node lacks magnitude
DEFAULT_TASK_PRIORITY = 0.5 # Default if HTA node lacks priority
# --- END MODIFIED ---

# Define a simple RESOURCE_MAP for potential future use (e.g., mapping labels to values)
# Currently unused within this specific file.
RESOURCE_MAP = {"low": 0.3, "medium": 0.6, "high": 0.9}

# Add at the top of the file
HTA_TREE_TOTAL_OPS = 0
HTA_TREE_FAILURE_COUNT = 0

def increment_hta_failure(reason: str = "unknown"):
    global HTA_TREE_FAILURE_COUNT
    HTA_TREE_FAILURE_COUNT += 1
    logger.error(f"HTA Tree Failure: {reason} | Failure count: {HTA_TREE_FAILURE_COUNT} | Total ops: {HTA_TREE_TOTAL_OPS}")

# In every public method (HTANode.from_dict, HTATree.from_dict, add_node, remove_node, update_node_status, find_node_by_id, flatten_tree, propagate_status, get_node_depth, etc):
# At the start: global HTA_TREE_TOTAL_OPS
# HTA_TREE_TOTAL_OPS += 1
# In every error/fallback/exception path: increment_hta_failure(reason)

class HTANode:
    """
    Represents a node in a Hierarchical Task Analysis (HTA) tree.

    Attributes:
        id (str): Unique identifier for the node.
        title (str): The title or short description of the HTA step.
        description (str): Longer explanation of what this step involves.
        status (str): Current status, e.g., "pending", "active", "completed", "pruned".
        priority (float): A numerical value indicating importance (e.g., 0.0-1.0).
        magnitude (float): A numerical value indicating impact/size (e.g., 1.0-10.0). # <-- Added magnitude attribute description
        is_milestone (bool): Flag indicating if this node represents a significant milestone.
        depends_on (List[str]): A list of IDs of nodes that must be completed before this one.
        estimated_energy (str): A string (e.g., "low", "medium", "high") representing energy cost.
        estimated_time (str): A string (e.g., "low", "medium", "high") representing time cost.
        children (List[HTANode]): A list of child HTANode objects.
        linked_tasks (List[str]): A list of task IDs linked to this node.
    """

    def __init__(
        self,
        id: str,
        title: str,
        description: str,
        status: str,
        priority: float,
        magnitude: float, # <-- Added magnitude parameter
        is_milestone: bool = False,
        depends_on: Optional[List[str]] = None,
        estimated_energy: str = "medium",
        estimated_time: str = "medium",
        children: Optional[List["HTANode"]] = None,
        linked_tasks: Optional[List[str]] = None,
    ):
        self.id = id
        self.title = title
        self.description = description
        self.status = status
        # Ensure priority is clamped between 0.0 and 1.0
        self.priority = max(0.0, min(1.0, priority))
        # --- MODIFIED: Assign magnitude ---
        self.magnitude = magnitude # Assuming magnitude doesn't need clamping here, adjust if needed
        # --- END MODIFIED ---
        self.is_milestone = is_milestone
        self.depends_on = depends_on if depends_on is not None else []
        self.estimated_energy = estimated_energy
        self.estimated_time = estimated_time
        self.children = children if children is not None else []
        self.linked_tasks: List[str] = linked_tasks if linked_tasks is not None else []

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the node."""
        return (
            f"HTANode(id='{self.id}', title='{self.title}', status='{self.status}', "
            f"priority={self.priority:.2f}, magnitude={self.magnitude:.1f}, " # <-- Added magnitude
            f"milestone={self.is_milestone}, "
            f"children_count={len(self.children)}, deps={len(self.depends_on)})"
        )

    # [link_task, update_status, mark_completed, adjust_priority_by_context,
    #  prune_if_unnecessary, dependencies_met methods remain unchanged]
    def link_task(self, task_id: str):
        """Links a task ID with this node if not already linked."""
        if task_id not in self.linked_tasks:
            self.linked_tasks.append(task_id)
            logger.info("Linked task '%s' to HTA node '%s'.", task_id, self.title)

    def update_status(self, new_status: str):
        """Update the status of this node. Propagation should be handled by HTATree."""
        old_status = self.status
        if old_status != new_status:
            self.status = new_status
            logger.info(
                "HTA node '%s' (id: %s) status changed from '%s' to '%s'.",
                self.title,
                self.id,
                old_status,
                new_status,
            )

    def mark_completed(self):
        """Marks this node as completed."""
        self.update_status("completed")

    def adjust_priority_by_context(self, context: Dict[str, Any]):
        """
        Dynamically adjust node priority based on context (e.g., user capacity).
        Clamps priority between 0.0 and 1.0.
        """
        capacity = context.get("capacity", 0.5)
        old_priority = self.priority
        scaling_factor = 1 + (capacity - 0.5) * 0.5
        current_priority = float(self.priority)
        self.priority = max(0.0, min(1.0, current_priority * scaling_factor))
        if old_priority != self.priority:
            logger.info(
                "Adjusted priority for node '%s' from %.2f to %.2f based on capacity %.2f",
                self.title, old_priority, self.priority, capacity,
            )

    def prune_if_unnecessary(self, condition: bool):
        """Mark this node and its descendants as pruned if the condition is true."""
        if condition and self.status != "pruned":
            self.update_status("pruned")
            for child in self.children:
                child.prune_if_unnecessary(True)

    def dependencies_met(self, node_map: Dict[str, "HTANode"]) -> bool:
        """
        Checks whether all dependencies of this node have been 'completed'.
        Uses a provided map for efficient node lookup.
        """
        if not self.depends_on: return True
        for dep_id in self.depends_on:
            dep_node = node_map.get(dep_id)
            if dep_node is None:
                logger.warning("Dependency check failed for node '%s': Dependency node '%s' not found.", self.title, dep_id)
                return False
            if dep_node.status.lower() != "completed": return False
        return True

    def to_dict(self) -> dict:
        """Serializes the HTANode to a dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "magnitude": self.magnitude, # <-- Added serialization
            "is_milestone": self.is_milestone,
            "depends_on": self.depends_on,
            "estimated_energy": self.estimated_energy,
            "estimated_time": self.estimated_time,
            "children": [child.to_dict() for child in self.children],
            "linked_tasks": self.linked_tasks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HTANode":
        """Deserializes an HTANode from a dictionary, ensuring default priority/magnitude."""
        global HTA_TREE_TOTAL_OPS
        HTA_TREE_TOTAL_OPS += 1
        if not data or "id" not in data or "title" not in data:
            increment_hta_failure(f"Cannot create HTANode: Missing 'id' or 'title'. Data: {data}")
            raise ValueError("Cannot create HTANode: Missing 'id' or 'title'. Data: %s", data)

        node_id = data["id"] # Get ID for logging context

        # --- MODIFIED: Robust Priority Assignment ---
        try:
            priority_val = float(data.get('priority', DEFAULT_TASK_PRIORITY))
        except (ValueError, TypeError):
            increment_hta_failure(f"Invalid priority '{data.get('priority')}' for node {node_id}. Using default {DEFAULT_TASK_PRIORITY}.")
            priority_val = DEFAULT_TASK_PRIORITY
        # Ensure priority is clamped after conversion
        priority_val = max(0.0, min(1.0, priority_val))
        # --- END MODIFIED ---

        # --- MODIFIED: Robust Magnitude Assignment ---
        try:
            magnitude_val = float(data.get('magnitude', DEFAULT_TASK_MAGNITUDE))
        except (ValueError, TypeError):
            increment_hta_failure(f"Invalid magnitude '{data.get('magnitude')}' for node {node_id}. Using default {DEFAULT_TASK_MAGNITUDE}.")
            magnitude_val = DEFAULT_TASK_MAGNITUDE
        # --- END MODIFIED ---

        # Recursively create children
        children_data = data.get("children", [])
        children = [cls.from_dict(child_data) for child_data in children_data if isinstance(child_data, dict)]

        # Create node using validated/defaulted values
        node = cls(
            id=node_id,
            title=data["title"],
            description=data.get("description", ""),
            status=data.get("status", "pending"),
            priority=priority_val, # Use validated/defaulted value
            magnitude=magnitude_val, # Use validated/defaulted value
            is_milestone=bool(data.get("is_milestone", False)),
            depends_on=data.get("depends_on", []),
            estimated_energy=data.get("estimated_energy", "medium"),
            estimated_time=data.get("estimated_time", "medium"),
            children=children,
            linked_tasks=data.get("linked_tasks", [])
        )
        return node

# --- HTATree class ---
class HTATree:
    """
    Represents the entire HTA tree structure, managing nodes and operations.
    """
    # [__init__, rebuild_node_map, get_node_map, set_root remain unchanged]
    def __init__(self, root: Optional[HTANode] = None):
        self.root = root
        self._node_map: Dict[str, HTANode] = {} # Internal map for quick node lookup
        if root:
            self.rebuild_node_map() # Build map initially if root exists

    def rebuild_node_map(self):
        """Rebuilds the internal dictionary mapping node IDs to nodes."""
        # Use the newly added flatten method
        self._node_map = {node.id: node for node in self.flatten()}
        logger.debug("HTA Tree node map rebuilt. Contains %d nodes.", len(self._node_map))

    def get_node_map(self) -> Dict[str, HTANode]:
        """Returns the current node map (builds it if empty)."""
        if not self._node_map and self.root:
            self.rebuild_node_map()
        return self._node_map

    def set_root(self, root_node: HTANode):
        """Sets the root node and rebuilds the node map."""
        self.root = root_node
        self.rebuild_node_map()

    # [update_node_status method remains unchanged]
    def update_node_status(self, node_id: str, new_status: str):
        """Updates a node's status and triggers status propagation."""
        node = self.find_node_by_id(node_id)
        if node:
            old_status = node.status
            node.update_status(new_status)
            # Propagate only if the status change could lead to parent completion
            if old_status != new_status and new_status.lower() in ["completed", "pruned"]:
                logger.info("Status updated for node '%s', triggering propagation check.", node.title)
                self.propagate_status()
        else:
            logger.warning("Cannot update status: Node with id '%s' not found.", node_id)

    # [to_dict method remains unchanged]
    def to_dict(self) -> dict:
        """Serializes the HTATree to a dictionary (only includes the root structure)."""
        return {"root": self.root.to_dict() if self.root else None}

    @classmethod
    def from_dict(cls, data: dict) -> "HTATree":
        """Deserializes the HTATree from a dictionary; expects a 'root' or 'hta_root' key."""
        # --- MODIFIED: Add logging for input data ---
        if not isinstance(data, dict):
             increment_hta_failure(f"Invalid data type passed to HTATree.from_dict: expected dict, got {type(data)}")
             return cls(root=None) # Return empty tree

        logger.debug("HTATree.from_dict called with data keys: %s", list(data.keys()))
        # --- END MODIFIED ---

        root_data = data.get("root")
        if root_data is None:
            root_data = data.get("hta_root")
        root_node = None
        if isinstance(root_data, dict):
            try:
                # HTANode.from_dict now handles defaulting priority/magnitude
                root_node = HTANode.from_dict(root_data)
            except ValueError as e:
                increment_hta_failure(f"Error creating root HTANode from dict: {e}. Data: {root_data}")
                root_node = None # Ensure root is None if creation fails
            except Exception as e:
                 increment_hta_failure(f"Unexpected error creating root HTANode from dict: {e}")
                 root_node = None
        elif root_data is not None:
             increment_hta_failure(f"Data for 'root'/'hta_root' key is not a dictionary: {type(root_data)}")

        return cls(root=root_node) # Node map will be built on first access or explicitly

    # [flatten_tree / flatten methods remain unchanged]
    def flatten_tree(self) -> List[HTANode]:
        """
        Flattens the tree into a list of all HTANode objects using DFS (iterative).
        """
        if not self.root:
            return []
        nodes = []
        stack = [self.root]
        visited = set()
        while stack:
            node = stack.pop()
            if node.id in visited: continue
            visited.add(node.id)
            nodes.append(node)
            # Add children to stack carefully
            if hasattr(node, 'children') and isinstance(node.children, list):
                 for child in reversed(node.children):
                     if isinstance(child, HTANode) and hasattr(child, 'id') and child.id not in visited:
                          stack.append(child)
        return nodes

    def flatten(self) -> List[HTANode]:
         """Alias for flatten_tree for compatibility."""
         return self.flatten_tree()

    # [propagate_status method remains unchanged]
    def propagate_status(self):
        """
        Recursively propagates status upward from the leaves.
        If all children of a node are 'completed' or 'pruned', the node is marked 'completed'.
        Should be called after a node status changes to 'completed' or 'pruned'.
        """
        if not self.root: return
        changed_nodes = set()
        def _propagate(node: HTANode) -> bool:
            if not node.children: return node.status.lower() in ["completed", "pruned"]
            all_children_done = all(_propagate(child) for child in node.children)
            if all_children_done and node.status.lower() not in ["completed", "pruned"]:
                old_status = node.status
                node.status = "completed"
                changed_nodes.add(node.id)
                logger.info("Propagated status: Node '%s' (id: %s) changed from '%s' to 'completed'.", node.title, node.id, old_status)
            return node.status.lower() in ["completed", "pruned"]
        _propagate(self.root)
        if changed_nodes: logger.info("Status propagation finished. Nodes updated: %s", changed_nodes)
        else: logger.debug("Status propagation check finished. No changes.") # Changed to debug

    # [find_node_by_id method remains unchanged]
    def find_node_by_id(self, node_id: str) -> Optional[HTANode]:
        """Searches the tree for a node with the given ID using the node map."""
        current_map = self.get_node_map()
        if current_map: return current_map.get(node_id)
        logger.warning("Node map empty/missing despite root existing, performing tree traversal for find_node_by_id.")
        if not self.root: return None
        queue = [self.root]
        while queue:
            current_node = queue.pop(0)
            if current_node.id == node_id: return current_node
            if hasattr(current_node, 'children'): # Check children exist
                 for child in current_node.children:
                     if isinstance(child, HTANode): # Check child is valid node
                          queue.append(child)
        return None

    # [add_node method remains unchanged]
    def add_node(self, parent_id: str, new_node: HTANode) -> bool:
        """
        Adds a new node as a child to the node with the given parent_id.
        Updates the node map.
        """
        parent = self.find_node_by_id(parent_id)
        if parent:
            if self.find_node_by_id(new_node.id):
                logger.warning("Cannot add node: Node with id '%s' already exists.", new_node.id)
                return False
            parent.children.append(new_node)
            # Add the new node and its potential children to the map
            subtree_nodes = [new_node]
            queue = list(new_node.children)
            while queue:
                 current = queue.pop(0)
                 subtree_nodes.append(current)
                 if hasattr(current, 'children'): queue.extend(current.children)
            for node_to_add in subtree_nodes:
                 if node_to_add.id not in self._node_map: self._node_map[node_to_add.id] = node_to_add
                 else: logger.warning("Node ID %s collision during add_node map update.", node_to_add.id)
            logger.info("Added node '%s' (id: %s) as child of '%s' (id: %s).", new_node.title, new_node.id, parent.title, parent.id)
            return True
        else:
            logger.warning("Cannot add node: Parent node '%s' not found.", parent_id)
            return False

    # [remove_node method remains unchanged]
    def remove_node(self, node_id: str) -> bool:
        """
        Removes the node with the specified ID (and its entire subtree) from the tree.
        Updates the node map.
        """
        if not self.root: logger.warning("Cannot remove node: Tree empty."); return False
        if self.root.id == node_id: logger.warning("Cannot remove root node."); return False
        queue = [self.root]; parent_map: Dict[str, Optional[HTANode]] = {self.root.id: None}
        node_to_remove: Optional[HTANode] = None; parent_node: Optional[HTANode] = None; found = False
        while queue:
            current = queue.pop(0)
            if current.id == node_id: node_to_remove = current; parent_node = parent_map.get(node_id); found = True; break
            if hasattr(current, 'children'):
                for child in current.children:
                    if isinstance(child, HTANode): parent_map[child.id] = current; queue.append(child)
        if not found or parent_node is None: logger.warning("Cannot remove node: Node '%s' not found or parent lookup failed.", node_id); return False
        try:
            parent_node.children.remove(node_to_remove)
            logger.info("Removed node '%s' (id: %s) from parent '%s'.", node_to_remove.title, node_to_remove.id, parent_node.title)
            ids_to_remove = set(); queue = [node_to_remove]
            while queue:
                current = queue.pop(0); ids_to_remove.add(current.id)
                if hasattr(current, 'children'): queue.extend(current.children)
            for removed_id in ids_to_remove: self._node_map.pop(removed_id, None)
            logger.debug("Updated node map after removing subtree at %s.", node_id)
            return True
        except ValueError: logger.error("Error removing node '%s': Not found in parent '%s' children.", node_to_remove.title, parent_node.title); return False

    # [get_node_depth method remains unchanged]
    def get_node_depth(self, node_id: str) -> int:
        """Calculates the depth of a node (root is depth 0)."""
        if not self.root: return -1
        queue = [(self.root, 0)]; visited = {self.root.id}
        while queue:
            current_node, depth = queue.pop(0)
            if current_node.id == node_id: return depth
            if hasattr(current_node, 'children'):
                for child in current_node.children:
                    if isinstance(child, HTANode) and hasattr(child, 'id') and child.id not in visited:
                        visited.add(child.id); queue.append((child, depth + 1))
        logger.warning("Node ID %s not found when calculating depth.", node_id); return -1

#############################################
# End of hta_tree.py
#############################################

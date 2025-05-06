# forest_app/modules/xp_mastery.py

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class XPMasteryEngine:
    """Engine for handling XP and mastery challenges."""

    # Define stage thresholds and types
    STAGES = [
        {"name": "Novice", "min_xp": 0, "max_xp": 100, "type": "Basic"},
        {"name": "Apprentice", "min_xp": 100, "max_xp": 300, "type": "Integration"},
        {"name": "Adept", "min_xp": 300, "max_xp": 600, "type": "Advanced"},
        {"name": "Master", "min_xp": 600, "max_xp": float("inf"), "type": "Expert"}
    ]

    def __init__(self):
        """Initialize the engine."""
        self.current_xp = 0.0
        self.completed_challenges: List[str] = []
        self.current_stage_index = 0

    def calculate_xp_gain(self, task: Dict[str, Any]) -> float:
        """
        Calculate XP gain for a completed task based on multiple factors.
        
        Args:
            task: Dictionary containing task details including priority,
                 complexity, time_spent, and dependencies.
        
        Returns:
            float: Calculated XP gain
        """
        try:
            # Validate task format
            if not isinstance(task, dict):
                raise TypeError("Invalid task format - must be a dictionary")
            
            # Base XP calculation
            base_xp = 10.0
            
            # Priority multiplier
            priority = task.get("priority", "medium")
            if not isinstance(priority, str):
                logger.warning(f"Invalid priority type: {type(priority)}. Using default 'medium'")
                priority = "medium"
                
            priority_multiplier = {
                "low": 0.5,
                "medium": 1.0,
                "high": 1.5,
                "critical": 2.0
            }.get(priority.lower(), 1.0)
            
            # Complexity multiplier
            complexity = task.get("complexity", "simple")
            if not isinstance(complexity, str):
                logger.warning(f"Invalid complexity type: {type(complexity)}. Using default 'simple'")
                complexity = "simple"
                
            complexity_multiplier = {
                "trivial": 0.5,
                "simple": 1.0,
                "moderate": 1.5,
                "complex": 2.0,
                "very_complex": 2.5
            }.get(complexity.lower(), 1.0)
            
            # Time investment bonus (caps at 2x)
            try:
                time_spent = float(task.get("time_spent", 0))
                if time_spent < 0:
                    logger.warning("Negative time_spent value. Using 0.")
                    time_spent = 0
            except (ValueError, TypeError):
                logger.warning(f"Invalid time_spent value: {task.get('time_spent')}. Using 0.")
                time_spent = 0
                
            time_multiplier = min(1 + (time_spent / 60), 2.0)  # Assumes time_spent in minutes
            
            # Dependency bonus (more XP for completing dependent tasks)
            dependencies = task.get("dependencies", [])
            if not isinstance(dependencies, list):
                logger.warning("Invalid dependencies format. Using empty list.")
                dependencies = []
                
            dependency_count = len(dependencies)
            dependency_multiplier = 1 + (0.1 * dependency_count)  # 10% bonus per dependency
            
            # Calculate final XP
            final_xp = (
                base_xp 
                * priority_multiplier 
                * complexity_multiplier 
                * time_multiplier 
                * dependency_multiplier
            )
            
            logger.debug(
                f"XP Calculation: base={base_xp}, priority={priority_multiplier}, "
                f"complexity={complexity_multiplier}, time={time_multiplier}, "
                f"dependencies={dependency_multiplier}, final={final_xp}"
            )
            
            return round(final_xp, 2)
            
        except Exception as e:
            logger.error(f"Error calculating XP gain: {e}")
            return 0.0  # Return 0 XP on any error

    def get_current_stage(self) -> Dict[str, Any]:
        """Get current stage information based on XP."""
        for i, stage in enumerate(self.STAGES):
            if self.current_xp >= stage["min_xp"] and self.current_xp < stage["max_xp"]:
                self.current_stage_index = i
                return stage
        return self.STAGES[0]  # Fallback to first stage

    def validate_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """
        Validate snapshot data structure.
        
        Args:
            snapshot: Dictionary containing snapshot data
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Basic type check
            if not isinstance(snapshot, dict):
                logger.error("Snapshot must be a dictionary")
                return False
                
            # Required keys check
            required_keys = ["user_id", "timestamp", "core_state"]
            if not all(key in snapshot for key in required_keys):
                logger.error(f"Missing required keys in snapshot: {required_keys}")
                return False
                
            # Type checks for required fields
            if not isinstance(snapshot.get("core_state"), dict):
                logger.error("core_state must be a dictionary")
                return False
                
            if not isinstance(snapshot.get("user_id"), str):
                logger.error("user_id must be a string")
                return False
                
            # Timestamp format check
            try:
                datetime.fromisoformat(snapshot.get("timestamp", "").replace("Z", "+00:00"))
            except (ValueError, TypeError, AttributeError):
                logger.error("Invalid timestamp format")
                return False
                
            # XP state validation if present
            core_state = snapshot.get("core_state", {})
            if "xp_mastery" in core_state:
                xp_state = core_state["xp_mastery"]
                if not isinstance(xp_state, dict):
                    logger.error("xp_mastery state must be a dictionary")
                    return False
                    
                try:
                    current_xp = float(xp_state.get("current_xp", 0.0))
                    if current_xp < 0:
                        logger.error("current_xp cannot be negative")
                        return False
                except (ValueError, TypeError):
                    logger.error("Invalid current_xp format")
                    return False
                    
                if not isinstance(xp_state.get("completed_challenges", []), list):
                    logger.error("completed_challenges must be a list")
                    return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating snapshot: {e}")
            return False

    async def generate_challenge_content(
        self,
        stage_info: Dict[str, Any],
        snapshot: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate mastery challenge content based on current stage.
        
        Args:
            stage_info: Dictionary containing stage information
            snapshot: Current memory snapshot
            **kwargs: Additional parameters
            
        Returns:
            Dict containing challenge content
        """
        try:
            # Validate inputs
            if not isinstance(stage_info, dict):
                raise TypeError("Invalid stage_info format")
            if not isinstance(snapshot, dict):
                raise TypeError("Invalid snapshot format")
                
            # Validate snapshot structure
            if not self.validate_snapshot(snapshot):
                logger.warning("Invalid snapshot provided for challenge generation")
                stage_info = self.get_current_stage()  # Fallback to current stage
            
            stage_name = stage_info.get("name", "Current Stage")
            challenge_type = stage_info.get("type", "Integration")
            
            # Generate appropriate challenge based on stage
            challenge_templates = {
                "Basic": "Complete a series of fundamental tasks to demonstrate basic mastery.",
                "Integration": "Combine multiple concepts to solve a complex problem.",
                "Advanced": "Design and implement an advanced solution with multiple components.",
                "Expert": "Create an innovative solution that pushes the boundaries of current capabilities."
            }
            
            challenge_description = challenge_templates.get(
                challenge_type,
                "Complete this challenge to demonstrate mastery."
            )
            
            content = (
                f"Mastery Challenge: {stage_name}\n"
                f"Type: {challenge_type}\n"
                f"Description: {challenge_description}\n"
                f"Current XP: {self.current_xp}"
            )

            return {
                "stage": stage_name,
                "type": challenge_type,
                "content": content,
                "created_at": datetime.now().isoformat(),
                "xp_required": self.STAGES[self.current_stage_index]["max_xp"],
                "current_xp": self.current_xp
            }
        except Exception as e:
            logger.error(f"Error generating challenge content: {e}")
            # Return a safe fallback challenge
            return {
                "stage": "Novice",
                "type": "Basic",
                "content": "Complete this challenge to progress.",
                "created_at": datetime.now().isoformat(),
                "xp_required": self.STAGES[0]["max_xp"],
                "current_xp": self.current_xp,
                "error": str(e)
            }

class XPMastery:
    """Legacy XP mastery class for backward compatibility."""

    # Define stage thresholds and types
    STAGES = [
        {"name": "Novice", "min_xp": 0, "max_xp": 100, "type": "Basic"},
        {"name": "Apprentice", "min_xp": 100, "max_xp": 300, "type": "Integration"},
        {"name": "Adept", "min_xp": 300, "max_xp": 600, "type": "Advanced"},
        {"name": "Master", "min_xp": 600, "max_xp": float("inf"), "type": "Expert"}
    ]

    def __init__(self):
        self.engine = XPMasteryEngine()

    def validate_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """
        Validate snapshot data structure.
        
        Args:
            snapshot: Dictionary containing snapshot data
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Basic type check
            if not isinstance(snapshot, dict):
                logger.error("Snapshot must be a dictionary")
                return False
                
            # Required keys check
            required_keys = ["user_id", "timestamp", "core_state"]
            if not all(key in snapshot for key in required_keys):
                logger.error(f"Missing required keys in snapshot: {required_keys}")
                return False
                
            # Type checks for required fields
            if not isinstance(snapshot.get("core_state"), dict):
                logger.error("core_state must be a dictionary")
                return False
                
            if not isinstance(snapshot.get("user_id"), str):
                logger.error("user_id must be a string")
                return False
                
            # Timestamp format check
            try:
                datetime.fromisoformat(snapshot.get("timestamp", "").replace("Z", "+00:00"))
            except (ValueError, TypeError, AttributeError):
                logger.error("Invalid timestamp format")
                return False
                
            # XP state validation if present
            core_state = snapshot.get("core_state", {})
            if "xp_mastery" in core_state:
                xp_state = core_state["xp_mastery"]
                if not isinstance(xp_state, dict):
                    logger.error("xp_mastery state must be a dictionary")
                    return False
                    
                try:
                    current_xp = float(xp_state.get("current_xp", 0.0))
                    if current_xp < 0:
                        logger.error("current_xp cannot be negative")
                        return False
                except (ValueError, TypeError):
                    logger.error("Invalid current_xp format")
                    return False
                    
                if not isinstance(xp_state.get("completed_challenges", []), list):
                    logger.error("completed_challenges must be a list")
                    return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating snapshot: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "current_xp": self.engine.current_xp,
            "completed_challenges": self.engine.completed_challenges,
            "current_stage": self.engine.get_current_stage()
        }

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update state from dictionary."""
        try:
            if not isinstance(data, dict):
                raise TypeError("Invalid data format")
                
            self.engine.current_xp = float(data.get("current_xp", 0.0))
            self.engine.completed_challenges = data.get("completed_challenges", [])
            if not isinstance(self.engine.completed_challenges, list):
                self.engine.completed_challenges = []
            # Stage will be recalculated based on XP
        except (ValueError, TypeError) as e:
            logger.error(f"Error updating XP Mastery state: {e}")
            self.engine.current_xp = 0.0
            self.engine.completed_challenges = []

    def get_current_stage(self, xp: float = None) -> Dict[str, Any]:
        """Get current stage based on XP."""
        if xp is not None:
            self.engine.current_xp = xp
        return self.engine.get_current_stage()

    def get_stage_for_xp(self, xp: float) -> Dict[str, Any]:
        """Get the stage for a given XP amount."""
        try:
            for stage in self.STAGES:
                if stage["min_xp"] <= xp <= stage.get("max_xp", float("inf")):
                    return stage
            return self.STAGES[0]  # Return first stage as default
        except Exception as e:
            logger.error(f"Error getting stage for XP: {e}")
            return self.STAGES[0]

    def get_next_stage(self, current_stage: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get the next stage after the current stage."""
        try:
            if not isinstance(current_stage, dict):
                raise TypeError("Invalid current_stage format")
                
            current_stage_index = -1
            for i, stage in enumerate(self.STAGES):
                if stage["name"] == current_stage.get("name"):
                    current_stage_index = i
                    break

            if current_stage_index >= 0 and current_stage_index < len(self.STAGES) - 1:
                return self.STAGES[current_stage_index + 1]
            return None
        except Exception as e:
            logger.error(f"Error getting next stage: {e}")
            return None

    def check_xp_stage(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Check current XP stage and return progression info."""
        try:
            # Validate snapshot structure
            if not isinstance(snapshot, dict):
                raise TypeError("Invalid snapshot type")

            # Get XP state
            core_state = snapshot.get("core_state", {})
            if not isinstance(core_state, dict):
                raise TypeError("Invalid core_state format")
                
            xp_state = core_state.get("xp_mastery", {})
            if not isinstance(xp_state, dict):
                raise TypeError("Invalid xp_mastery state format")
                
            try:
                current_xp = float(xp_state.get("current_xp", 0.0))
                if current_xp < 0:
                    logger.warning("Negative XP value found, resetting to 0.0")
                    current_xp = 0.0
            except (ValueError, TypeError):
                logger.warning("Invalid current_xp value, defaulting to 0.0")
                current_xp = 0.0
                
            completed_challenges = xp_state.get("completed_challenges", [])
            if not isinstance(completed_challenges, list):
                logger.warning("Invalid completed_challenges format, using empty list")
                completed_challenges = []

            # Find current stage
            current_stage = None
            next_stage = None
            try:
                for i, stage in enumerate(self.STAGES):
                    if current_xp >= stage["min_xp"] and (i == len(self.STAGES) - 1 or current_xp < self.STAGES[i + 1]["min_xp"]):
                        current_stage = stage
                        if i < len(self.STAGES) - 1:
                            next_stage = self.STAGES[i + 1]
                        break

                if not current_stage:
                    logger.warning("Could not determine current stage, defaulting to first stage")
                    current_stage = self.STAGES[0]
                    next_stage = self.STAGES[1] if len(self.STAGES) > 1 else None
            except Exception as e:
                logger.error(f"Error determining stage: {e}")
                current_stage = self.STAGES[0]
                next_stage = self.STAGES[1] if len(self.STAGES) > 1 else None

            # Calculate XP needed for next stage
            try:
                xp_needed = next_stage["min_xp"] - current_xp if next_stage else 0.0
                xp_needed = max(0.0, xp_needed)  # Ensure non-negative
            except Exception as e:
                logger.error(f"Error calculating XP needed: {e}")
                xp_needed = 0.0

            return {
                "current_stage": current_stage["name"],
                "current_xp": current_xp,
                "next_stage": next_stage["name"] if next_stage else "Max Level",
                "xp_needed": xp_needed,
                "completed_challenges": completed_challenges
            }

        except Exception as e:
            logger.error(f"Error checking XP stage: {e}")
            return {
                "current_stage": self.STAGES[0]["name"],
                "current_xp": 0.0,
                "next_stage": self.STAGES[1]["name"] if len(self.STAGES) > 1 else "Max Level",
                "xp_needed": self.STAGES[1]["min_xp"] if len(self.STAGES) > 1 else 0.0,
                "completed_challenges": [],
                "error": str(e)
            }

    async def generate_challenge_content(
        self,
        stage_info: Dict[str, Any],
        snapshot: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate challenge content using the engine."""
        return await self.engine.generate_challenge_content(stage_info, snapshot, **kwargs)

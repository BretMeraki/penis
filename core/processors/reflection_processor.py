# forest_app/core/processors/reflection_processor.py

import logging
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, cast

# --- Core & Module Imports ---
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.core.utils import clamp01
from forest_app.core.harmonic_framework import SilentScoring, HarmonicRouting
from forest_app.modules.cognitive.sentiment import (
    SecretSauceSentimentEngineHybrid,
    SentimentOutput,
    SentimentInput,
)
from forest_app.modules.cognitive.practical_consequence import PracticalConsequenceEngine
from forest_app.hta_tree.task_engine import TaskEngine
from forest_app.modules.cognitive.narrative_modes import NarrativeModesEngine
from forest_app.core.soft_deadline_manager import schedule_soft_deadlines, Feature as SDFeature
from forest_app.integrations.llm import (
    LLMClient,
    ArbiterStandardResponse,
    LLMError,
    LLMValidationError
)

# --- Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    # Fallback if flags cannot be imported - assume features are off
    def is_enabled(feature): return False
    class Feature:
        SENTIMENT_ANALYSIS = "FEATURE_ENABLE_SENTIMENT_ANALYSIS"
        NARRATIVE_MODES = "FEATURE_ENABLE_NARRATIVE_MODES"
        SOFT_DEADLINES = "FEATURE_ENABLE_SOFT_DEADLINES"  # Use the core feature flag
        ENABLE_POETIC_ARBITER_VOICE = "FEATURE_ENABLE_POETIC_ARBITER_VOICE"
        CORE_TASK_ENGINE = "FEATURE_ENABLE_CORE_TASK_ENGINE"
        PRACTICAL_CONSEQUENCE = "FEATURE_ENABLE_PRACTICAL_CONSEQUENCE"

# --- Constants ---
from forest_app.config.constants import (
    REFLECTION_CAPACITY_NUDGE_BASE,
    REFLECTION_SHADOW_NUDGE_BASE,
    MAGNITUDE_THRESHOLDS,
    DEFAULT_RESONANCE_THEME,
)

# Default task details for fallback scenarios
FALLBACK_TASK_DETAILS = {
    "title": "Reflect on your current focus",
    "description": "Take a moment to consider what feels most important right now.",
    "priority": 5,
    "magnitude": 5,
    "status": "incomplete",
    "soft_deadline": None,
    "parent_id": None,
}

logger = logging.getLogger(__name__)

# Define valid thresholds for task generation
valid_thresholds = {
    "min_confidence": 0.7,
    "max_complexity": 0.8,
    "min_relevance": 0.6
}

# Add at the top of the file
FALLBACK_TASK_COUNT = 0

def increment_fallback_count(reason: str = "unknown"):
    global FALLBACK_TASK_COUNT
    FALLBACK_TASK_COUNT += 1
    logger.warning(f"Fallback triggered in ReflectionProcessor due to: {reason} | Fallback count: {FALLBACK_TASK_COUNT}")

# --- Helper Functions (Could be moved to utils if used elsewhere) ---

def prune_context(snap_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimize prompt size while keeping key information.
    
    Args:
        snap_dict: Dictionary representation of a MemorySnapshot
        
    Returns:
        Dict containing pruned context with only essential information
        
    Note:
        This function might need access to component state if features like
        FINANCIAL_READINESS or DESIRE_ENGINE are enabled and used for context pruning.
        If so, the necessary state might need to be passed in or handled differently.
    """
    try:
        # Extract core metrics with type safety
        ctx = {
            "shadow_score": float(snap_dict.get("shadow_score", 0.5)),
            "capacity": float(snap_dict.get("capacity", 0.5)),
            "magnitude": float(snap_dict.get("magnitude", 5.0)),
        }
        
        # Add optional fields if available
        if is_enabled(Feature.NARRATIVE_MODES):
            last_ritual = snap_dict.get("last_ritual_mode")
            if last_ritual is not None:
                ctx["last_ritual_mode"] = str(last_ritual)
                
        current_path = snap_dict.get("current_path")
        if current_path is not None:
            ctx["current_path"] = str(current_path)
            
        return ctx
    except (TypeError, ValueError) as e:
        logger.error(f"Error pruning context: {e}")
        return {
            "shadow_score": 0.5,
            "capacity": 0.5,
            "magnitude": 5.0
        }
    except Exception as e:
        logger.exception(f"Unexpected error pruning context: {e}")
        return {
            "shadow_score": 0.5,
            "capacity": 0.5,
            "magnitude": 5.0
        }

def describe_magnitude(value: float) -> str:
    """
    Describes magnitude based on predefined thresholds.
    
    Args:
        value: The magnitude value to describe
        
    Returns:
        String description of the magnitude level
        
    Note:
        Uses MAGNITUDE_THRESHOLDS from constants to determine the description.
        Returns "Unknown" if the value cannot be processed or thresholds are invalid.
    """
    try:
        float_value = float(value)
        
        # Validate and convert thresholds
        valid_thresholds: Dict[str, float] = {}
        for k, v in MAGNITUDE_THRESHOLDS.items():
            try:
                if isinstance(v, (int, float)):
                    valid_thresholds[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
                
        if not valid_thresholds:
            logger.error("No valid thresholds found in MAGNITUDE_THRESHOLDS")
            return "Unknown"
            
        # Sort thresholds in descending order
        sorted_thresholds = sorted(
            valid_thresholds.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        
        # Find appropriate threshold
        for label, thresh in sorted_thresholds:
            if float_value >= thresh:
                return str(label)
                
        # Return lowest threshold label if no match
        return str(sorted_thresholds[-1][0]) if sorted_thresholds else "Dormant"
        
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting value/threshold for magnitude: {e} (Value: {value})")
        return "Unknown"
    except Exception as e:
        logger.exception(f"Error describing magnitude for value {value}: {e}")
        return "Unknown"

# --- Reflection Processor Class ---

class ReflectionProcessor:
    """Handles the workflow for processing user reflections."""

    def __init__(
        self,
        llm_client: LLMClient,
        task_engine: TaskEngine,
        # Optional engines (can be DummyService instances if disabled/not injected)
        sentiment_engine: Optional[SecretSauceSentimentEngineHybrid] = None,
        practical_consequence_engine: Optional[PracticalConsequenceEngine] = None,
        narrative_engine: Optional[NarrativeModesEngine] = None,
        silent_scorer: Optional[SilentScoring] = None,
        harmonic_router: Optional[HarmonicRouting] = None,
    ) -> None:
        """Initialize the ReflectionProcessor with required and optional dependencies.
        
        Args:
            llm_client: Language model client for generating responses
            task_engine: Engine for task management and generation
            sentiment_engine: Optional engine for sentiment analysis
            practical_consequence_engine: Optional engine for consequence analysis
            narrative_engine: Optional engine for narrative mode processing
            silent_scorer: Optional engine for silent scoring
            harmonic_router: Optional engine for harmonic routing
        
        Raises:
            TypeError: If llm_client or task_engine are invalid types
        """
        # Store injected dependencies
        self.llm_client = llm_client
        self.task_engine = task_engine
        self.sentiment_engine = sentiment_engine
        self.practical_consequence_engine = practical_consequence_engine
        self.narrative_engine = narrative_engine
        self.silent_scorer = silent_scorer
        self.harmonic_router = harmonic_router

        # Validate critical dependencies
        if not isinstance(self.llm_client, LLMClient) or type(self.llm_client).__name__ == 'DummyService':
            logger.critical("ReflectionProcessor initialized with invalid or dummy LLMClient!")
            raise TypeError("Invalid LLMClient provided to ReflectionProcessor.")
            
        if not isinstance(self.task_engine, TaskEngine) or type(self.task_engine).__name__ == 'DummyService':
            logger.critical("ReflectionProcessor initialized with invalid or dummy TaskEngine!")
            raise TypeError("Invalid TaskEngine provided to ReflectionProcessor.")

        # Log warnings for optional engines if they're dummies but features are enabled
        if is_enabled(Feature.SENTIMENT_ANALYSIS) and (not isinstance(self.sentiment_engine, SecretSauceSentimentEngineHybrid) or type(self.sentiment_engine).__name__ == 'DummyService'):
            logger.warning("ReflectionProcessor: SENTIMENT_ANALYSIS feature enabled but sentiment engine is invalid or dummy.")
            
        if is_enabled(Feature.NARRATIVE_MODES) and (not isinstance(self.narrative_engine, NarrativeModesEngine) or type(self.narrative_engine).__name__ == 'DummyService'):
            logger.warning("ReflectionProcessor: NARRATIVE_MODES feature enabled but narrative engine is invalid or dummy.")

        logger.info("ReflectionProcessor initialized successfully.")


    async def process(self, user_input: str, snapshot: MemorySnapshot) -> Dict[str, Any]:
        """Process user reflection and generate next tasks."""
        if not user_input:
            # Return a fallback task for empty input instead of raising an error
            return {
                "tasks": [self._get_fallback_task("empty_input")],
                "arbiter_response": "(No reflection provided. Please share your thoughts to help guide the process.)",
                "error": None,
                "magnitude_description": "Unknown"
            }

        error_message = None
        magnitude_description = "Unknown"
        generated_tasks = []
        fallback_task = None
        arbiter_response = ""

        try:
            # --- 1. Sentiment Analysis ---
            if is_enabled(Feature.SENTIMENT_ANALYSIS) and self.sentiment_engine:
                try:
                    sentiment_result = await self.sentiment_engine.analyze_emotional_field(
                        SentimentInput(text_to_analyze=user_input)
                    )
                    # Update capacity based on sentiment
                    snapshot.capacity = clamp01(snapshot.capacity + REFLECTION_CAPACITY_NUDGE_BASE * sentiment_result.score)
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {e}")

            # --- 2. Narrative Mode Processing ---
            style_directive = ""
            if is_enabled(Feature.NARRATIVE_MODES) and self.narrative_engine:
                try:
                    narrative_result = self.narrative_engine.determine_narrative_mode(snapshot.to_dict())
                    if not isinstance(narrative_result, dict):
                        logger.error(f"NarrativeModesEngine returned non-dict: {narrative_result}")
                        narrative_result = {}
                    style_directive = narrative_result.get("style_directive", "")
                except Exception as e:
                    logger.error(f"Error determining narrative mode: {e}")

            # --- 3. Practical Consequence Updates ---
            if is_enabled(Feature.PRACTICAL_CONSEQUENCE) and self.practical_consequence_engine:
                try:
                    self.practical_consequence_engine.update_signals_from_reflection(user_input)
                except Exception as e:
                    logger.error(f"Error updating practical consequences: {e}")

            # --- 4. LLM Processing ---
            try:
                # Construct prompt with style directive
                prompt = self._construct_arbiter_prompt(
                    user_input=user_input,
                    snapshot_dict=snapshot.to_dict(),
                    conversation_history=snapshot.conversation_history,
                    primary_task=None,  # We don't have task info at this point
                    task_titles=[],
                    style_directive_input=style_directive
                )
                llm_response = await self.llm_client.generate(prompt, ArbiterStandardResponse)
                arbiter_response = getattr(llm_response, "narrative", "")
            except LLMValidationError as e:
                error_message = f"LLM response validation error: {str(e)}"
                logger.error(error_message)
                return {
                    "tasks": [self._get_fallback_task("validation_error")],
                    "arbiter_response": f"(LLM validation error: {str(e)})",
                    "error": error_message,
                    "magnitude_description": "Unknown"
                }
            except LLMError as e:
                error_message = f"Error processing reflection: {str(e)}"
                logger.error(error_message)
                return {
                    "tasks": [self._get_fallback_task("llm_error")],
                    "arbiter_response": f"(LLM error: {str(e)})",
                    "error": error_message,
                    "magnitude_description": "Unknown"
                }

            # --- 5. Task Generation ---
            logger.debug("Getting next step from task engine")
            task_result = self.task_engine.get_next_step(snapshot.to_dict())
            logger.debug(f"Task engine response: {task_result}")
            generated_tasks = task_result.get("tasks", [])
            fallback_task = task_result.get("fallback_task")
            primary_task = task_result.get("primary_task")
            logger.debug(f"Extracted tasks - generated: {generated_tasks}, primary: {primary_task}, fallback: {fallback_task}")

            # --- 6. Soft Deadline Scheduling ---
            logger.debug("Checking soft deadline scheduling conditions:")
            logger.debug(f"SOFT_DEADLINES enabled (core): {is_enabled(Feature.SOFT_DEADLINES)}")
            logger.debug(f"Current path: {snapshot.current_path}")
            logger.debug(f"Primary task: {primary_task}")
            logger.debug(f"Generated tasks: {generated_tasks}")

            if is_enabled(Feature.SOFT_DEADLINES) and snapshot.current_path == "structured":
                try:
                    # Use a set to track task IDs we've already added
                    seen_task_ids = set()
                    tasks_for_deadline = []

                    if primary_task and primary_task.get("id") not in seen_task_ids:
                        tasks_for_deadline.append(primary_task)
                        seen_task_ids.add(primary_task.get("id"))
                        logger.debug(f"Added primary task to deadline scheduling: {primary_task}")

                    for task in generated_tasks:
                        if task.get("id") not in seen_task_ids:
                            tasks_for_deadline.append(task)
                            seen_task_ids.add(task.get("id"))
                            logger.debug(f"Added generated task to deadline scheduling: {task}")
                    
                    if tasks_for_deadline:  # Only schedule if there are tasks
                        logger.debug(f"Scheduling deadlines for tasks: {tasks_for_deadline}")
                        schedule_soft_deadlines(
                            snapshot=snapshot,
                            tasks=tasks_for_deadline
                        )
                        logger.debug("Successfully scheduled soft deadlines")
                except Exception as e:
                    logger.error(f"Error scheduling soft deadlines: {e}")
                    logger.exception("Full stack trace:")

            # Update snapshot state
            snapshot.current_batch_reflections.append(user_input)
            snapshot.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Describe magnitude for response
            if generated_tasks and "magnitude" in generated_tasks[0]:
                magnitude_description = describe_magnitude(generated_tasks[0]["magnitude"])

            return {
                "tasks": generated_tasks or ([fallback_task] if fallback_task else [self._get_fallback_task()]),
                "arbiter_response": arbiter_response,
                "error": error_message,
                "magnitude_description": magnitude_description
            }

        except Exception as e:
            error_message = f"Unexpected error in reflection processing: {str(e)}"
            logger.exception(error_message)
            return {
                "tasks": [self._get_fallback_task("unexpected_error")],
                "arbiter_response": "(An unexpected error occurred while processing your reflection.)",
                "error": error_message,
                "magnitude_description": "Unknown"
            }

    # --- Internal Helper Methods ---

    def _get_fallback_task(self, reason: str = "unknown") -> Dict[str, Any]:
        increment_fallback_count(reason)
        fallback_id = f"fallback_{uuid.uuid4()}"
        return {
            "id": fallback_id,
            "title": "Fallback Task",
            "description": f"This is a fallback task generated due to: {reason}",
            "priority": 1,
            "magnitude": 1,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "fallback"
        }

    def _construct_arbiter_prompt(
        self,
        user_input: str,
        snapshot_dict: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        primary_task: Optional[Dict[str, Any]],
        task_titles: List[str],
        style_directive_input: str = ""
    ) -> str:
        """Constructs the prompt for the Arbiter LLM call."""
        # Context Pruning (Simplified for example)
        pruned_snap_ctx = prune_context(snapshot_dict)
        context_summary = json.dumps(pruned_snap_ctx)

        # Task Representation
        task_summary = "No active task"
        if primary_task:
            task_summary = f"Primary Task: {primary_task.get('title', 'N/A')}"
            if len(task_titles) > 1:
                task_summary += f" | Other Tasks: {', '.join(task_titles[1:])}"

        # History Formatting (Basic example)
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]]) # Last 5 messages

        # Style Directive
        style_text = f"Style: {style_directive_input}" if style_directive_input else "Style: Default"
        if is_enabled(Feature.ENABLE_POETIC_ARBITER_VOICE):
             style_text = f"Style: Poetic and metaphorical. {style_directive_input}"

        # Prompt Construction (Example structure)
        prompt = f"""
Context Summary: {context_summary}

Recent Conversation:
{history_text}

Current Task Focus: {task_summary}

User's Latest Reflection: {user_input}

Instructions: Respond as the Forest Arbiter. Acknowledge the reflection briefly. Provide a narrative connecting the reflection to the current task(s) and the overall context. Refine the primary task details if necessary based on the reflection. Ensure your response follows the requested '{style_text}'. Your response must be a JSON object matching the ArbiterStandardResponse format, including:
- 'narrative' (string)
- optional 'task' (object) with at least the following fields: 'title', 'description', 'priority', 'magnitude', and 'status'.
"""
        logger.debug("Constructed Arbiter Prompt:\n%s", prompt[:500] + "..." if len(prompt) > 500 else prompt) # Log truncated prompt
        return prompt
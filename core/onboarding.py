# forest_app/core/onboarding.py

"""
Forest OS Onboarding and Session‑Management

This module contains:
  1. `onboard_user`: one‑time initialization of a JSON snapshot
     with NLP‑derived baselines (deep‑copy + persist).
  2. `run_onboarding`: CLI flow to capture a top‑level goal (Seed),
     target date, journey path, reflection, and baseline assessment.
  3. `run_forest_session` / `run_forest_session_async`: ongoing
     heartbeat loops for applying withering updates and persisting state.
"""

from __future__ import annotations

import sys
import time
import copy
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional, TYPE_CHECKING, TypeVar, cast, Union

# Use TYPE_CHECKING to prevent circular imports
if TYPE_CHECKING:
    from forest_app.core.orchestrator import ForestOrchestrator
    from forest_app.snapshot.snapshot import MemorySnapshot as MemorySnapshotType

from forest_app.utils.baseline_loader import load_user_baselines
from forest_app.config.constants import ORCHESTRATOR_HEARTBEAT_SEC

from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.modules.baseline_assessment import BaselineAssessmentEngine
from forest_app.hta_tree.seed import SeedManager

# Type variables
T = TypeVar('T')
SnapshotDict = Dict[str, Any]
SaveSnapshotCallable = Callable[[SnapshotDict], None]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# 1. One‑time, programmatic baseline injection
# -----------------------------------------------------------------------------

def onboard_user(
    snapshot: Union[SnapshotDict, MemorySnapshot],
    baselines: Dict[str, float],
    save_snapshot: SaveSnapshotCallable
) -> SnapshotDict:
    """
    One-time initialization of the user snapshot from NLP-derived baselines.

    Args:
        snapshot: Either a dict or MemorySnapshot instance containing user state
        baselines: Dictionary of baseline metrics
        save_snapshot: Callback to persist the snapshot
        
    Returns:
        Dict[str, Any]: The initialized snapshot dictionary
        
    Raises:
        ValueError: If baselines are invalid or missing required fields
        TypeError: If snapshot or save_snapshot are of wrong type
        RuntimeError: If snapshot persistence fails
    """
    if not isinstance(baselines, dict):
        raise TypeError("baselines must be a dictionary")
    if not callable(save_snapshot):
        raise TypeError("save_snapshot must be a callable")
        
    # Convert MemorySnapshot to dict if needed
    if isinstance(snapshot, MemorySnapshot):
        snapshot_dict = snapshot.to_dict()
    else:
        snapshot_dict = copy.deepcopy(snapshot)
        
    # Validate baseline structure
    if not all(isinstance(v, (int, float)) for v in baselines.values()):
        raise ValueError("All baseline values must be numeric")
        
    # Initialize component state
    cs = snapshot_dict.setdefault("component_state", {})
    cs["baselines"] = baselines

    try:
        # Load and validate baselines
        new_snapshot = load_user_baselines(snapshot_dict)
    except Exception as e:
        logger.error("Failed to load user baselines: %s", e)
        raise ValueError(f"Invalid baseline data: {e}")

    try:
        save_snapshot(new_snapshot)
        logger.info("Initial snapshot persisted during onboarding")
    except Exception as e:
        logger.error("Failed to save snapshot during onboarding: %s", e)
        raise RuntimeError(f"Failed to persist snapshot: {e}")

    return new_snapshot


# -----------------------------------------------------------------------------
# 2. CLI‑driven HTA seed onboarding flow
# -----------------------------------------------------------------------------

def _prompt(text: str) -> str:
    sys.stdout.write(f"{text.strip()}\n> ")
    sys.stdout.flush()
    return sys.stdin.readline().strip()


def _parse_date_iso(date_str: str) -> Optional[str]:
    try:
        dt = datetime.fromisoformat(date_str.strip())
        return dt.date().isoformat()
    except ValueError:
        return None


def _recommend_completion_date(hta_scope: int) -> str:
    days = max(hta_scope, 1) * 2
    return (datetime.utcnow() + timedelta(days=days)).date().isoformat()


def run_onboarding(snapshot: MemorySnapshot) -> None:
    """
    Run the entire CLI onboarding flow and mutate `snapshot` in place.

    Args:
        snapshot: The MemorySnapshot instance to update
        
    Raises:
        ValueError: If input validation fails
        RuntimeError: If critical onboarding steps fail
        KeyboardInterrupt: If user cancels the process
    """
    if not isinstance(snapshot, MemorySnapshot):
        raise TypeError("snapshot must be a MemorySnapshot instance")
        
    try:
        # 0. Seed details
        goal_title = _prompt(
            "What is the primary goal you wish to cultivate? "
            "(e.g. 'Run a 5k', 'Launch my blog')"
        ).strip()
        
        if not goal_title:
            raise ValueError("Goal title cannot be empty")
            
        seed_domain = _prompt(
            "In one word, which life domain does this goal belong to? "
            "(e.g. health, career, creativity)"
        ).strip().lower()
        
        if not seed_domain:
            raise ValueError("Domain cannot be empty")
            
    except (EOFError, KeyboardInterrupt) as e:
        logger.error("Onboarding interrupted during initial prompts: %s", e)
        raise KeyboardInterrupt("Onboarding cancelled by user") from e

    try:
        seed_manager = SeedManager()
        seed = seed_manager.plant_seed(goal_title, seed_domain, additional_context=None)
        if not seed:
            raise RuntimeError("Failed to create seed")
            
        snapshot.component_state["seed_manager"] = seed_manager.to_dict()
    except Exception as e:
        logger.error("Failed to initialize seed: %s", e)
        raise RuntimeError(f"Seed initialization failed: {e}")

    # Estimate HTA scope for date recommendation
    try:
        tree = getattr(seed, "hta_tree", None)
        hta_scope = 1  # Default scope
        if tree is not None:
            if hasattr(tree, "child_count"):
                hta_scope = tree.child_count
            elif isinstance(tree, dict):
                hta_scope = tree.get("root", {}).get("child_count", 1)
    except Exception as e:
        logger.warning("Failed to get HTA scope, using default: %s", e)
        hta_scope = 1

    # 1. Completion date
    date_iso = None
    while not date_iso:
        try:
            date_input = _prompt(
                "Enter your target completion date for this goal (YYYY‑MM‑DD) "
                "or type 'recommend' to let the forest suggest one:"
            ).strip().lower()

            if date_input == "recommend":
                date_iso = _recommend_completion_date(hta_scope)
                print(f"\n🌲 The forest recommends {date_iso} as a gentle target.\n")
            else:
                date_iso = _parse_date_iso(date_input)
                if not date_iso:
                    print("❌ Invalid date format. Please use YYYY‑MM‑DD.")
                    
        except (EOFError, KeyboardInterrupt) as e:
            logger.error("Onboarding interrupted during date prompt: %s", e)
            raise KeyboardInterrupt("Onboarding cancelled by user") from e
            
    snapshot.estimated_completion_date = date_iso

    # 2. Journey path
    options = {"1": "structured", "2": "blended", "3": "open"}
    path_chosen = False
    while not path_chosen:
        try:
            choice = _prompt(
                "Choose your journey mode:\n"
                "  1) Structured – clear soft deadlines and strong guidance\n"
                "  2) Blended    – guideposts without penalties\n"
                "  3) Open       – no deadlines, introspection‑heavy\n"
                "Enter 1, 2, or 3:"
            ).strip()

            if choice in options:
                snapshot.current_path = options[choice]
                path_chosen = True
            else:
                print("❌ Please enter 1, 2, or 3.")
                
        except (EOFError, KeyboardInterrupt) as e:
            logger.error("Onboarding interrupted during path selection: %s", e)
            raise KeyboardInterrupt("Onboarding cancelled by user") from e

    # 3. Reflection
    try:
        where_text = _prompt(
            "Describe where you are right now in relation to this goal. "
            "Feel free to share thoughts, feelings, or context:"
        ).strip()
        
        if not where_text:
            logger.warning("User provided empty reflection")
            where_text = "No initial reflection provided."
            
    except (EOFError, KeyboardInterrupt) as e:
        logger.error("Onboarding interrupted during reflection prompt: %s", e)
        raise KeyboardInterrupt("Onboarding cancelled by user") from e
        
    snapshot.core_state["where_you_are"] = where_text

    # 4. Baseline assessment
    print("\n🌿 Establishing your baseline… this may take a moment.\n")
    assessor = BaselineAssessmentEngine()
    try:
        baseline_data = asyncio.run(assessor.assess_baseline(goal_title, where_text))
        if not isinstance(baseline_data, dict):
            raise TypeError("Invalid baseline data format")

        # Initialize dev_index with validation
        development_data = baseline_data.get("development")
        if not isinstance(development_data, dict):
            raise ValueError("Missing or invalid development data")
            
        snapshot.dev_index.update_from_dict({
            "indexes": development_data,
            "adjustment_history": []
        })
        snapshot.component_state["dev_index"] = snapshot.dev_index.to_dict()

        # Validate and populate core metrics
        required_metrics = ["capacity", "shadow_score", "magnitude", "relationship"]
        for metric in required_metrics:
            value = baseline_data.get(metric)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid or missing {metric} value")
            setattr(snapshot, metric, float(value))

    except Exception as e:
        logger.error("Baseline assessment failed: %s", e)
        raise RuntimeError(f"Failed to establish baseline: {e}")

    # Onboarding complete
    snapshot.baseline_established = True
    logger.info("Onboarding completed successfully")
    print("\n✅ Onboarding complete! Your journey begins.\n")


# -----------------------------------------------------------------------------
# 3. Blocking & async heartbeat loops for ongoing session maintenance
# -----------------------------------------------------------------------------

def run_forest_session(
    snapshot: Dict[str, Any],
    save_snapshot: SaveSnapshotCallable,
    lock: Optional[threading.Lock] = None
) -> None:
    """
    Run a blocking forest session that updates withering and persists state.
    
    Args:
        snapshot: Dictionary containing user state
        save_snapshot: Callback to persist snapshot changes
        lock: Optional threading lock for synchronization
        
    Raises:
        RuntimeError: If critical session operations fail
        KeyboardInterrupt: If session is manually interrupted
    """
    if not isinstance(snapshot, dict):
        raise TypeError("snapshot must be a dictionary")
    if not callable(save_snapshot):
        raise TypeError("save_snapshot must be a callable")
        
    session_id = str(snapshot.get("user_id", "unknown"))
    orch = ForestOrchestrator(saver=save_snapshot)
    
    logger.info(
        "Starting blocking forest session for session=%s (interval=%s sec)",
        session_id, ORCHESTRATOR_HEARTBEAT_SEC
    )
    
    try:
        while True:
            start_time = time.monotonic()
            try:
                if lock:
                    with lock:
                        orch._update_withering(snapshot)
                        orch._save_component_states(snapshot)
                else:
                    orch._update_withering(snapshot)
                    orch._save_component_states(snapshot)
                    
            except Exception as tick_err:
                logger.exception(
                    "Error during heartbeat tick for session=%s: %s",
                    session_id, tick_err
                )
                # Don't exit on tick errors, try to continue
                
            # Calculate and handle sleep timing
            elapsed = time.monotonic() - start_time
            sleep_duration = max(0, ORCHESTRATOR_HEARTBEAT_SEC - elapsed)
            
            try:
                time.sleep(sleep_duration)
            except KeyboardInterrupt:
                raise
            except Exception as sleep_err:
                logger.error(
                    "Error during heartbeat sleep for session=%s: %s",
                    session_id, sleep_err
                )
                # Don't exit on sleep errors, try to continue
                
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; stopping session=%s", session_id)
        try:
            # Attempt final state save
            orch._save_component_states(snapshot)
        except Exception as e:
            logger.error(
                "Error saving state at shutdown for session=%s: %s",
                session_id, e
            )
    finally:
        logger.info("Blocking forest session stopped for session=%s", session_id)


async def run_forest_session_async(
    snapshot: Dict[str, Any],
    save_snapshot: SaveSnapshotCallable,
    lock: Optional[threading.Lock] = None
) -> None:
    """
    Run an async forest session that updates withering and persists state.
    
    Args:
        snapshot: Dictionary containing user state
        save_snapshot: Callback to persist snapshot changes
        lock: Optional threading lock for synchronization
        
    Raises:
        RuntimeError: If critical session operations fail
        asyncio.CancelledError: If session is cancelled
    """
    if not isinstance(snapshot, dict):
        raise TypeError("snapshot must be a dictionary")
    if not callable(save_snapshot):
        raise TypeError("save_snapshot must be a callable")
        
    session_id = str(snapshot.get("user_id", "unknown"))
    orch = ForestOrchestrator(saver=save_snapshot)
    
    logger.info(
        "Starting async forest session for session=%s (interval=%s sec)",
        session_id, ORCHESTRATOR_HEARTBEAT_SEC
    )
    
    try:
        while True:
            start_time = asyncio.get_running_loop().time()
            try:
                if lock:
                    with lock:
                        orch._update_withering(snapshot)
                        orch._save_component_states(snapshot)
                else:
                    orch._update_withering(snapshot)
                    orch._save_component_states(snapshot)
                    
            except Exception as tick_err:
                logger.exception(
                    "Error during async heartbeat tick for session=%s: %s",
                    session_id, tick_err
                )
                # Don't exit on tick errors, try to continue
                
            # Calculate and handle sleep timing
            elapsed = asyncio.get_running_loop().time() - start_time
            sleep_duration = max(0, ORCHESTRATOR_HEARTBEAT_SEC - elapsed)
            
            try:
                await asyncio.sleep(sleep_duration)
            except asyncio.CancelledError:
                raise
            except Exception as sleep_err:
                logger.error(
                    "Error during async heartbeat sleep for session=%s: %s",
                    session_id, sleep_err
                )
                # Don't exit on sleep errors, try to continue
                
    except asyncio.CancelledError:
        logger.info("Async session cancelled for session=%s", session_id)
        try:
            # Attempt final state save
            orch._save_component_states(snapshot)
        except Exception as e:
            logger.error(
                "Error saving state at cancellation for session=%s: %s",
                session_id, e
            )
    except Exception as e:
        logger.error(
            "Unhandled error in async session for session=%s: %s",
            session_id, e
        )
        raise RuntimeError(f"Async session failed: {e}")
    finally:
        logger.info("Async forest session stopped for session=%s", session_id)

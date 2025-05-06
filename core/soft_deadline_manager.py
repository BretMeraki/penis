"""forest_app/modules/soft_deadline_manager.py

Assigns and manages *soft deadlines* for tasks, driven by the user's
estimated completion date and journey path (structured, blended, open).
Respects the SOFT_DEADLINES feature flag.

Rules
-----
- **Structured**: Deadlines are distributed **evenly** from *now* to the
  `estimated_completion_date`.
- **Blended**: Same even distribution **plus jitter** of ±20 % to feel
  like flexible guideposts.
- **Open**: **No deadlines** are attached.
- **Feature Disabled**: **No deadlines** are attached; existing ones removed.

The manager writes an ISO‑8601 `soft_deadline` string into each task
(dict) and returns the updated list.
"""

from __future__ import annotations

import random
import logging # Added logging import
from datetime import datetime, timedelta, timezone # Added timezone
from typing import List, Dict, Any, Iterable, Optional # Added Optional

# --- Import Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger = logging.getLogger("soft_deadline_init")
    logger.warning("Feature flags module not found in soft_deadline_manager. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        SOFT_DEADLINES = "FEATURE_ENABLE_SOFT_DEADLINES" # Define the specific flag
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True

# Assume MemorySnapshot might not be available if core snapshot system changes
try:
     from forest_app.snapshot.snapshot import MemorySnapshot
except ImportError:
     logger = logging.getLogger("soft_deadline_init")
     logger.error("Failed to import MemorySnapshot. Soft deadline functions may fail.")
     # Define a dummy class or use Any if snapshot type hint is crucial elsewhere
     class MemorySnapshot:
          current_path: str = "structured"
          estimated_completion_date: Optional[str] = None
          task_backlog: List[Dict[str, Any]] = []


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    """Return an ISO‑8601 string without microseconds, timezone aware (UTC)."""
    # Ensure datetime is timezone aware before formatting
    if dt.tzinfo is None:
         dt = dt.replace(tzinfo=timezone.utc) # Assume UTC if naive
    else:
         dt = dt.astimezone(timezone.utc) # Convert to UTC if already aware
    return dt.replace(microsecond=0).isoformat(timespec='seconds') # Keep Z implicit via UTC


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def schedule_soft_deadlines(
    snapshot: MemorySnapshot,
    tasks: Iterable[Dict[str, Any]],
    *,
    jitter_pct: float = 0.20,
    override_existing: bool = False,
) -> List[Dict[str, Any]]:
    """
    Assign `soft_deadline` fields to each task in *tasks*.
    If SOFT_DEADLINES feature is disabled, removes existing deadlines.

    Parameters
    ----------
    snapshot : MemorySnapshot
        Provides `current_path` and `estimated_completion_date`.
    tasks : iterable of task dicts
        Each task will be mutated in‑place.
    jitter_pct : float, optional
        Fractional jitter applied in **blended** mode (±20 % default).
    override_existing : bool, optional
        If False (default), tasks that already have `soft_deadline` are
        left untouched (unless feature is disabled or path is open).

    Returns
    -------
    list of dict
        Reference to the same task dicts, updated.
    """
    tasks_list = list(tasks) # Convert iterable to list for modification and len()

    # --- Feature Flag Check ---
    if not is_enabled(Feature.SOFT_DEADLINES):
        logger.debug("Skipping deadline scheduling: SOFT_DEADLINES feature disabled. Removing existing deadlines.")
        # Ensure no lingering deadlines if feature is off
        for t in tasks_list:
            if isinstance(t, dict): # Check if t is actually a dict
                 t.pop("soft_deadline", None)
        return tasks_list
    # --- End Check ---

    # Feature is enabled, proceed with path-based logic
    path = getattr(snapshot, "current_path", "structured").lower()
    if path == "open":
        logger.debug("Current path is 'open'. Removing existing deadlines.")
        # Ensure no lingering deadlines for open path
        for t in tasks_list:
            if isinstance(t, dict):
                 t.pop("soft_deadline", None)
        return tasks_list

    # Structured or Blended path requires estimated completion date
    est_comp_date_str = getattr(snapshot, "estimated_completion_date", None)
    if not est_comp_date_str:
        logger.error("Snapshot missing 'estimated_completion_date'; cannot generate soft deadlines for path '%s'.", path)
        # Decide behavior: raise error, or just return tasks unmodified? Returning unmodified for now.
        # Or remove existing deadlines? Let's remove for consistency.
        for t in tasks_list:
             if isinstance(t, dict): t.pop("soft_deadline", None)
        # raise ValueError("Snapshot missing `estimated_completion_date`")
        return tasks_list

    try:
        # Assume completion date string includes timezone or is UTC
        end_dt = datetime.fromisoformat(est_comp_date_str.replace("Z", "+00:00"))
        # Ensure end_dt is timezone-aware (assume UTC if not specified)
        if end_dt.tzinfo is None:
             end_dt = end_dt.replace(tzinfo=timezone.utc)
        else:
             end_dt = end_dt.astimezone(timezone.utc) # Convert to UTC

    except (ValueError, TypeError) as e:
         logger.error("Invalid format for 'estimated_completion_date': '%s'. Error: %s. Cannot generate deadlines.", est_comp_date_str, e)
         for t in tasks_list:
              if isinstance(t, dict): t.pop("soft_deadline", None)
         return tasks_list


    now = datetime.now(timezone.utc) # Use timezone-aware UTC now
    if end_dt <= now:
        logger.warning("Estimated completion date is in the past. Using default 7-day span.")
        end_dt = now + timedelta(days=7)

    total_span_sec = (end_dt - now).total_seconds()
    if total_span_sec <= 0: # Should not happen with the check above, but belt-and-suspenders
         logger.warning("Calculated time span for deadlines is zero or negative. Cannot schedule.")
         for t in tasks_list:
              if isinstance(t, dict): t.pop("soft_deadline", None)
         return tasks_list

    num_tasks = len(tasks_list)
    if num_tasks == 0:
        logger.debug("No tasks provided to schedule_soft_deadlines.")
        return tasks_list

    # Avoid division by zero if only one task
    even_step = total_span_sec / num_tasks if num_tasks > 0 else total_span_sec

    updated = []
    for idx, task in enumerate(tasks_list, start=1):
        if not isinstance(task, dict):
             logger.warning("Skipping non-dictionary item in tasks list.")
             updated.append(task) # Append non-dict item back? Or skip? Appending for now.
             continue

        if not override_existing and task.get("soft_deadline"):
            logger.debug("Task %s already has deadline, skipping (override=False).", task.get('id', idx))
            updated.append(task)
            continue

        # Base offset - distribute across the *remaining* span more accurately
        offset_sec = (total_span_sec / num_tasks) * idx

        # Jitter for blended path
        if path == "blended":
            jitter_range = even_step * jitter_pct # Jitter relative to the step size
            offset_sec += random.uniform(-jitter_range, jitter_range)
            # Clamp offset to ensure deadline is not before now or after end_dt
            offset_sec = max(0.0, min(offset_sec, total_span_sec))

        try:
            deadline_dt = now + timedelta(seconds=offset_sec)
            task["soft_deadline"] = _iso(deadline_dt)
            logger.debug("Set deadline for task %s: %s", task.get('id', idx), task["soft_deadline"])
        except OverflowError:
            logger.error("Overflow calculating deadline for task %s with offset %s. Skipping deadline.", task.get('id', idx), offset_sec)
            task.pop("soft_deadline", None) # Remove potential bad value

        updated.append(task) # Append the modified (or original if skipped) task

    return updated


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def schedule_backlog(snapshot: MemorySnapshot, *, override_existing=False) -> None:
    """
    Assign deadlines to *all* tasks in `snapshot.task_backlog`.
    Does nothing if SOFT_DEADLINES feature is disabled.
    """
    # --- Feature Flag Check ---
    if not is_enabled(Feature.SOFT_DEADLINES):
        logger.debug("Skipping schedule_backlog: SOFT_DEADLINES feature disabled.")
        # Optionally remove existing deadlines from backlog here if desired when feature is off
        # for task in getattr(snapshot, "task_backlog", []):
        #     if isinstance(task, dict): task.pop("soft_deadline", None)
        return
    # --- End Check ---

    if not hasattr(snapshot, "task_backlog") or not isinstance(snapshot.task_backlog, list):
        logger.warning("Cannot schedule backlog: snapshot.task_backlog is missing or not a list.")
        return

    logger.info("Scheduling soft deadlines for task backlog...")
    # The core function will handle the actual scheduling logic and flag check
    schedule_soft_deadlines(
        snapshot, snapshot.task_backlog, override_existing=override_existing
    )
    logger.info("Finished scheduling soft deadlines for task backlog.")


def hours_until_deadline(task: Dict[str, Any]) -> float:
    """
    Return hours until this task's soft deadline.
    Returns float('inf') if no deadline or if SOFT_DEADLINES feature is disabled.
    """
    # --- Feature Flag Check ---
    if not is_enabled(Feature.SOFT_DEADLINES):
        # If feature is off, act as if no deadline exists
        return float("inf")
    # --- End Check ---

    if not isinstance(task, dict):
         return float("inf") # Cannot get deadline from non-dict

    sd = task.get("soft_deadline")
    if not sd or not isinstance(sd, str):
        return float("inf")

    try:
        # Attempt to parse ISO string, assuming UTC if no offset
        # Handle potential 'Z' suffix
        if sd.endswith("Z"):
            sd_dt = datetime.fromisoformat(sd.replace("Z", "+00:00"))
        else:
             # If no Z and no offset, assume UTC
             sd_dt = datetime.fromisoformat(sd)
             if sd_dt.tzinfo is None:
                  sd_dt = sd_dt.replace(tzinfo=timezone.utc)

        # Ensure comparison is between two timezone-aware UTC datetimes
        now_utc = datetime.now(timezone.utc)
        time_diff_seconds = (sd_dt - now_utc).total_seconds()

        # Return 0 if deadline has passed, otherwise hours remaining
        return max(time_diff_seconds / 3600.0, 0.0)

    except (ValueError, TypeError) as e:
        logger.warning("Could not parse soft_deadline '%s': %s. Returning infinity.", sd, e)
        return float("inf")
    except Exception as e:
         logger.error("Unexpected error calculating hours_until_deadline for '%s': %s", sd, e, exc_info=True)
         return float("inf")

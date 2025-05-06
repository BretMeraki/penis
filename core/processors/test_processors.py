"""Test module for core processors."""
import os
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from typing import Dict, Any, List
import logging
import json
import copy
from dataclasses import dataclass, field

import pytest
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient
from forest_app.core.main import app
from forest_app.routers.onboarding import add_context_endpoint
from forest_app.hta_tree.task_engine import TaskEngine
from forest_app.core.security import get_current_active_user
from forest_app.hta_tree.hta_service import HTAService
from forest_app.hta_tree.hta_tree import HTATree
from forest_app.integrations.llm import LLMClient, LLMError, LLMValidationError
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.core.logging_tracking import TaskFootprintLogger
from forest_app.modules.cognitive.sentiment import SecretSauceSentimentEngineHybrid, SentimentOutput
from forest_app.modules.cognitive.practical_consequence import PracticalConsequenceEngine
from forest_app.modules.cognitive.narrative_modes import NarrativeModesEngine
from forest_app.modules.resource.xp_mastery import XPMastery
from forest_app.core.feature_flags import Feature
from forest_app.routers.onboarding import OnboardingResponse
from forest_app.core.processors.reflection_processor import ReflectionProcessor
from forest_app.core.processors.completion_processor import CompletionProcessor
from forest_app.modules.resource.xp_mastery import XPMasteryEngine

# Mock settings before importing the processors
with patch('forest_app.config.settings.settings', new_callable=Mock) as mock_settings:
    mock_settings.GOOGLE_API_KEY = "test_api_key"
    mock_settings.DB_CONNECTION_STRING = "postgresql://test:test@localhost:5432/test_db"
    mock_settings.FEATURE_ENABLE_SENTIMENT_ANALYSIS = True
    mock_settings.FEATURE_ENABLE_NARRATIVE_MODES = True
    mock_settings.FEATURE_ENABLE_SOFT_DEADLINES = True
    mock_settings.FEATURE_ENABLE_POETIC_ARBITER_VOICE = True
    mock_settings.FEATURE_ENABLE_CORE_TASK_ENGINE = True
    mock_settings.FEATURE_ENABLE_CORE_HTA = True
    mock_settings.FEATURE_ENABLE_PRACTICAL_CONSEQUENCE = False  # Disable practical consequence (singular)
    mock_settings.TESTING = True

# --- Test Fixtures ---

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that raises errors on demand."""
    mock = AsyncMock(spec=LLMClient)
    
    async def generate_with_error(*args, **kwargs):
        if kwargs.get("should_fail", False):
            raise LLMError("Simulated LLM failure")
        if kwargs.get("validation_fail", False):
            raise LLMValidationError("Invalid response format")
        return {
            "narrative": "Test narrative",
            "task": {
                "id": "test_task_1",
                "title": "Refined Task",
                "description": "Refined Description",
                "priority": 3,
                "magnitude": 4
            }
        }
    
    mock.generate = AsyncMock(side_effect=generate_with_error)
    return mock

@pytest.fixture
def mock_task_engine():
    """Create a mock task engine that can simulate failures."""
    mock = Mock(spec=TaskEngine)
    
    def get_next_step_with_error(snapshot_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Return empty task list if batch is completed and no reflections
        if (snapshot_dict.get("current_frontier_batch_ids", []) == [] and 
            not snapshot_dict.get("current_batch_reflections", [])):
            return {
                "tasks": [],
                "fallback_task": None
            }
        # Return empty task list if there's an LLM error
        if snapshot_dict.get("llm_error"):
            return {
                "tasks": [],
                "fallback_task": None
            }
        # Generate task based on reflections
        tasks = []
        if snapshot_dict.get("current_batch_reflections"):
            tasks.append({
                "id": f"reflection_task_1",
                "title": f"Task from reflection",
                "description": f"Generated from: {snapshot_dict['current_batch_reflections'][-1][:50]}...",
                "priority": 5,
                "magnitude": 5,
                "status": "incomplete",
                "hta_node_id": f"test_node_1"
            })
        return {
            "tasks": tasks,
            "fallback_task": None
        }
    
    mock.get_next_step = Mock(side_effect=get_next_step_with_error)
    return mock

@pytest.fixture
def mock_hta_service():
    """Create a mock HTA service that can simulate failures."""
    mock = AsyncMock(spec=HTAService)
    
    async def load_tree_with_error(snapshot: MemorySnapshot, should_fail: bool = False):
        if should_fail:
            raise ValueError("Failed to load HTA tree")
        if not isinstance(snapshot, MemorySnapshot):
            raise TypeError("Invalid snapshot type")
        return {
            "id": "test_tree",
            "nodes": [{"id": "test_node", "status": "incomplete"}]
        }
    
    async def save_tree_with_error(snapshot: MemorySnapshot, tree: Dict[str, Any], should_fail: bool = False):
        if should_fail:
            return False
        if not tree or not isinstance(tree, dict):
            raise ValueError("Invalid tree data")
        return True
    
    async def evolve_tree_with_error(tree: Dict[str, Any], reflections: List[str], should_fail: bool = False):
        if should_fail:
            return None
        if not reflections:
            return None
        return {
            "id": "evolved_tree",
            "nodes": [{"id": "evolved_node", "status": "incomplete"}]
        }
    
    mock.load_tree = AsyncMock(side_effect=load_tree_with_error)
    mock.save_tree = AsyncMock(side_effect=save_tree_with_error)
    mock.update_node_status = AsyncMock(return_value=True)
    mock.evolve_tree = AsyncMock(side_effect=evolve_tree_with_error)
    return mock

@pytest.fixture
def mock_snapshot():
    """Create a mock snapshot with basic attributes."""
    snapshot = Mock(spec=MemorySnapshot)
    
    # Set up mock snapshot with estimated completion date and current path
    future_date = datetime.now(timezone.utc) + timedelta(days=7)
    
    # Create real lists for list attributes
    task_backlog = []
    current_batch_reflections = []
    conversation_history = []
    current_frontier_batch_ids = []
    
    # Configure the mock with initial values
    snapshot.configure_mock(**{
        "task_backlog": task_backlog,
        "current_batch_reflections": current_batch_reflections,
        "conversation_history": conversation_history,
        "current_frontier_batch_ids": current_frontier_batch_ids,
        "withering_level": 0.0,
        "core_state": {"hta_tree": {}},
        "capacity": 0.5,
        "shadow_score": 0.5,
        "current_path": "structured",
        "estimated_completion_date": future_date.isoformat()
    })

    # Set up to_dict to return a dictionary with all necessary fields
    snapshot.to_dict.return_value = {
        "estimated_completion_date": future_date.isoformat(),
        "current_path": "structured",
        "shadow_score": 0.5,
        "capacity": 0.5,
        "magnitude": 5.0,
        "task_backlog": task_backlog,
        "current_batch_reflections": current_batch_reflections,
        "conversation_history": conversation_history,
        "current_frontier_batch_ids": current_frontier_batch_ids,
        "withering_level": 0.0,
        "core_state": {"hta_tree": {}}
    }

    return snapshot

@pytest.fixture
def mock_xp_engine():
    """Create a mock XP Mastery engine for testing."""
    mock = Mock(spec=XPMasteryEngine)
    mock.current_xp = 0.0
    mock.completed_challenges = []
    mock.current_stage_index = 0
    
    def calculate_xp_with_error(task: Dict[str, Any], should_fail: bool = False) -> float:
        if should_fail:
            raise ValueError("Simulated XP calculation failure")
        if not isinstance(task, dict):
            raise TypeError("Invalid task format")
            
        base_xp = 10.0
        priority_mult = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.0}.get(task.get("priority", "medium"), 1.0)
        complexity_mult = {"trivial": 0.5, "simple": 1.0, "moderate": 1.5, "complex": 2.0}.get(task.get("complexity", "simple"), 1.0)
        return round(base_xp * priority_mult * complexity_mult, 2)
    
    def get_current_stage_with_error(should_fail: bool = False) -> Dict[str, Any]:
        if should_fail:
            raise ValueError("Simulated stage lookup failure")
        stages = [
            {"name": "Novice", "min_xp": 0, "max_xp": 100, "type": "Basic"},
            {"name": "Apprentice", "min_xp": 100, "max_xp": 300, "type": "Integration"}
        ]
        for stage in stages:
            if mock.current_xp >= stage["min_xp"] and mock.current_xp < stage["max_xp"]:
                return stage
        return stages[0]
    
    mock.calculate_xp_gain = Mock(side_effect=calculate_xp_with_error)
    mock.get_current_stage = Mock(side_effect=get_current_stage_with_error)
    mock.validate_snapshot = Mock(return_value=True)
    mock.generate_challenge_content = AsyncMock(return_value={
        "stage": "Novice",
        "type": "Basic",
        "content": "Test challenge content",
        "created_at": datetime.now().isoformat(),
        "xp_required": 100,
        "current_xp": 0.0
    })
    
    return mock

# --- Fake HTAValidationModel for onboarding test ---
class FakeHTARoot:
    def __init__(self):
        self.id = "root_testuuid"
        self.title = "Run a marathon"
        self.description = "Plan for marathon"
        self.priority = 0.5
        self.depends_on = []
        self.estimated_energy = "medium"
        self.estimated_time = "medium"
        self.linked_tasks = []
        self.is_milestone = True
        self.rationale = ""
        self.status_suggestion = "pending"
        self.children = []
    def model_dump(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "depends_on": self.depends_on,
            "estimated_energy": self.estimated_energy,
            "estimated_time": self.estimated_time,
            "linked_tasks": self.linked_tasks,
            "is_milestone": self.is_milestone,
            "rationale": self.rationale,
            "status_suggestion": self.status_suggestion,
            "children": self.children,
        }

class FakeHTAValidationModel:
    def __init__(self):
        self.hta_root = FakeHTARoot()
    def model_dump(self, mode=None):
        return {"hta_root": self.hta_root.model_dump()}
    @staticmethod
    def model_validate(data):
        return FakeHTAValidationModel()
    def model_dump_json(self, indent=None):
        import json
        return json.dumps(self.model_dump(), indent=indent)

class FakeSeed:
    def __init__(self, *args, **kwargs):
        self.description = kwargs.get("description", "Run a marathon")
        self.seed_id = kwargs.get("seed_id", "seed_testuuid")
        self.seed_name = kwargs.get("seed_name", "Run a marathon")
        self.seed_domain = kwargs.get("seed_domain", "General")
        self.status = kwargs.get("status", "active")
        self.hta_tree = kwargs.get("hta_tree", {"root": {"id": "root_testuuid"}})
        self.created_at = kwargs.get("created_at", "2023-01-01T00:00:00Z")
    def to_dict(self):
        return {
            "description": self.description,
            "seed_id": self.seed_id,
            "seed_name": self.seed_name,
            "seed_domain": self.seed_domain,
            "status": self.status,
            "hta_tree": self.hta_tree,
            "created_at": self.created_at,
        }

@dataclass
class SimpleSnapshot:
    current_path: str = "structured"
    estimated_completion_date: str = ""
    task_backlog: list = field(default_factory=list)
    current_frontier_batch_ids: list = field(default_factory=list)
    capacity: float = 1.0
    conversation_history: list = field(default_factory=list)
    # Add any other fields needed for the test
    def to_dict(self):
        return {
            "current_path": self.current_path,
            "estimated_completion_date": self.estimated_completion_date,
            "task_backlog": self.task_backlog,
            "current_frontier_batch_ids": self.current_frontier_batch_ids,
            "capacity": self.capacity,
            "conversation_history": self.conversation_history
        }

# --- ReflectionProcessor Tests ---

@pytest.mark.asyncio
async def test_reflection_processor_init_validation():
    """Test ReflectionProcessor initialization validation."""
    with pytest.raises(TypeError):
        ReflectionProcessor(llm_client=None, task_engine=Mock(spec=TaskEngine))
    
    with pytest.raises(TypeError):
        ReflectionProcessor(llm_client=Mock(spec=LLMClient), task_engine=None)

@pytest.mark.asyncio
async def test_reflection_processor_empty_input(
    mock_llm_client,
    mock_task_engine,
    mock_snapshot
):
    """Test handling of empty input in reflection processing."""
    processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine
    )
    
    result = await processor.process("", mock_snapshot)
    result_empty = await processor.process("   ", mock_snapshot)
    
    assert isinstance(result["tasks"], list)
    assert isinstance(result_empty["tasks"], list)
    assert len(result["tasks"]) > 0
    assert result["tasks"][0]["id"].startswith("fallback_")

@pytest.mark.asyncio
async def test_reflection_processor_llm_validation_error(
    mock_llm_client,
    mock_task_engine,
    mock_snapshot
):
    """Test handling of LLM validation errors."""
    processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine
    )
    
    mock_llm_client.generate = AsyncMock(side_effect=LLMValidationError("Invalid response format"))
    
    result = await processor.process("Test reflection", mock_snapshot)
    
    assert "(LLM validation error: Invalid response format)" in result["arbiter_response"]
    assert isinstance(result["tasks"], list)
    assert len(result["tasks"]) > 0

@pytest.mark.asyncio
async def test_reflection_processor_snapshot_mutation(
    mock_llm_client,
    mock_task_engine,
    mock_snapshot
):
    """Test that reflection processing properly updates snapshot state."""
    processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine
    )
    
    original_reflections = len(mock_snapshot.current_batch_reflections)
    original_history = len(mock_snapshot.conversation_history)
    
    await processor.process("Test reflection", mock_snapshot)
    
    assert len(mock_snapshot.current_batch_reflections) == original_reflections + 1
    assert len(mock_snapshot.conversation_history) > original_history

# --- CompletionProcessor Tests ---

@pytest.mark.asyncio
async def test_completion_processor_init_validation():
    """Test CompletionProcessor initialization validation."""
    with pytest.raises(TypeError):
        CompletionProcessor(hta_service=None, task_engine=Mock(spec=TaskEngine))
    
    with pytest.raises(TypeError):
        CompletionProcessor(hta_service=Mock(spec=HTAService), task_engine=None)

@pytest.mark.asyncio
async def test_completion_processor_invalid_inputs(
    mock_hta_service,
    mock_task_engine,
    mock_snapshot
):
    """Test handling of various invalid inputs in completion processing."""
    processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine
    )
    
    mock_db = Mock(spec=Session)
    mock_logger = Mock(spec=TaskFootprintLogger)
    
    # Test empty task ID
    result_empty = await processor.process(
        task_id="",
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    assert not result_empty["success"]
    assert "Invalid task_id" in result_empty["error"]
    
    # Test invalid snapshot type
    result_invalid_snapshot = await processor.process(
        task_id="test_task",
        success=True,
        snapshot={},  # type: ignore
        db=mock_db,
        task_logger=mock_logger
    )
    assert not result_invalid_snapshot["success"]
    assert "Invalid snapshot type" in result_invalid_snapshot["error"]

@pytest.mark.asyncio
async def test_completion_processor_withering_update(
    mock_hta_service,
    mock_task_engine,
    mock_snapshot
):
    """Test withering level updates on task completion."""
    processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine
    )
    
    task = {
        "id": "test_task",
        "title": "Test Task"
    }
    mock_snapshot.task_backlog = [task]
    mock_snapshot.withering_level = 0.5  # Set initial withering level higher than 0
    original_withering = mock_snapshot.withering_level
    
    mock_db = Mock(spec=Session)
    mock_logger = Mock(spec=TaskFootprintLogger)
    
    # Test successful completion
    result_success = await processor.process(
        task_id="test_task",
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    assert mock_snapshot.withering_level < original_withering
    
    # Test failed completion
    mock_snapshot.withering_level = original_withering
    result_failure = await processor.process(
        task_id="test_task",
        success=False,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    assert mock_snapshot.withering_level == original_withering

@pytest.mark.asyncio
async def test_completion_processor_batch_evolution(
    mock_hta_service,
    mock_task_engine,
    mock_snapshot
):
    """Test batch completion and HTA evolution."""
    processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine
    )
    
    # Set up mock snapshot with proper attributes
    mock_snapshot.task_backlog = [{"id": "test_task", "hta_node_id": "test_node", "title": "Test Task"}]
    mock_snapshot.current_frontier_batch_ids = ["test_task"]
    mock_snapshot.current_batch_reflections = ["Test reflection"]
    mock_snapshot.core_state = {"hta_tree": {}}
    
    # Set up mock HTA service
    mock_hta_service.load_tree = AsyncMock(return_value=HTATree())
    mock_hta_service.evolve_tree = AsyncMock(return_value=HTATree())
    mock_hta_service.save_tree = AsyncMock()
    
    mock_db = Mock(spec=Session)
    mock_logger = Mock(spec=TaskFootprintLogger)
    mock_logger.log_task_completion = AsyncMock()
    
    result = await processor.process(
        task_id="test_task",
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    
    assert result["batch_completed"]
    assert mock_hta_service.evolve_tree.called
    assert len(mock_snapshot.current_frontier_batch_ids) == 0
    assert len(mock_snapshot.current_batch_reflections) == 0

# --- Integration Tests ---

@pytest.mark.asyncio
async def test_reflection_completion_full_cycle(
    mock_llm_client,
    mock_task_engine,
    mock_hta_service,
    mock_snapshot
):
    """Test a complete reflection-completion cycle with all components."""
    # Set up mock snapshot
    mock_snapshot.current_batch_reflections = []
    mock_snapshot.task_backlog = []
    mock_snapshot.current_frontier_batch_ids = []
    mock_snapshot.core_state = {"hta_tree": {}}
    mock_snapshot.current_path = "structured"
    future_date = datetime.now(timezone.utc) + timedelta(days=7)
    mock_snapshot.estimated_completion_date = future_date.isoformat()

    # Set up mock task engine to return tasks
    mock_task_engine.get_next_step.return_value = {
        "tasks": [
            {
                "id": "test_task_1",
                "title": "Test Task 1",
                "description": "Test Description 1",
                "priority": 5,
                "magnitude": 5,
                "status": "incomplete",
                "hta_node_id": "test_node_1"
            }
        ],
        "fallback_task": None,
        "primary_task": {
            "id": "test_task_1",
            "title": "Test Task 1",
            "description": "Test Description 1",
            "priority": 5,
            "magnitude": 5,
            "status": "incomplete",
            "hta_node_id": "test_node_1"
        }
    }

    # Set up mock LLM client
    mock_llm_client.generate.return_value = {
        "narrative": "Test narrative",
        "task": None
    }

    # Set up mock HTA service
    mock_hta_service.evolve_tree = AsyncMock(return_value=HTATree())

    mock_sentiment = Mock()
    mock_sentiment.analyze_emotional_field = Mock()
    mock_pce = Mock()
    mock_pce.update_signals_from_reflection = Mock()
    mock_narrative = Mock()
    mock_narrative.determine_narrative_mode = Mock()
    reflection_processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine,
        sentiment_engine=mock_sentiment,
        practical_consequence_engine=mock_pce,
        narrative_engine=mock_narrative
    )

    completion_processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine
    )

    # Process multiple reflections
    reflections = [
        "First reflection",
        "Second reflection",
        "Third reflection"
    ]

    tasks_generated = []
    for reflection in reflections:
        result = await reflection_processor.process(reflection, mock_snapshot)
        tasks_generated.extend(result["tasks"])

    assert len(tasks_generated) > 0
    assert len(mock_snapshot.current_batch_reflections) == len(reflections)

    # Complete each task
    mock_db = Mock(spec=Session)
    mock_logger = Mock(spec=TaskFootprintLogger)
    mock_logger.log_task_completion = AsyncMock()  # Add the async mock method

    for task in tasks_generated:
        mock_snapshot.current_frontier_batch_ids.append(task["id"])
        mock_snapshot.task_backlog.append(task)  # Add task to backlog
        result = await completion_processor.process(
            task_id=task["id"],
            success=True,
            snapshot=mock_snapshot,
            db=mock_db,
            task_logger=mock_logger
        )
        assert result["success"]

    # Verify final state
    assert len(mock_snapshot.task_backlog) == 0
    assert mock_task_engine.get_next_step.called

@pytest.mark.asyncio
async def test_error_recovery_cycle(
    mock_llm_client,
    mock_task_engine,
    mock_hta_service,
    mock_snapshot
):
    """Test recovery from errors during a reflection-completion cycle."""
    reflection_processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine
    )
    
    completion_processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine
    )
    
    # First reflection with LLM failure
    mock_llm_client.generate = AsyncMock(side_effect=LLMError("First failure"))
    mock_task_engine.get_next_step.return_value = {
        "tasks": [
            {
                "id": "fallback_task_1",
                "title": "Reflect on your current focus",
                "description": "Take a moment to consider what feels most important right now.",
                "priority": 5,
                "magnitude": 5,
                "status": "incomplete",
                "hta_node_id": "test_node_1"
            }
        ],
        "fallback_task": None
    }
    result1 = await reflection_processor.process("Test reflection 1", mock_snapshot)
    assert result1["tasks"][0]["id"].startswith("fallback_")

    # Second reflection succeeds
    mock_llm_client.generate = AsyncMock(return_value={
        "narrative": "Success",
        "task": {
            "id": "test_task_2",
            "title": "Test Task 2",
            "description": "Test Description 2",
            "priority": 5,
            "magnitude": 5,
            "status": "incomplete",
            "hta_node_id": "test_node_2"
        }
    })
    # Update task engine response for second reflection
    mock_task_engine.get_next_step = Mock(return_value={
        "tasks": [
            {
                "id": "test_task_2",
                "title": "Test Task 2",
                "description": "Test Description 2",
                "priority": 5,
                "magnitude": 5,
                "status": "incomplete",
                "hta_node_id": "test_node_2"
            }
        ],
        "fallback_task": None,
        "primary_task": {
            "id": "test_task_2",
            "title": "Test Task 2",
            "description": "Test Description 2",
            "priority": 5,
            "magnitude": 5,
            "status": "incomplete",
            "hta_node_id": "test_node_2"
        }
    })
    result2 = await reflection_processor.process("Test reflection 2", mock_snapshot)
    
    # Try completing tasks
    mock_db = Mock(spec=Session)
    mock_logger = Mock(spec=TaskFootprintLogger)
    mock_logger.log_task_completion = AsyncMock(return_value=None)  # Ensure it returns None

    # Add the fallback task to the backlog
    mock_snapshot.task_backlog = [result1["tasks"][0], result2["tasks"][0]]  # Add both tasks to the backlog

    # HTA failure during first completion
    mock_hta_service.load_tree = AsyncMock(side_effect=ValueError("HTA failure"))
    mock_hta_service.evolve_tree = AsyncMock(return_value={"id": "evolved_tree"})
    mock_hta_service.save_tree = AsyncMock()
    completion1 = await completion_processor.process(
        task_id=result1["tasks"][0]["id"],
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    assert completion1["success"]  # Base operation should succeed despite HTA failure
    
    # HTA works for second completion
    mock_hta_service.load_tree = AsyncMock(return_value={"id": "test_tree"})
    mock_hta_service.evolve_tree = AsyncMock(return_value={"id": "evolved_tree"})
    mock_hta_service.save_tree = AsyncMock()
    mock_hta_service.update_node_status = AsyncMock(return_value=True)
    completion2 = await completion_processor.process(
        task_id=result2["tasks"][0]["id"],
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    assert completion2["success"]

# --- Additional Component Tests ---

@pytest.mark.asyncio
async def test_sentiment_analysis_and_metrics(
    mock_llm_client,
    mock_task_engine,
    mock_snapshot
):
    """Test sentiment analysis and its impact on capacity/shadow metrics."""
    from unittest.mock import AsyncMock
    # Set up mock sentiment engine
    mock_sentiment_engine = Mock(spec=SecretSauceSentimentEngineHybrid)
    
    # Set up mock snapshot with initial values
    mock_snapshot.capacity = 0.5
    mock_snapshot.shadow_score = 0.5
    mock_snapshot.core_state = {"hta_tree": {}}
    
    # Set up mock task engine
    mock_task_engine.get_next_step.return_value = {
        "tasks": [
            {
                "id": "test_task_1",
                "title": "Test Task 1",
                "description": "Test Description 1",
                "priority": 5,
                "magnitude": 5,
                "status": "incomplete",
                "hta_node_id": "test_node_1"
            }
        ],
        "fallback_task": None
    }
    
    # Set up mock sentiment analysis
    async def analyze_with_score(*args, **kwargs):
        return SentimentOutput(score=0.8)  # High positive sentiment
    
    mock_sentiment_engine.analyze_emotional_field = AsyncMock(side_effect=analyze_with_score)
    
    mock_sentiment = Mock()
    mock_sentiment.analyze_emotional_field = Mock()
    mock_pce = Mock()
    mock_pce.update_signals_from_reflection = Mock()
    mock_narrative = Mock()
    mock_narrative.determine_narrative_mode = Mock()
    processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine,
        sentiment_engine=mock_sentiment_engine,
        practical_consequence_engine=mock_pce,
        narrative_engine=mock_narrative
    )
    
    original_capacity = mock_snapshot.capacity
    
    with patch('forest_app.core.feature_flags.is_enabled', return_value=True):
        result = await processor.process("Very positive reflection", mock_snapshot)
    
    assert mock_snapshot.capacity > original_capacity
    assert mock_sentiment_engine.analyze_emotional_field.called
    
    # Test error handling
    mock_sentiment_engine.analyze_emotional_field = AsyncMock(side_effect=Exception("Sentiment analysis failed"))
    result_error = await processor.process("Test reflection", mock_snapshot)
    assert isinstance(result_error["tasks"], list)  # Should still generate tasks despite sentiment error

@pytest.mark.asyncio
async def test_narrative_modes_and_style(
    mock_llm_client,
    mock_task_engine,
    mock_snapshot
):
    """Test narrative mode transitions and style application."""
    from unittest.mock import Mock
    mock_narrative_engine = Mock(spec=NarrativeModesEngine)
    mock_narrative_engine.determine_narrative_mode = Mock(
        return_value={"style_directive": "poetic"}
    )
    
    # Set up mock task engine
    mock_task_engine.get_next_step.return_value = {
        "tasks": [
            {
                "id": "test_task_1",
                "title": "Test Task 1",
                "description": "Test Description 1",
                "priority": 5,
                "magnitude": 5,
                "status": "incomplete",
                "hta_node_id": "test_node_1"
            }
        ],
        "fallback_task": None
    }
    
    # Set up mock LLM client
    mock_llm_client.generate.return_value = {
        "narrative": "Test narrative",
        "task": None
    }
    
    mock_sentiment = Mock()
    mock_sentiment.analyze_emotional_field = Mock()
    mock_pce = Mock()
    mock_pce.update_signals_from_reflection = Mock()
    reflection_processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine,
        sentiment_engine=mock_sentiment,
        practical_consequence_engine=mock_pce,
        narrative_engine=mock_narrative_engine
    )
    
    with patch('forest_app.core.feature_flags.is_enabled', return_value=True):
        result = await reflection_processor.process("Test reflection", mock_snapshot)
    
    assert mock_narrative_engine.determine_narrative_mode.called
    # Verify style directive was passed to LLM
    assert any("poetic" in str(args) for args in mock_llm_client.generate.call_args_list)
    
    # Test error handling
    mock_narrative_engine.determine_narrative_mode = Mock(side_effect=Exception("Narrative mode error"))
    result_error = await reflection_processor.process("Test reflection", mock_snapshot)
    assert isinstance(result_error["tasks"], list)  # Should still work without narrative mode

@pytest.mark.asyncio
async def test_soft_deadline_scheduling(
    mock_llm_client,
    mock_task_engine,
    mock_snapshot
):
    """Test soft deadline scheduling for tasks."""
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('forest_app.core.processors.reflection_processor')
    logger.setLevel(logging.DEBUG)

    # Set up mock task engine and LLM client to return the same test_task
    test_task = {
        "id": "test_task_1",
        "title": "Test Task",
        "description": "Test Description",
        "priority": 5,
        "magnitude": 5,
        "status": "incomplete",
        "hta_node_id": "test_node"
    }
    mock_task_engine.get_next_step.return_value = {
        "tasks": [copy.deepcopy(test_task)],
        "fallback_task": None,
        "primary_task": copy.deepcopy(test_task)
    }
    mock_llm_client.generate.return_value = {
        "narrative": "Test narrative",
        "task": copy.deepcopy(test_task)
    }

    mock_sentiment = Mock()
    mock_sentiment.analyze_emotional_field = Mock()
    mock_pce = Mock()
    mock_pce.update_signals_from_reflection = Mock()
    mock_narrative = Mock()
    mock_narrative.determine_narrative_mode = Mock()
    reflection_processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine,
        sentiment_engine=mock_sentiment,
        practical_consequence_engine=mock_pce,
        narrative_engine=mock_narrative
    )

    # Use the mock_snapshot fixture (Mock(spec=MemorySnapshot))
    mock_snapshot.current_batch_reflections = []
    mock_snapshot.to_dict.return_value["current_batch_reflections"] = []
    mock_snapshot.current_path = "structured"
    mock_snapshot.to_dict.return_value["current_path"] = "structured"
    # Ensure mock_task_engine.get_next_step returns tasks only if current_path is 'structured'
    def get_next_step_conditional(snapshot_dict):
        if snapshot_dict.get('current_path', None) == "structured":
            return {
                "tasks": [copy.deepcopy(test_task)],
                "fallback_task": None,
                "primary_task": copy.deepcopy(test_task)
            }
        else:
            return {
                "tasks": [],
                "fallback_task": None,
                "primary_task": None
            }
    mock_task_engine.get_next_step.side_effect = get_next_step_conditional

    # Mock all feature flag checks
    with patch('forest_app.core.processors.reflection_processor.is_enabled', return_value=True) as mock_is_enabled, \
         patch('forest_app.core.feature_flags.is_enabled', return_value=True) as mock_is_enabled2, \
         patch('forest_app.core.soft_deadline_manager.is_enabled', return_value=True) as mock_is_enabled3, \
         patch('forest_app.core.processors.reflection_processor.schedule_soft_deadlines') as mock_schedule, \
         patch('forest_app.core.processors.reflection_processor.prune_context', return_value={"shadow_score": 0.5, "capacity": 1.0, "magnitude": 5.0, "current_path": "structured"}) as mock_prune:

        # Configure mock_schedule to return the tasks with deadlines
        def mock_schedule_side_effect(snapshot, tasks, **kwargs):
            for task in tasks:
                task["soft_deadline"] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
            return list(tasks)

        mock_schedule.side_effect = mock_schedule_side_effect

        # Process the reflection
        result = await reflection_processor.process("Test reflection", mock_snapshot)

        # Verify the soft deadline scheduler was called
        assert mock_schedule.called

        # Verify all components were called
        assert mock_sentiment.analyze_emotional_field.called
        assert mock_pce.update_signals_from_reflection.called
        assert mock_narrative.determine_narrative_mode.called

        # Complete the generated task
        from forest_app.core.processors.completion_processor import CompletionProcessor
        completion_processor = CompletionProcessor(
            hta_service=Mock(spec=HTAService),
            task_engine=mock_task_engine
        )
        mock_db = Mock(spec=Session)
        mock_logger = Mock(spec=TaskFootprintLogger)
        mock_logger.log_task_completion = AsyncMock()  # Add async mock for task completion

        for task in result["tasks"]:
            mock_snapshot.task_backlog.append(task)  # Add task to backlog
            mock_db = Mock(spec=Session)
            mock_logger = Mock(spec=TaskFootprintLogger)
            mock_logger.log_task_completion = AsyncMock()

            completion_result = await completion_processor.process(
                task_id=task["id"],
                success=True,
                snapshot=mock_snapshot,
                db=mock_db,
                task_logger=mock_logger
            )
            assert completion_result["success"]
        
        # Verify final state
        assert mock_task_engine.get_next_step.called
        # Verify snapshot updates
        assert len(mock_snapshot.current_frontier_batch_ids) == 0  # All tasks (even failed) are removed

@pytest.mark.asyncio
async def test_practical_consequence_updates(
    mock_llm_client,
    mock_task_engine,
    mock_snapshot
):
    """Test practical consequence signal updates from reflections."""
    mock_pce = Mock(spec=PracticalConsequenceEngine)
    mock_pce.update_signals_from_reflection = Mock()

    mock_sentiment = Mock()
    mock_sentiment.analyze_emotional_field = Mock()
    mock_narrative = Mock()
    mock_narrative.determine_narrative_mode = Mock()
    reflection_processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine,
        sentiment_engine=mock_sentiment,
        practical_consequence_engine=mock_pce,
        narrative_engine=mock_narrative
    )

    test_reflection = "Test reflection"
    
    # Enable the practical consequence feature for this test
    with patch('forest_app.core.processors.reflection_processor.is_enabled', return_value=True) as mock_is_enabled:
        result = await reflection_processor.process(test_reflection, mock_snapshot)

        # Verify the feature flag was checked
        mock_is_enabled.assert_any_call(Feature.PRACTICAL_CONSEQUENCE)
        # Verify the PCE was called
        assert mock_pce.update_signals_from_reflection.called
        assert mock_pce.update_signals_from_reflection.call_args[0][0] == test_reflection

@pytest.mark.asyncio
async def test_full_integration_with_all_components(
    mock_llm_client,
    mock_task_engine,
    mock_hta_service,
    mock_snapshot
):
    """Test complete integration of all components in the core loop."""
    mock_sentiment = Mock()
    mock_sentiment.analyze_emotional_field = Mock()
    
    mock_pce = Mock()
    mock_pce.update_signals_from_reflection = Mock()
    
    mock_narrative = Mock()
    mock_narrative.determine_narrative_mode = Mock()

    # Set up mock task engine to return a task
    test_task = {
        "id": "test_task_1",
        "title": "Test Task",
        "description": "Test Description",
        "priority": 5,
        "magnitude": 5,
        "status": "incomplete",
        "hta_node_id": "test_node"
    }
    mock_task_engine.get_next_step.side_effect = lambda *args, **kwargs: {
        "tasks": [test_task],
        "fallback_task": None,
        "primary_task": test_task
    }

    # Set up mock snapshot with estimated completion date
    future_date = datetime.now(timezone.utc) + timedelta(days=7)
    mock_snapshot.estimated_completion_date = future_date.isoformat()
    mock_snapshot.to_dict.return_value = {
        "estimated_completion_date": future_date.isoformat(),
        "current_path": "structured",
        "shadow_score": 0.5,
        "capacity": 0.5,
        "magnitude": 5.0
    }
    mock_snapshot.current_path = "structured"
    mock_snapshot.task_backlog = []
    mock_snapshot.conversation_history = []

    # Set up mock snapshot to properly handle list operations
    mock_reflections = []
    type(mock_snapshot).current_batch_reflections = PropertyMock(return_value=mock_reflections)

    reflection_processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine,
        sentiment_engine=mock_sentiment,
        practical_consequence_engine=mock_pce,
        narrative_engine=mock_narrative
    )
    
    completion_processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine
    )
    
    # Process reflection with all components active
    from forest_app.core.feature_flags import Feature
    with patch('forest_app.core.feature_flags.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE, Feature.SOFT_DEADLINES, Feature.SENTIMENT_ANALYSIS, Feature.NARRATIVE_MODES, Feature.PRACTICAL_CONSEQUENCE]), \
         patch('forest_app.core.processors.reflection_processor.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE, Feature.SOFT_DEADLINES, Feature.SENTIMENT_ANALYSIS, Feature.NARRATIVE_MODES, Feature.PRACTICAL_CONSEQUENCE]), \
         patch('forest_app.core.processors.completion_processor.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE, Feature.SOFT_DEADLINES, Feature.SENTIMENT_ANALYSIS, Feature.NARRATIVE_MODES, Feature.PRACTICAL_CONSEQUENCE]), \
         patch('forest_app.core.soft_deadline_manager.is_enabled', side_effect=lambda x: True), \
         patch('forest_app.core.processors.reflection_processor.schedule_soft_deadlines') as mock_schedule, \
         patch('forest_app.core.processors.reflection_processor.prune_context', return_value={"shadow_score": 0.5, "capacity": 1.0, "magnitude": 5.0, "current_path": "structured"}) as mock_prune:

        # Set up mock prune_context to return a valid dictionary
        mock_prune.return_value = {
            "shadow_score": 0.5,
            "capacity": 0.5,
            "magnitude": 5.0,
            "current_path": "structured"
        }

        # Process reflection
        result = await reflection_processor.process(
            user_input="I need to focus on my work",
            snapshot=mock_snapshot
        )

        # Verify soft deadline scheduling was called
        assert mock_schedule.called

        # Verify all components were called
        assert mock_sentiment.analyze_emotional_field.called
        assert mock_pce.update_signals_from_reflection.called
        assert mock_narrative.determine_narrative_mode.called

        # Complete the generated task
        mock_db = Mock(spec=Session)
        mock_logger = Mock(spec=TaskFootprintLogger)
        mock_logger.log_task_completion = AsyncMock()  # Add async mock for task completion

        for task in result["tasks"]:
            mock_snapshot.task_backlog.append(task)  # Add task to backlog
            mock_db = Mock(spec=Session)
            mock_logger = Mock(spec=TaskFootprintLogger)
            mock_logger.log_task_completion = AsyncMock()

            completion_result = await completion_processor.process(
                task_id=task["id"],
                success=True,
                snapshot=mock_snapshot,
                db=mock_db,
                task_logger=mock_logger
            )
            assert completion_result["success"]
        
        # Verify final state
        assert mock_task_engine.get_next_step.called
        # Verify snapshot updates
        assert len(mock_snapshot.current_frontier_batch_ids) == 0  # All tasks (even failed) are removed

# --- XP Mastery Tests ---

@pytest.mark.asyncio
async def test_xp_calculation_basic(mock_xp_engine):
    """Test basic XP calculation with different task parameters."""
    tasks = [
        {
            "priority": "low",
            "complexity": "simple",
            "time_spent": 30,
            "dependencies": []
        },
        {
            "priority": "high",
            "complexity": "complex",
            "time_spent": 60,
            "dependencies": ["task1", "task2"]
        }
    ]
    
    # Test low priority simple task
    xp1 = mock_xp_engine.calculate_xp_gain(tasks[0])
    assert xp1 == 5.0  # base(10) * priority(0.5) * complexity(1.0)
    
    # Test high priority complex task
    xp2 = mock_xp_engine.calculate_xp_gain(tasks[1])
    assert xp2 == 30.0  # base(10) * priority(1.5) * complexity(2.0)

@pytest.mark.asyncio
async def test_xp_stage_progression(mock_xp_engine):
    """Test XP stage progression and threshold handling."""
    # Start at Novice
    initial_stage = mock_xp_engine.get_current_stage()
    assert initial_stage["name"] == "Novice"
    
    # Progress to Apprentice
    mock_xp_engine.current_xp = 150
    apprentice_stage = mock_xp_engine.get_current_stage()
    assert apprentice_stage["name"] == "Apprentice"
    
    # Test stage boundary
    mock_xp_engine.current_xp = 100  # Exact threshold
    boundary_stage = mock_xp_engine.get_current_stage()
    assert boundary_stage["name"] == "Apprentice"

@pytest.mark.asyncio
async def test_completion_processor_xp_integration(
    mock_hta_service,
    mock_task_engine,
    mock_xp_engine,
    mock_snapshot
):
    """Test integration between CompletionProcessor and XP system."""
    processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine,
        xp_engine=mock_xp_engine
    )
    
    # Add a task to the snapshot
    test_task = {
        "id": "test_task_1",
        "title": "Test Task",
        "priority": "high",
        "complexity": "complex",
        "time_spent": 45,
        "dependencies": ["dep1"],
        "hta_node_id": "test_node"
    }
    mock_snapshot.task_backlog = [test_task]
    
    # Process successful completion
    mock_db = Mock(spec=Session)
    mock_logger = Mock(spec=TaskFootprintLogger)
    mock_logger.log_task_completion = AsyncMock()

    result = await processor.process(
        task_id="test_task_1",
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    
    assert result["success"] is True
    assert "xp_update" in result
    assert result["xp_update"]["xp_gained"] > 0
    assert result["xp_update"]["current_stage"] == mock_xp_engine.get_current_stage()["name"]

@pytest.mark.asyncio
async def test_xp_error_handling(
    mock_hta_service,
    mock_task_engine,
    mock_xp_engine,
    mock_snapshot
):
    """Test error handling in XP calculations and updates."""
    processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine,
        xp_engine=mock_xp_engine
    )
    
    # Test with invalid task format
    mock_snapshot.task_backlog = [{"id": "test_task_1"}]  # Missing required fields

    mock_db = Mock(spec=Session)
    mock_logger = Mock(spec=TaskFootprintLogger)
    mock_logger.log_task_completion = AsyncMock()

    result = await processor.process(
        task_id="test_task_1",
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    
    assert result["success"] is True
    assert result["xp_update"]["xp_gained"] == 10.0  # Should fall back to base XP
    
    # Test with XP calculation failure
    mock_xp_engine.calculate_xp_gain = Mock(side_effect=ValueError("XP calculation error"))
    
    # Reset the task in the backlog since it was removed in the previous process call
    mock_snapshot.task_backlog = [{"id": "test_task_1"}]

    result = await processor.process(
        task_id="test_task_1",
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,  # Reuse the same mock_db
        task_logger=mock_logger  # Reuse the same mock_logger
    )
    
    assert result["success"] is True
    assert "xp_update" in result
    assert result["xp_update"]["xp_gained"] == 0
    assert "error" in result["xp_update"]
    assert isinstance(result["xp_update"]["error"], str)

@pytest.mark.asyncio
async def test_challenge_generation(mock_xp_engine, mock_snapshot):
    """Test mastery challenge generation at stage transitions."""
    # Setup initial stage
    mock_xp_engine.current_xp = 0
    initial_stage = mock_xp_engine.get_current_stage()
    
    # Generate challenge
    challenge = await mock_xp_engine.generate_challenge_content(
        initial_stage,
        mock_snapshot.to_dict()
    )
    
    assert challenge["stage"] == "Novice"
    assert challenge["type"] == "Basic"
    assert "content" in challenge
    assert "xp_required" in challenge
    assert "current_xp" in challenge
    
    # Test challenge generation with invalid snapshot
    mock_xp_engine.validate_snapshot = Mock(return_value=False)
    challenge = await mock_xp_engine.generate_challenge_content(
        initial_stage,
        {"invalid": "snapshot"}
    )
    
    assert challenge["stage"] == mock_xp_engine.get_current_stage()["name"]

@pytest.mark.asyncio
async def test_legacy_xp_mastery():
    """Test legacy XP Mastery class functionality."""
    from forest_app.modules.resource.xp_mastery import XPMastery
    
    # Initialize legacy class
    xp_mastery = XPMastery()
    
    # Test initial state
    initial_state = xp_mastery.to_dict()
    assert initial_state["current_xp"] == 0.0
    assert initial_state["completed_challenges"] == []
    assert initial_state["current_stage"]["name"] == "Novice"
    
    # Test state update
    update_data = {
        "current_xp": 150.0,
        "completed_challenges": ["challenge1", "challenge2"]
    }
    xp_mastery.update_from_dict(update_data)
    
    updated_state = xp_mastery.to_dict()
    assert updated_state["current_xp"] == 150.0
    assert len(updated_state["completed_challenges"]) == 2
    assert updated_state["current_stage"]["name"] == "Apprentice"
    
    # Test invalid state update
    invalid_data = {
        "current_xp": "invalid",
        "completed_challenges": None
    }
    xp_mastery.update_from_dict(invalid_data)
    
    # Should reset to initial state on error
    error_state = xp_mastery.to_dict()
    assert error_state["current_xp"] == 0.0
    assert error_state["completed_challenges"] == []
    assert error_state["current_stage"]["name"] == "Novice"

@pytest.mark.asyncio
async def test_xp_stage_check(mock_xp_engine, mock_snapshot):
    """Test XP stage checking and progression information."""
    from forest_app.modules.resource.xp_mastery import XPMastery

    # Set up mock snapshot with required fields
    mock_snapshot.to_dict.return_value = {
        "user_id": "test_user",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "core_state": {
            "xp_mastery": {
                "current_xp": 50.0,
                "completed_challenges": ["challenge1"],
                "current_stage": {"name": "Novice", "min_xp": 0, "max_xp": 100, "type": "Basic"}
            }
        }
    }

    # Create XPMastery instance with mock engine
    xp_mastery = XPMastery()
    xp_mastery.engine = mock_xp_engine

    # Set up mock engine state
    mock_xp_engine.current_xp = 50.0
    mock_xp_engine.completed_challenges = ["challenge1"]
    mock_xp_engine.check_stage = Mock(return_value={
        "current_stage": "Novice",
        "current_xp": 50.0,
        "next_stage": "Intermediate",
        "xp_needed": 50.0,
        "completed_challenges": ["challenge1"]
    })

    # Test with valid snapshot
    progression_info = xp_mastery.check_xp_stage(mock_snapshot.to_dict())
    assert "current_stage" in progression_info
    assert progression_info["current_stage"] == "Novice"
    assert "current_xp" in progression_info
    assert "next_stage" in progression_info
    assert "xp_needed" in progression_info
    assert "completed_challenges" in progression_info
    assert len(progression_info["completed_challenges"]) == 1

@pytest.mark.asyncio
async def test_full_xp_lifecycle(
    mock_hta_service,
    mock_task_engine,
    mock_snapshot
):
    """Test complete XP lifecycle with real XP Mastery engine."""
    from forest_app.modules.resource.xp_mastery import XPMasteryEngine
    
    # Initialize with real XP engine
    xp_engine = XPMasteryEngine()
    processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine,
        xp_engine=xp_engine
    )
    
    # Create test tasks with varying complexity
    tasks = [
        {
            "id": "task1",
            "title": "Simple Task",
            "priority": "low",
            "complexity": "simple",
            "time_spent": 30,
            "dependencies": [],
            "hta_node_id": "node1"
        },
        {
            "id": "task2",
            "title": "Complex Task",
            "priority": "high",
            "complexity": "complex",
            "time_spent": 60,
            "dependencies": ["task1"],
            "hta_node_id": "node2"
        }
    ]
    
    mock_snapshot.task_backlog = tasks
    initial_stage = xp_engine.get_current_stage()
    
    # Complete first task
    mock_db = Mock(spec=Session)
    mock_logger = Mock(spec=TaskFootprintLogger)
    mock_logger.log_task_completion = AsyncMock()

    result1 = await processor.process(
        task_id="task1",
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    
    assert result1["success"] is True
    assert result1["xp_update"]["xp_gained"] > 0
    assert result1["xp_update"]["current_stage"] == initial_stage["name"]
    
    # Complete second task
    result2 = await processor.process(
        task_id="task2",
        success=True,
        snapshot=mock_snapshot,
        db=mock_db,
        task_logger=mock_logger
    )
    
    assert result2["success"] is True
    assert result2["xp_update"]["xp_gained"] > result1["xp_update"]["xp_gained"]
    
    # Verify final state
    final_stage = xp_engine.get_current_stage()
    assert xp_engine.current_xp == (
        result1["xp_update"]["xp_gained"] + 
        result2["xp_update"]["xp_gained"]
    )
    
    # Test stage progression if XP threshold reached
    if xp_engine.current_xp >= 100:
        assert final_stage["name"] == "Apprentice"
        assert "challenge" in result2["xp_update"]
        assert result2["xp_update"]["stage_changed"] is True

@pytest.mark.asyncio
async def test_core_loop_onboarding_to_hta_evolution(
    mock_llm_client,
    mock_task_engine,
    mock_hta_service,
    mock_snapshot
):
    """
    Isolated test for the core loop: onboarding, context setting, HTA generation,
    task assignment, task completion, and HTA tree evolution.
    """
    from unittest.mock import patch
    # Patch is_enabled in feature_flags, reflection_processor, and completion_processor to only enable core features
    from forest_app.core.feature_flags import Feature
    with patch('forest_app.core.feature_flags.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE]), \
         patch('forest_app.core.processors.reflection_processor.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE]), \
         patch('forest_app.core.processors.completion_processor.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE]):
        # 1. Onboarding/context setting (simulate user input)
        from forest_app.snapshot.snapshot import MemorySnapshot
        snapshot = MemorySnapshot()
        snapshot.current_path = "structured"
        snapshot.activated_state = {"activated": True}
        snapshot.core_state = {"hta_tree": {"root": {"id": "root_testuuid", "title": "Run a marathon", "description": "Plan for marathon", "priority": 0.5, "depends_on": [], "estimated_energy": "medium", "estimated_time": "medium", "linked_tasks": [], "is_milestone": True, "rationale": "", "status_suggestion": "pending", "children": []}}}
        snapshot.task_backlog = []
        snapshot.current_batch_reflections = []
        snapshot.current_frontier_batch_ids = []

        # 2. HTA Generation (simulate LLM/HTA service)
        from forest_app.hta_tree.hta_tree import HTATree
        mock_hta_service.load_tree = AsyncMock(return_value=HTATree())
        mock_hta_service.evolve_tree = AsyncMock(return_value={"id": "evolved_root", "nodes": [{"id": "evolved_node", "status": "incomplete"}]})
        mock_hta_service.save_tree = AsyncMock(return_value=True)

        # 3. Task Assignment (simulate task engine)
        mock_task_engine.get_next_step.side_effect = lambda *args, **kwargs: {
            "tasks": [
                {
                    "id": "test_task",
                    "title": "Test Task",
                    "description": "Test Description",
                    "priority": 5,
                    "magnitude": 5,
                    "status": "incomplete",
                    "hta_node_id": "test_node"
                }
            ],
            "fallback_task": None
        }

        # 4. Mock LLM client to return a task with id 'test_task'
        mock_llm_client.generate = AsyncMock(return_value={
            "narrative": "Test narrative",
            "task": {
                "id": "test_task",
                "title": "Test Task",
                "description": "Test Description",
                "priority": 5,
                "magnitude": 5,
                "status": "incomplete",
                "hta_node_id": "test_node"
            }
        })

        # 5. ReflectionProcessor and CompletionProcessor setup
        mock_sentiment = Mock()
        mock_sentiment.analyze_emotional_field = Mock()
        mock_pce = Mock()
        mock_pce.update_signals_from_reflection = Mock()
        mock_narrative = Mock()
        mock_narrative.determine_narrative_mode = Mock()
        reflection_processor = ReflectionProcessor(
            llm_client=mock_llm_client,
            task_engine=mock_task_engine,
            sentiment_engine=mock_sentiment,
            practical_consequence_engine=mock_pce,
            narrative_engine=mock_narrative
        )
        completion_processor = CompletionProcessor(
            hta_service=mock_hta_service,
            task_engine=mock_task_engine
        )

        # 6. Simulate user reflection and task assignment
        reflection = "My goal is to finish a project."
        result = await reflection_processor.process(reflection, snapshot)
        assert "tasks" in result and len(result["tasks"]) == 1
        task = result["tasks"][0]
        assert task["id"] == "test_task"

        # 7. Simulate task completion and HTA tree evolution
        snapshot.current_frontier_batch_ids.append(task["id"])
        snapshot.task_backlog.append(task)
        print("Before process, current_frontier_batch_ids:", snapshot.current_frontier_batch_ids)
        mock_db = Mock(spec=Session)
        mock_logger = Mock(spec=TaskFootprintLogger)
        mock_logger.log_task_completion = AsyncMock()

        completion_result = await completion_processor.process(
            task_id=task["id"],
            success=True,
            snapshot=snapshot,
            db=mock_db,
            task_logger=mock_logger
        )
        print("After process, current_frontier_batch_ids:", snapshot.current_frontier_batch_ids)
        assert completion_result["success"]
        assert mock_task_engine.get_next_step.called
        # Verify snapshot updates
        assert len(snapshot.current_frontier_batch_ids) == 0  # All tasks (even failed) are removed

@pytest.mark.asyncio
async def test_core_loop_full_simulation_diagnostic(mock_llm_client, mock_task_engine, mock_hta_service):
    """
    Full simulation and diagnostic test of the core loop with a comprehensive report.
    """
    from forest_app.snapshot.snapshot import MemorySnapshot
    from forest_app.hta_tree.hta_tree import HTATree
    from forest_app.core.feature_flags import Feature
    from unittest.mock import Mock, AsyncMock, patch

    # Setup
    snapshot = MemorySnapshot()
    snapshot.current_path = "structured"
    snapshot.activated_state = {"activated": True}
    snapshot.core_state = {"hta_tree": {"root": {"id": "root_testuuid", "title": "Run a marathon", "description": "Plan for marathon", "priority": 0.5, "depends_on": [], "estimated_energy": "medium", "estimated_time": "medium", "linked_tasks": [], "is_milestone": True, "rationale": "", "status_suggestion": "pending", "children": []}}}
    snapshot.task_backlog = []
    snapshot.current_batch_reflections = []
    snapshot.current_frontier_batch_ids = []

    # Mock HTA service
    mock_hta_service.load_tree = AsyncMock(return_value=HTATree())
    mock_hta_service.evolve_tree = AsyncMock(return_value=HTATree())
    mock_hta_service.save_tree = AsyncMock(return_value=True)

    # Mock task engine to return a batch of tasks
    batch_tasks = [
        {
            "id": f"test_task_{i}",
            "title": f"Test Task {i}",
            "description": f"Test Description {i}",
            "priority": 5,
            "magnitude": 5,
            "status": "incomplete",
            "hta_node_id": f"test_node_{i}"
        }
        for i in range(3)
    ]
    mock_task_engine.get_next_step.side_effect = lambda *args, **kwargs: {
        "tasks": batch_tasks,
        "fallback_task": None
    }

    # Mock LLM client to return a narrative and a batch of tasks
    mock_llm_client.generate = AsyncMock(return_value={
        "narrative": "Test narrative",
        "task": None
    })

    # Processors
    mock_sentiment = Mock()
    mock_sentiment.analyze_emotional_field = Mock()
    mock_pce = Mock()
    mock_pce.update_signals_from_reflection = Mock()
    mock_narrative = Mock()
    mock_narrative.determine_narrative_mode = Mock()
    reflection_processor = ReflectionProcessor(
        llm_client=mock_llm_client,
        task_engine=mock_task_engine,
        sentiment_engine=mock_sentiment,
        practical_consequence_engine=mock_pce,
        narrative_engine=mock_narrative
    )
    completion_processor = CompletionProcessor(
        hta_service=mock_hta_service,
        task_engine=mock_task_engine
    )

    # Patch feature flags to enable core features
    with patch('forest_app.core.feature_flags.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE, Feature.SOFT_DEADLINES, Feature.SENTIMENT_ANALYSIS, Feature.NARRATIVE_MODES, Feature.PRACTICAL_CONSEQUENCE]), \
         patch('forest_app.core.processors.reflection_processor.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE, Feature.SOFT_DEADLINES, Feature.SENTIMENT_ANALYSIS, Feature.NARRATIVE_MODES, Feature.PRACTICAL_CONSEQUENCE]), \
         patch('forest_app.core.processors.completion_processor.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE, Feature.SOFT_DEADLINES, Feature.SENTIMENT_ANALYSIS, Feature.NARRATIVE_MODES, Feature.PRACTICAL_CONSEQUENCE]), \
         patch('forest_app.core.soft_deadline_manager.is_enabled', side_effect=lambda x: True), \
         patch('forest_app.core.processors.reflection_processor.schedule_soft_deadlines') as mock_schedule, \
         patch('forest_app.core.processors.reflection_processor.prune_context', return_value={"shadow_score": 0.5, "capacity": 1.0, "magnitude": 5.0, "current_path": "structured"}) as mock_prune:

        # Set up mock prune_context to return a valid dictionary
        mock_prune.return_value = {
            "shadow_score": 0.5,
            "capacity": 0.5,
            "magnitude": 5.0,
            "current_path": "structured"
        }

        # Process reflection
        result = await reflection_processor.process(
            user_input="I need to focus on my work",
            snapshot=snapshot
        )

        # Verify soft deadline scheduling was called
        assert mock_schedule.called

        # Verify all components were called
        assert mock_sentiment.analyze_emotional_field.called
        assert mock_pce.update_signals_from_reflection.called
        assert mock_narrative.determine_narrative_mode.called

        # Complete the generated task
        mock_db = Mock(spec=Session)
        mock_logger = Mock(spec=TaskFootprintLogger)
        mock_logger.log_task_completion = AsyncMock()  # Add async mock for task completion

        for task in result["tasks"]:
            snapshot.task_backlog.append(task)  # Add task to backlog
            mock_db = Mock(spec=Session)
            mock_logger = Mock(spec=TaskFootprintLogger)
            mock_logger.log_task_completion = AsyncMock()

            completion_result = await completion_processor.process(
                task_id=task["id"],
                success=True,
                snapshot=snapshot,
                db=mock_db,
                task_logger=mock_logger
            )
            assert completion_result["success"]
        
        # Verify final state
        assert mock_task_engine.get_next_step.called
        # Verify snapshot updates
        assert len(snapshot.current_frontier_batch_ids) == 0  # All tasks (even failed) are removed

def test_onboarding_always_issues_first_task():
    """
    Guarantee that after onboarding (goal + context), the system always issues a first task.
    """
    client = TestClient(app)
    class FakeUUID:
        hex = 'testuuid'
        def __str__(self):
            return 'testuuid'
    with patch("uuid.uuid4", return_value=FakeUUID()):
        # Set up goal first
        response_goal = client.post("/onboarding/set_goal", json={"goal_description": "Run a marathon"})
        assert response_goal.status_code == 200
        # Add context
        response_context = client.post("/onboarding/add_context", json={"context_reflection": "I have 6 months to train."})
        assert response_context.status_code == 200
        data = response_context.json()
        assert data["onboarding_status"] == "completed"
        assert data["first_task"] is not None
        assert data["first_task"]["id"] == "reflect_testuuid"
        assert data["first_task"]["title"] == "Deep Reflection Session: Uncovering Insights"

def override_task_engine():
    from unittest.mock import Mock
    mock = Mock(spec=TaskEngine)
    mock.get_next_step.return_value = {
        "base_task": {"id": "reflect_testuuid", "title": "Deep Reflection Session: Uncovering Insights"},
        "tasks": [],
        "fallback_task": {"id": "reflect_testuuid", "title": "Deep Reflection Session: Uncovering Insights"}
    }
    return mock

def override_get_current_active_user():
    from unittest.mock import Mock
    mock_user = Mock()
    mock_user.id = 1
    return mock_user

@pytest.fixture(autouse=True)
def override_dependencies():
    app.dependency_overrides[TaskEngine] = override_task_engine
    app.dependency_overrides[get_current_active_user] = override_get_current_active_user
    yield
    app.dependency_overrides = {} 
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

# Import the classes under test
from forest_app.core.processors import ReflectionProcessor, CompletionProcessor
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.hta_tree.task_engine import TaskEngine
from forest_app.hta_tree.hta_service import HTAService
from forest_app.hta_tree.hta_tree import HTATree
from forest_app.core.logging_tracking import TaskFootprintLogger
from forest_app.integrations.llm import LLMClient
from sqlalchemy.orm import Session
from forest_app.core.feature_flags import Feature

@pytest.mark.asyncio
async def test_isolated_core_loop():
    """
    Fully isolated, robust test for the core loop: onboarding, context setting, HTA generation,
    task assignment, batch task completion, and HTA tree evolution.
    """
    with patch('forest_app.core.feature_flags.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE]), \
         patch('forest_app.core.processors.reflection_processor.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE]), \
         patch('forest_app.core.processors.completion_processor.is_enabled', side_effect=lambda x: x in [Feature.CORE_HTA, Feature.CORE_TASK_ENGINE]):

        # --- Setup snapshot ---
        future_date = datetime.now(timezone.utc) + timedelta(days=7)
        snapshot = MemorySnapshot()
        snapshot.current_path = "structured"
        snapshot.activated_state = {"activated": True}
        snapshot.core_state = {"hta_tree": {}}
        snapshot.task_backlog = []
        snapshot.current_batch_reflections = []
        snapshot.current_frontier_batch_ids = []
        snapshot.conversation_history = []
        snapshot.withering_level = 0.0
        snapshot.estimated_completion_date = future_date.isoformat()
        snapshot.shadow_score = 0.5
        snapshot.capacity = 0.5
        snapshot.magnitude = 5.0

        # --- Mock HTA service ---
        mock_hta_service = AsyncMock(spec=HTAService)
        hta_tree = HTATree()
        mock_hta_service.load_tree = AsyncMock(return_value=hta_tree)
        mock_hta_service.evolve_tree = AsyncMock(return_value=hta_tree)
        mock_hta_service.save_tree = AsyncMock(return_value=True)

        # --- Mock Task Engine ---
        batch_size = 3
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
            for i in range(batch_size)
        ]
        mock_task_engine = Mock(spec=TaskEngine)
        mock_task_engine.get_next_step.return_value = {
            "tasks": batch_tasks,
            "fallback_task": None
        }

        # --- Mock LLM client ---
        mock_llm_client = AsyncMock(spec=LLMClient)
        mock_llm_client.generate = AsyncMock(return_value={
            "narrative": "Test narrative",
            "task": batch_tasks[0]
        })

        # --- Processors ---
        reflection_processor = ReflectionProcessor(
            llm_client=mock_llm_client,
            task_engine=mock_task_engine
        )
        completion_processor = CompletionProcessor(
            hta_service=mock_hta_service,
            task_engine=mock_task_engine
        )

        # --- Simulate user reflection and task assignment ---
        reflection = "My goal is to finish a project."
        result = await reflection_processor.process(reflection, snapshot)
        assert "tasks" in result and len(result["tasks"]) == batch_size, "Task batch size mismatch"
        tasks = result["tasks"]
        for i, task in enumerate(tasks):
            assert task["id"] == f"test_task_{i}", f"Task ID mismatch at index {i}"

        # --- Simulate batch task completion and HTA tree evolution ---
        snapshot.current_frontier_batch_ids = [task["id"] for task in tasks]
        snapshot.task_backlog = list(tasks)
        mock_db = Mock(spec=Session)
        mock_logger = Mock(spec=TaskFootprintLogger)
        mock_logger.log_task_completion = AsyncMock()

        # Complete all but the last task, evolution should NOT be called yet
        for i, task in enumerate(tasks[:-1]):
            result = await completion_processor.process(
                task_id=task["id"],
                success=True,
                snapshot=snapshot,
                db=mock_db,
                task_logger=mock_logger
            )
            assert result["success"], f"Task {i} completion failed"
            assert not mock_hta_service.evolve_tree.called, f"HTA evolution triggered too early at task {i}"
            assert task["id"] not in snapshot.current_frontier_batch_ids, f"Task {i} not removed from batch"
            # Optionally, check batch size
            assert len(snapshot.current_frontier_batch_ids) == batch_size - (i + 1), f"Batch size incorrect after task {i}"

        # Complete the last task, evolution SHOULD be called
        last_task = tasks[-1]
        result = await completion_processor.process(
            task_id=last_task["id"],
            success=True,
            snapshot=snapshot,
            db=mock_db,
            task_logger=mock_logger
        )
        assert result["success"], "Last task completion failed"
        assert mock_hta_service.evolve_tree.called, "HTA evolution not triggered after last task"
        assert len(snapshot.current_frontier_batch_ids) == 0, "Batch not empty after all tasks completed" 
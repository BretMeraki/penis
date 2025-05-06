import pytest
import asyncio
from forest_app.hta_tree.hta_service import HTAService
from forest_app.hta_tree.hta_tree import HTATree
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.hta_tree.hta_models import HTAResponseModel, HTANodeModel
from forest_app.integrations.llm import HTAEvolveResponse
import re
import json

# --- Helper: Minimal HTAValidationModel/HTANodeModel mocks ---
class DummyHTANodeModel:
    def __init__(self):
        self.id = "root_1"
        self.title = "Root Goal"
        self.description = "Root node"
        self.priority = 1.0
        self.depends_on = []
        self.estimated_energy = "medium"
        self.estimated_time = "medium"
        self.linked_tasks = []
        self.is_milestone = True
        self.rationale = ""
        self.status_suggestion = "pending"
        self.children = [
            DummyChildNode()
        ]
    def model_dump(self, mode=None):
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
            "children": [c.model_dump() for c in self.children],
        }

class DummyChildNode:
    def __init__(self):
        self.id = "child_1"
        self.title = "First Task"
        self.description = "Do something important"
        self.priority = 0.8
        self.depends_on = []
        self.estimated_energy = "medium"
        self.estimated_time = "medium"
        self.linked_tasks = []
        self.is_milestone = False
        self.rationale = ""
        self.status_suggestion = "pending"
        self.children = []
    def model_dump(self, mode=None):
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
            "children": [],
        }

class DummyHTAValidationModel:
    def __init__(self):
        self.hta_root = DummyHTANodeModel()
    def model_dump(self, mode=None):
        return {"hta_root": self.hta_root.model_dump()}

class DummyHTAValidationModelEvolved:
    def __init__(self):
        self.hta_root = DummyHTANodeModel()
        # Add a new child to simulate evolution
        self.hta_root.children.append(DummyEvolvedChildNode())
    def model_dump(self, mode=None):
        return {"hta_root": self.hta_root.model_dump()}

class DummyEvolvedChildNode(DummyChildNode):
    def __init__(self):
        super().__init__()
        self.id = "child_2"
        self.title = "Second Task"
        self.description = "Do something new"
        self.priority = 0.7

class DummyLLMClient:
    async def generate(self, *args, **kwargs):
        # Extract the expected root node ID from the prompt
        prompt = ""
        if "prompt_parts" in kwargs and kwargs["prompt_parts"]:
            prompt = kwargs["prompt_parts"][0]
        elif args:
            prompt = args[0]
        match = re.search(r"Root MUST use: '([^']+)'", prompt)
        root_id = match.group(1) if match else "root_1"
        child_node = HTANodeModel(
            id="child_1",
            title="First Task",
            description="Do something important",
            priority=0.8,
            depends_on=[],
            estimated_energy="medium",
            estimated_time="medium",
            linked_tasks=[],
            is_milestone=False,
            rationale="",
            status_suggestion="pending",
            children=[],
        )
        root_node = HTANodeModel(
            id=root_id,
            title="Root Goal",
            description="Root node",
            priority=1.0,
            depends_on=[],
            estimated_energy="medium",
            estimated_time="medium",
            linked_tasks=[],
            is_milestone=True,
            rationale="",
            status_suggestion="pending",
            children=[child_node],
        )
        return HTAResponseModel(hta_root=root_node)

    async def request_hta_evolution(self, *args, **kwargs):
        # Simulate evolution by adding a new child node
        evolved_child = HTANodeModel(
            id="child_2",
            title="Second Task",
            description="Do something new",
            priority=0.7,
            depends_on=[],
            estimated_energy="medium",
            estimated_time="medium",
            linked_tasks=[],
            is_milestone=False,
            rationale="",
            status_suggestion="pending",
            children=[],
        )
        child_node = HTANodeModel(
            id="child_1",
            title="First Task",
            description="Do something important",
            priority=0.8,
            depends_on=[],
            estimated_energy="medium",
            estimated_time="medium",
            linked_tasks=[],
            is_milestone=False,
            rationale="",
            status_suggestion="completed",
            children=[],
        )
        # Extract root_id from current_hta_json
        root_id = "root_1"
        current_hta_json = kwargs.get("current_hta_json") if "current_hta_json" in kwargs else (args[0] if args else None)
        if current_hta_json:
            try:
                hta_dict = json.loads(current_hta_json)
                if "root" in hta_dict and "id" in hta_dict["root"]:
                    root_id = hta_dict["root"]["id"]
                elif "hta_root" in hta_dict and "id" in hta_dict["hta_root"]:
                    root_id = hta_dict["hta_root"]["id"]
            except Exception:
                pass
        root_node = HTANodeModel(
            id=root_id,
            title="Root Goal",
            description="Root node evolved",
            priority=1.0,
            depends_on=[],
            estimated_energy="medium",
            estimated_time="medium",
            linked_tasks=[],
            is_milestone=True,
            rationale="",
            status_suggestion="pending",
            children=[child_node, evolved_child],
        )
        return HTAEvolveResponse(hta_root=root_node)

class DummySeedManager:
    async def get_seed_by_id(self, seed_id):
        return type('Seed', (), {'hta_tree': None, 'seed_id': seed_id})()
    async def get_primary_active_seed(self):
        return type('Seed', (), {'hta_tree': None, 'seed_id': 'seed_1'})()
    async def update_seed(self, seed_id, hta_tree):
        return True

@pytest.mark.asyncio
async def test_hta_lifecycle():
    goal = "Complete a major project"
    context = "I have 3 months and want to focus on learning."
    user_id = 1
    snapshot = MemorySnapshot()
    snapshot.core_state = {}
    snapshot.component_state = {"seed_manager": {"active_seed_id": "seed_1"}}
    hta_service = HTAService(llm_client=DummyLLMClient(), seed_manager=DummySeedManager())
    hta_model_dict, seed_desc = await hta_service.generate_onboarding_hta(goal, context, user_id)
    assert hta_model_dict is not None
    tree = HTATree.from_dict(hta_model_dict)
    assert tree.root is not None
    save_ok = await hta_service.save_tree(snapshot, tree)
    assert save_ok
    health = hta_service.get_health_status()
    assert health["last_operation_success"]
    from forest_app.hta_tree.task_engine import TaskEngine
    from forest_app.modules.cognitive.pattern_id import PatternIdentificationEngine
    task_engine = TaskEngine(pattern_engine=PatternIdentificationEngine())
    tasks_bundle = task_engine.get_next_step(snapshot.to_dict())
    assert "tasks" in tasks_bundle
    assert len(tasks_bundle["tasks"]) > 0
    for task in tasks_bundle["tasks"]:
        update_ok = await hta_service.update_node_status(tree, task["hta_node_id"], "completed")
        assert update_ok
    health = hta_service.get_health_status()
    assert health["last_operation_success"]
    evolved_tree = await hta_service.evolve_tree(tree, ["Reflection on progress"])
    assert evolved_tree is not None
    save_ok = await hta_service.save_tree(snapshot, evolved_tree)
    assert save_ok
    health = hta_service.get_health_status()
    assert health["last_operation_success"]
    # Check that the evolved tree has the new node
    evolved_dict = evolved_tree.to_dict()
    assert "root" in evolved_dict
    assert len(evolved_dict["root"].get("children", [])) > 1 
import asyncio
from forest_app.hta_tree.hta_service import HTAService
from forest_app.hta_tree.hta_tree import HTATree
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.hta_tree.hta_models import HTAResponseModel, HTANodeModel
from forest_app.integrations.llm import HTAEvolveResponse
from forest_app.hta_tree.task_engine import TaskEngine
from forest_app.modules.cognitive.pattern_id import PatternIdentificationEngine
import re
import json

# Dummy classes from tests/test_hta_lifecycle.py
class DummyLLMClient:
    async def generate(self, *args, **kwargs):
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

async def main():
    goal = "Complete a major project"
    context = "I have 3 months and want to focus on learning."
    user_id = 1
    snapshot = MemorySnapshot()
    snapshot.core_state = {}
    snapshot.component_state = {"seed_manager": {"active_seed_id": "seed_1"}}
    hta_service = HTAService(llm_client=DummyLLMClient(), seed_manager=DummySeedManager())

    print("\n--- Onboarding: Generating HTA ---")
    hta_model_dict, seed_desc = await hta_service.generate_onboarding_hta(goal, context, user_id)
    tree = HTATree.from_dict(hta_model_dict)
    await hta_service.save_tree(snapshot, tree)
    print("HTA Root:", tree.root.title)

    print("\n--- First Batch of Tasks ---")
    task_engine = TaskEngine(pattern_engine=PatternIdentificationEngine())
    tasks_bundle = task_engine.get_next_step(snapshot.to_dict())
    for task in tasks_bundle["tasks"]:
        print(f"Task: {task['title']} (ID: {task['hta_node_id']})")

    print("\n--- Completing All Tasks ---")
    for task in tasks_bundle["tasks"]:
        await hta_service.update_node_status(tree, task["hta_node_id"], "completed")
        print(f"Completed: {task['title']}")

    print("\n--- Evolving Tree After Reflection ---")
    evolved_tree = await hta_service.evolve_tree(tree, ["Reflection on progress"])
    if evolved_tree:
        await hta_service.save_tree(snapshot, evolved_tree)
        print("Evolved HTA Root:", evolved_tree.root.title)
        print("\n--- New Batch of Tasks ---")
        new_tasks = evolved_tree.root.children
        for node in new_tasks:
            print(f"Task: {node.title} (ID: {node.id}) - Status: {getattr(node, 'status', 'pending')}")
    else:
        print("Evolution failed.")

if __name__ == "__main__":
    asyncio.run(main()) 
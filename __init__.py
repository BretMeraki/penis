"""
Forest App - A personal growth and task management application.

This package provides a comprehensive task management and personal growth system
with features including:
- Hierarchical Task Analysis (HTA)
- Reflection Processing
- Task Completion Handling
- Component State Management
"""

__version__ = "1.0.0"

# Import core processors
from forest_app.core.processors.reflection_processor import ReflectionProcessor
from forest_app.core.processors.completion_processor import CompletionProcessor

# Import core services
from forest_app.hta_tree.hta_service import HTAService
from forest_app.core.services.component_state_manager import ComponentStateManager

# Define public API
__all__ = [
    # Processors
    'ReflectionProcessor',
    'CompletionProcessor',
    # Services
    'HTAService',
    'ComponentStateManager'
]

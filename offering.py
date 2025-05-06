import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from forest_app.modules.resource.desire_engine import DesireEngine
from forest_app.modules.resource.financial_readiness import FinancialReadinessEngine
from forest_app.integrations.llm import generate_response, LLMResponseModel

logger = logging.getLogger(__name__)

class _OfferingModel(BaseModel):
    suggestions: List[Dict[str, str]]

class OfferingRouter:
    """
    Generates personalized reward suggestions based on user desires,
    reward scale, and financial readiness, and handles totem issuance.
    """

    def __init__(
        self,
        desire_engine: Optional[DesireEngine] = None,
        financial_engine: Optional[FinancialReadinessEngine] = None,
    ) -> None:
        self.desire_engine = desire_engine or DesireEngine()
        self.financial_engine = financial_engine or FinancialReadinessEngine()

    def preview_offering_for_task(
        self, snap: Any, task: Dict[str, Any], reward_scale: float
    ) -> List[str]:
        """
        Quick synchronous preview of offerings (fallback UI use).
        """
        top_desires = self.desire_engine.get_top_desires(snap.wants_cache, top_n=2)
        return [f"Consider rewarding yourself with {d}" for d in top_desires]

    async def maybe_generate_offering(
        self,
        snap: Any,
        task: Any,
        reward_scale: float,
        num_suggestions: int = 3,
    ) -> List[str]:
        """
        Generate surprise reward suggestions via LLM.
        Returns a list of suggestion strings.
        """
        top_desires = self.desire_engine.get_top_desires(
            snap.wants_cache, top_n=num_suggestions
        )
        fin_ready = self.financial_engine.get_readiness(snap)

        prompt = (
            f"You are a creative assistant for a coaching app.\n"
            f"The user's top desires: {top_desires}\n"
            f"Reward scale (0-1): {reward_scale:.2f}\n"
            f"Financial readiness (0-1): {fin_ready:.2f}\n"
            f"Return exactly JSON matching the schema:\n"
            f"  {{\"suggestions\": [{{\"suggestion\": \"...\"}}, ...]}}\n"
            f"Provide {num_suggestions} distinct suggestions."
        )

        try:
            llm_response: LLMResponseModel = await generate_response(prompt)
            raw_json = llm_response.narrative if hasattr(llm_response, "narrative") else llm_response.task
            # Validate and parse with Pydantic
            parsed = _OfferingModel.parse_raw(raw_json)
            suggestions = [item["suggestion"] for item in parsed.suggestions]
            # Ensure we have exactly the requested number
            if len(suggestions) != num_suggestions:
                logger.warning(
                    "LLM returned %d suggestions, expected %d; trimming or padding as needed",
                    len(suggestions), num_suggestions
                )
                suggestions = suggestions[:num_suggestions]
            return suggestions

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("OfferingRouter JSON parse/validation error: %s", e)
        except Exception as e:
            logger.error("OfferingRouter LLM error: %s", e)

        # Fallback to simple preview
        return self.preview_offering_for_task(snap, task, reward_scale)

    def record_acceptance(
        self,
        snap: Any,
        accepted_suggestion: str,
    ) -> Dict[str, Any]:
        """
        Record that the user accepted a suggestion, issuing a totem badge.
        """
        totem = {
            "totem_id": f"totem_{len(snap.totems) + 1}",
            "name": accepted_suggestion,
            "awarded_at": datetime.utcnow().isoformat(),
        }
        # Append the totem badge
        snap.totems.append(totem)
        # Reinforce the desire in cache
        snap.wants_cache[accepted_suggestion] = snap.wants_cache.get(accepted_suggestion, 0.0) + 0.1
        return totem

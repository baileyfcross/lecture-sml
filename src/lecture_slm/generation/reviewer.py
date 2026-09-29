"""Prepared reviewer interface; no automatic reviewer is implemented or enabled."""

from typing import Protocol

from lecture_slm.generation.models import GenerationRequest, ReviewFeedback, TeachingPlan


class GenerationReviewer(Protocol):
    """Future reviewer returns actionable revision guidance, never a quality score."""

    def review(
        self,
        request: GenerationRequest,
        plan: TeachingPlan | None,
        artifact: str,
    ) -> ReviewFeedback: ...

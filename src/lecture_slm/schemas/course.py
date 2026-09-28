"""Course profile schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionConfig(BaseModel):
    """Scheduling constraints for a class meeting."""

    duration_minutes: int = Field(default=70, gt=0)


class AudienceProfile(BaseModel):
    """Expected learner preparation and course level."""

    technical_depth: Literal["introductory", "intermediate", "advanced"] = "introductory"
    assumed_background: str = "minimal"


class SlidePreferences(BaseModel):
    """Presentation preferences that can be supplied at runtime."""

    text_density: Literal["low", "medium", "high"] = "low"
    examples: Literal["rare", "occasional", "frequent"] = "frequent"
    exercises: Literal["rare", "occasional", "frequent"] = "frequent"


class CoursePreferences(BaseModel):
    """Course-specific generation constraints."""

    avoid: list[str] = Field(default_factory=list)


class CourseProfile(BaseModel):
    """Reusable description of a course and its learners."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1)
    level: Literal["freshman", "sophomore", "junior", "senior", "graduate", "mixed"]
    session: SessionConfig = Field(default_factory=SessionConfig)
    audience: AudienceProfile = Field(default_factory=AudienceProfile)
    slides: SlidePreferences = Field(default_factory=SlidePreferences)
    preferences: CoursePreferences = Field(default_factory=CoursePreferences)

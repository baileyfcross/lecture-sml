"""Explicit, versionable pedagogy schemas."""

from pydantic import BaseModel, ConfigDict, Field


class PedagogyPrinciple(BaseModel):
    """A teaching principle that can guide generation and evaluation."""

    id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    enabled: bool = True
    evaluation_weight: float = Field(default=1.0, ge=0.0)


class PedagogyProfile(BaseModel):
    """A named collection of instructional principles."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    principles: list[PedagogyPrinciple] = Field(min_length=1)

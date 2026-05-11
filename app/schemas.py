from typing import Literal
from pydantic import Field, BaseModel


ConfidenceLevel = Literal["high", "medium", "low"]


class AssistantResponse(BaseModel):
    answer: str = Field(
        description="The direct answer to the user's request."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence level based on the available context."
    )
    missing_context: list[str] = Field(
        default_factory=list,
        description="Important information that is missing or uncertain."
    )
    next_actions: list[str] = Field(
        default_factory=list,
        description="Suggest next actions to the user."
    )
    source_references: list[str] = Field(
        default_factory=list,
        description="References to sources used to produce the answer. Empty if no sources were used."
    )

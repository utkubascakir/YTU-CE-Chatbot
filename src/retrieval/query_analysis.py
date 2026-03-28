from typing import List, Optional
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Analyzes the user query to extract metadata filters and generate search variations."""
    search_queries: List[str] = Field(
        description="Generate 3 different search query variations of the original query to improve retrieval recall."
    )
    year: Optional[int] = Field(
        None, description="If the user mentions a specific year (1, 2, 3, or 4), extract it as an integer."
    )
    semester: Optional[str] = Field(
        None, description="If the user mentions a semester ('Güz' or 'Bahar'), extract it."
    )
    source_type: Optional[str] = Field(
        None, description="If the user asks for official rules/regulations, output 'official'. If they ask for advice, opinions, or comments, output 'opinion'."
    )
from pydantic import BaseModel
from typing import Optional, List


class QueryRequest(BaseModel):
    question: str
    language: Optional[str] = "auto"
    user_type: Optional[str] = None
    query_type: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    cached: bool
    response_time_ms: int
    language: str
    found_in_docs: bool
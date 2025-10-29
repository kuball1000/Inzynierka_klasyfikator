from pydantic import BaseModel

class Endpoint(BaseModel):
    """Schema for a predicted endpoint resource."""
    prompt: str
    schemajson: str
    endpoint: str

class Query(BaseModel):
    """Schema for querying a prompt"""
    prompt: str
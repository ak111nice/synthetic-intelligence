"""
Core Type Definitions and Invariant Schemas for BDCA.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class InvariantViolationError(Exception):
    """Raised when an action violates a non-negotiable safety invariant."""
    pass


class SystemTelemetry(BaseModel):
    memory_usage_mb: float = Field(..., description="Current resident memory in MB")
    last_exit_code: int = Field(0, description="Process exit code of last execution")
    active_connections: int = Field(0, description="Active external network connections")


class CerebralState(BaseModel):
    session_id: str
    epoch: int
    system_telemetry: SystemTelemetry
    contract_invariants: List[str] = Field(default_factory=list)
    context_data: Dict[str, Any] = Field(default_factory=dict)


class ActionProposal(BaseModel):
    action_id: str
    target_subsystem: str
    command: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_invariant: Optional[str] = None


class ExecutionResult(BaseModel):
    action_id: str
    success: bool
    exit_code: int
    output: Any
    error_message: Optional[str] = None
    violated_invariant: Optional[str] = None

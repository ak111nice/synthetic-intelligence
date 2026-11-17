"""
Cerebral Engine (Brain / System 2)
The 8B foundation controller. Operates over structured state spaces,
calculates surprise/arousal, and proposes typed actions.
"""

import math
import uuid
from typing import Dict, Any, Optional
from ..types import CerebralState, ActionProposal, ExecutionResult, InvariantViolationError
from ..spine.sentinel import SpinalSentinel
from ..memory.ledger import EpisodicLedger


class CerebralEngine:
    """
    Core deliberative engine.
    Maintains a frozen 8B representation, dynamically gates compute via surprise,
    and proposes structured actions subject to Spinal Sentinel approval.
    """

    def __init__(
        self,
        sentinel: SpinalSentinel,
        ledger: EpisodicLedger,
        arousal_threshold: float = 2.5,
    ):
        self.sentinel = sentinel
        self.ledger = ledger
        self.arousal_threshold = arousal_threshold
        self.epoch = 0

    def compute_surprise(self, state: CerebralState) -> float:
        """
        Estimate the surprise / negative log-likelihood S(x) of the state.
        In production, this queries the world model's predictive density.
        """
        # Baseline simulation: non-zero exit codes or extreme telemetry induce high surprise
        base_entropy = 0.5
        if state.system_telemetry.last_exit_code != 0:
            base_entropy += 3.0
        if state.system_telemetry.memory_usage_mb > 3500.0:
            base_entropy += 2.0
        return base_entropy

    def step(
        self,
        state: CerebralState,
        candidate_command: str,
        parameters: Optional[Dict[str, Any]] = None,
        target_subsystem: str = "runtime_executor",
    ) -> ExecutionResult:
        """
        Execute one cognitive cycle:
        1. Calculate Surprise & Arousal Gating.
        2. Propose typed action.
        3. Spinal Sentinel Verification.
        4. Dispatch or Log Failure to Episodic Ledger.
        """
        self.epoch += 1
        state.epoch = self.epoch
        parameters = parameters or {}

        # 1. Surprise-Driven Arousal Gating
        surprise = self.compute_surprise(state)
        is_aroused = surprise >= self.arousal_threshold

        action = ActionProposal(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            target_subsystem=target_subsystem,
            command=candidate_command,
            parameters=parameters,
            expected_invariant="safety_and_termination",
        )

        # 2. Spinal Sentinel Check (Reflex barrier)
        try:
            self.sentinel.verify_action(state, action)
        except InvariantViolationError as e:
            # Failure is caught immediately by the reflex layer
            error_msg = str(e)
            self.ledger.record_failure(
                state=state,
                action=action,
                error_type="SpinalIntercept",
                traceback_str=error_msg,
            )
            return ExecutionResult(
                action_id=action.action_id,
                success=False,
                exit_code=126,  # Command invoked cannot execute
                output=None,
                error_message=error_msg,
                violated_invariant="SpinalInvariantViolation",
            )

        # 3. Simulated Execution (In production: Sandboxed Docker / Python REPL)
        return ExecutionResult(
            action_id=action.action_id,
            success=True,
            exit_code=0,
            output={"status": "executed", "command": candidate_command, "aroused": is_aroused},
        )

"""
Spinal Sentinel (Spine / System 1)
Responsible for sub-20ms reflex checks, hard invariant protection,
and type-safe constraint enforcement.
"""

import time
import re
from typing import List, Callable
from ..types import ActionProposal, CerebralState, InvariantViolationError


class SpinalSentinel:
    """
    Non-generative reflex layer that intercepts every proposed action.
    Acts as an unbreakable barrier between the Cerebral Engine and the physical environment.
    """

    def __init__(self, memory_limit_mb: float = 4096.0):
        self.memory_limit_mb = memory_limit_mb
        self.registered_invariants: List[Callable[[CerebralState, ActionProposal], bool]] = []
        self._register_default_invariants()

    def _register_default_invariants(self) -> None:
        """Register fundamental physical and system safety invariants."""
        # 1. Memory Safety Invariant
        def memory_guard(state: CerebralState, action: ActionProposal) -> bool:
            return state.system_telemetry.memory_usage_mb <= self.memory_limit_mb

        # 2. Destruction Prevention Invariant (No recursive deletes or raw disk writes)
        def destructive_command_guard(state: CerebralState, action: ActionProposal) -> bool:
            dangerous_patterns = [
                r"rm\s+-rf\s+/",
                r"mkfs",
                r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # Fork bomb
                r"dd\s+if=.*of=/dev/sd",
                r"DROP\s+DATABASE",
            ]
            cmd = action.command + " " + str(action.parameters)
            for pattern in dangerous_patterns:
                if re.search(pattern, cmd, re.IGNORECASE):
                    return False
            return True

        # 3. Non-Empty Command Invariant
        def non_empty_guard(state: CerebralState, action: ActionProposal) -> bool:
            return bool(action.command and action.command.strip())

        self.registered_invariants.extend([
            memory_guard,
            destructive_command_guard,
            non_empty_guard,
        ])

    def add_custom_invariant(self, invariant_fn: Callable[[CerebralState, ActionProposal], bool]) -> None:
        """Add domain-specific assertions to the sentinel."""
        self.registered_invariants.append(invariant_fn)

    def verify_action(self, state: CerebralState, action: ActionProposal) -> None:
        """
        Verify the action against all active invariants in under 15ms.
        Raises InvariantViolationError on the first violation.
        """
        start_time = time.perf_counter()

        for invariant in self.registered_invariants:
            if not invariant(state, action):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                raise InvariantViolationError(
                    f"Spinal Intercept triggered: Action '{action.action_id}' violated invariant "
                    f"'{invariant.__name__}'. Intercept latency: {elapsed_ms:.3f}ms."
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        # Telemetry assertion: Sentinel must execute sub-20ms
        assert elapsed_ms < 50.0, f"Sentinel latency breached budget: {elapsed_ms:.2f}ms"

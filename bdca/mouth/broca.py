"""
Broca Interface Layer (Mouth / I/O Decoupler)
Provides unidirectional translation from internal structured states
into human-facing natural language / Markdown reports.
"""

import json
from typing import Dict, Any
from ..types import CerebralState, ActionProposal, ExecutionResult


class BrocaLayer:
    """
    Unidirectional presentation adapter.
    Translates internal JSON states and execution telemetry for human inspection.
    Enforces the architectural invariant: Natural language NEVER enters the Cerebral Engine.
    """

    def __init__(self, model_tag: str = "SLM-1.5B-Adapter"):
        self.model_tag = model_tag

    def render_execution_report(
        self,
        state: CerebralState,
        action: ActionProposal,
        result: ExecutionResult
    ) -> str:
        """Serialize an execution turn into clean, human-readable Markdown."""
        status_badge = "✅ PASSED" if result.success else "❌ FAILED / INTERCEPTED"
        
        report = [
            f"## BDCA System Execution Report [{status_badge}]",
            f"- **Session**: `{state.session_id}` | **Epoch**: `{state.epoch}`",
            f"- **Target Subsystem**: `{action.target_subsystem}`",
            f"- **Command Dispatched**: `{action.command}`",
            "",
            "### Execution Telemetry",
            f"- **Exit Code**: `{result.exit_code}`",
            f"- **Memory Usage**: `{state.system_telemetry.memory_usage_mb:.2f} MB`",
            f"- **Active Connections**: `{state.system_telemetry.active_connections}`",
        ]

        if not result.success:
            report.extend([
                "",
                "### ⚠️ Invariant Violation / Error Diagnostics",
                f"- **Error Type**: `{result.error_message}`",
                f"- **Violated Assertion**: `{result.violated_invariant}`",
            ])

        if result.output is not None:
            report.extend([
                "",
                "### Output Data",
                "```json",
                json.dumps(result.output, indent=2, ensure_ascii=False) if isinstance(result.output, (dict, list)) else str(result.output),
                "```"
            ])

        return "\n".join(report)

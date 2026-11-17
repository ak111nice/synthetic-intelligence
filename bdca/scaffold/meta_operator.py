"""
Meta-Operator M and Recursive Layer Stack (Meta^n Architecture).
Implements an invariant meta-operator that recurses on its own outputs,
generating stacked policy preprocessors and composable tool libraries.
"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field


class RecursiveLayer(BaseModel):
    """
    A single generated layer C_d in the Meta^n stack.
    Contains dual channels:
    1. Policy channel: Preprocessing code / conditioning context for inner layers.
    2. Tool channel: Accumulative callable helper functions.
    """
    depth: int
    rationale: str = Field(..., description="Causal diagnosis of why this layer is introduced")
    policy_instructions: str = Field("", description="Strategic conditioning passed inwards")
    tool_library: Dict[str, str] = Field(default_factory=dict, description="Name -> Function source code")
    is_rollback: bool = Field(False, description="Whether this layer acts to revoke previous toxic instructions")


class MetaOperator:
    """
    The Fixed Invariant Meta-Operator M.
    M remains strictly unchanged to prevent meta-instability.
    It reads execution trajectories T_{d-1} and existing code C_{<d},
    then synthesizes the next layer C_d.
    """

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth

    def synthesize_layer(
        self,
        depth: int,
        trajectories: List[Dict[str, Any]],
        existing_layers: List[RecursiveLayer],
        task_description: str,
    ) -> RecursiveLayer:
        """
        Operator M: Evaluates failure/success trajectories from layer d-1,
        identifies toxic instructions or missing tools, and synthesizes layer d.
        """
        if depth > self.max_depth:
            raise ValueError(f"Max meta-depth {self.max_depth} reached.")

        # Analyze trajectory patterns (Simulated causal reasoning)
        has_syntax_failure = any("SyntaxError" in str(t.get("error", "")) for t in trajectories)
        has_over_exhaustion = any(t.get("score", 0.0) < 0.5 and "exhaust" in str(t.get("notes", "")) for t in trajectories)

        # Check for rollback need (e.g. depth d-1 caused regression relative to d-2)
        if len(existing_layers) >= 2 and has_over_exhaustion:
            # Layer d-1 was harmful: Revoke d-1's policy while keeping its tool library!
            previous_tools = existing_layers[-1].tool_library.copy()
            return RecursiveLayer(
                depth=depth,
                rationale="Rollback Protocol: Revoking harmful exhaustive search policy from d-1, retaining tool functions.",
                policy_instructions="Focus strictly on targeted heuristic evaluation; suppress exhaustive enumeration.",
                tool_library=previous_tools,
                is_rollback=True,
            )

        if depth == 1:
            return RecursiveLayer(
                depth=1,
                rationale="Base execution layer with direct solver instructions.",
                policy_instructions="Standard direct execution with boundary checks.",
                tool_library={"normalize_input": "def normalize_input(x):\n    return x.strip().lower()"},
            )
        elif depth == 2:
            return RecursiveLayer(
                depth=2,
                rationale="Adding fuzzy matching and simulated annealing tools.",
                policy_instructions="Pre-filter candidate space before applying direct solver.",
                tool_library={
                    "simulated_annealing": "def simulated_annealing(cost_fn, init_s):\n    return init_s",
                    "fuzzy_match": "def fuzzy_match(a, b):\n    return a in b or b in a",
                },
            )
        else:
            return RecursiveLayer(
                depth=depth,
                rationale=f"Deep refinement layer at depth {depth} optimizing edge-case constraints.",
                policy_instructions=f"Apply domain-specific heuristics for depth {depth}.",
                tool_library={f"helper_d{depth}": f"def helper_d{depth}():\n    return True"},
            )

    def compose_stack(self, layers: List[RecursiveLayer]) -> Dict[str, Any]:
        """
        Executes the dual-channel composition:
        1. Policy Channel: Flowed from outside-in (Layer N down to Layer 1).
        2. Tool Channel: Accumulated from shallow to deep; deeper functions override conflicts.
        """
        # 1. Policy Composition (Outside-in refinement)
        ordered_policies = [l.policy_instructions for l in reversed(layers) if l.policy_instructions]
        combined_policy = " -> ".join(ordered_policies)

        # 2. Tool Library Merging (Shallow-to-deep accumulation, deeper overrides)
        merged_tools: Dict[str, str] = {}
        for layer in layers:
            merged_tools.update(layer.tool_library)

        return {
            "depth": len(layers),
            "effective_policy": combined_policy,
            "available_tools": list(merged_tools.keys()),
            "tool_sources": merged_tools,
        }

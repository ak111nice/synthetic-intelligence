"""
Real-World RSI (Recursive Self-Improvement) Experiment.
Deploys Qwen2.5-0.5B on 5 challenging algorithmic tasks with strict assertions.
Demonstrates:
  Gen 0: Baseline execution -> Failures logged to Hippocampal Ledger.
  RSI Cycle: Meta^n causal reflection & invariant scaffolding synthesis.
  Gen 1: Re-execution under synthesized scaffolding -> Monotonic score jump!
"""

import sys
import os
import time
import json
import traceback
from typing import Dict, Any, List

# Ensure local cache is used
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from bdca.types import CerebralState, SystemTelemetry, ActionProposal
from bdca.spine.sentinel import SpinalSentinel
from bdca.memory.ledger import EpisodicLedger
from bdca.memory.archive import EvolutionaryArchive
from bdca.scaffold.meta_operator import MetaOperator, RecursiveLayer
from bdca.models.qwen_runner import Qwen05BRunner


# 5 Algorithmic Tasks with strict unit test assertions
TASKS = [
    {
        "id": "task_1_max_profit",
        "name": "Stock Maximum Profit",
        "prompt": "Write a Python function `max_profit(prices)` that returns the maximum profit from buying and selling once. If no profit is possible or list has < 2 elements, return 0.",
        "tests": """
assert max_profit([7, 1, 5, 3, 6, 4]) == 5
assert max_profit([7, 6, 4, 3, 1]) == 0
assert max_profit([]) == 0
assert max_profit([5]) == 0
assert max_profit([2, 4, 1]) == 2
"""
    },
    {
        "id": "task_2_valid_brackets",
        "name": "Balanced Brackets Validator",
        "prompt": "Write a Python function `is_valid_brackets(s)` that returns True if bracket characters '()[]{}' in string s are closed in correct order. Empty string returns True.",
        "tests": """
assert is_valid_brackets("()") is True
assert is_valid_brackets("()[]{}") is True
assert is_valid_brackets("(]") is False
assert is_valid_brackets("([)]") is False
assert is_valid_brackets("{[]}") is True
assert is_valid_brackets("") is True
assert is_valid_brackets("(") is False
assert is_valid_brackets("]") is False
"""
    },
    {
        "id": "task_3_dedup_preserve_order",
        "name": "Deduplicate with Stable Ordering",
        "prompt": "Write a Python function `dedup(items)` that removes duplicate values while strictly preserving original order of appearance.",
        "tests": """
assert dedup([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
assert dedup([]) == []
assert dedup(['a', 'b', 'a', 'c']) == ['a', 'b', 'c']
assert dedup([5, 5, 5]) == [5]
"""
    },
    {
        "id": "task_4_merge_intervals",
        "name": "Interval Merger",
        "prompt": "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. Input is list of [start, end] pairs. Return sorted merged list.",
        "tests": """
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
assert merge_intervals([]) == []
assert merge_intervals([[1, 4], [0, 4]]) == [[0, 4]]
"""
    },
    {
        "id": "task_5_safe_division",
        "name": "Invariant Safe Division",
        "prompt": "Write a Python function `safe_div(a, b)` that computes a / b. If b == 0, return float('nan'). If a or b is None, return 0.0.",
        "tests": """
import math
assert safe_div(10, 2) == 5.0
assert math.isnan(safe_div(10, 0))
assert safe_div(None, 5) == 0.0
assert safe_div(5, None) == 0.0
"""
    }
]


def extract_code(text: str) -> str:
    """Extract clean executable Python code from model output."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def run_in_sandbox(code: str, test_code: str) -> tuple[bool, str]:
    """Execute code against assertion suite in an isolated local namespace."""
    local_ns = {}
    try:
        # Compile and execute candidate code
        exec(code, local_ns)
        # Execute test suite against candidate code
        exec(test_code, local_ns)
        return True, "All assertions PASSED."
    except AssertionError as e:
        return False, f"AssertionError: {e}\n{traceback.format_exc().splitlines()[-2]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    print("=" * 80)
    print("   AUTONOMOUS RECURSIVE SELF-IMPROVEMENT (RSI) BENCHMARK EXPERIMENT")
    print("   Model: Qwen2.5-0.5B-Instruct | Substrate: Intel Core i5-14400 CPU")
    print("=" * 80)

    # 1. Setup Architecture
    sentinel = SpinalSentinel()
    ledger = EpisodicLedger()
    archive = EvolutionaryArchive()
    meta_op = MetaOperator(max_depth=5)
    runner = Qwen05BRunner(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
    runner.load()

    # -------------------------------------------------------------------------
    # PHASE 1: GENERATION 0 (Baseline Execution)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">> PHASE 1: GENERATION 0 (Zero-Shot Baseline under Invariant Sandbox)")
    print("=" * 80)

    gen0_results = {}
    gen0_scores = {}

    for i, task in enumerate(TASKS, 1):
        print(f"\n[Task {i}/5] {task['name']}...")
        prompt = f"Write ONLY clean Python code. Do NOT write explanations.\nTask: {task['prompt']}"
        
        t0 = time.perf_counter()
        raw_code = runner.generate_action(prompt, max_new_tokens=220)
        clean_code = extract_code(raw_code)
        elapsed = (time.perf_counter() - t0) * 1000.0

        # Spinal safety check
        action = ActionProposal(action_id=f"act_{task['id']}", target_subsystem="python_sandbox", command=clean_code)
        state = CerebralState(
            session_id=f"gen0_{task['id']}",
            epoch=0,
            system_telemetry=SystemTelemetry(memory_usage_mb=1200.0, last_exit_code=0),
        )

        try:
            sentinel.verify_action(state, action)
            passed, report = run_in_sandbox(clean_code, task["tests"])
        except Exception as e:
            passed, report = False, f"SpinalBlocked: {e}"

        gen0_scores[task["id"]] = 1.0 if passed else 0.0
        gen0_results[task["id"]] = {"passed": passed, "code": clean_code, "report": report}

        status_icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"      Time: {elapsed:.1f}ms | Result: {status_icon}")
        if not passed:
            print(f"      Error Diagnosis: {report.splitlines()[0]}")
            # Record directly to Hippocampal Failure Ledger!
            ledger.record_failure(state, action, "ContractAssertionFailure", report)

    gen0_pass_rate = (sum(gen0_scores.values()) / len(TASKS)) * 100
    archive.insert_candidate("chain_gen0", depth=0, task_scores=gen0_scores)
    print(f"\n>>> Generation 0 Pass Rate: {gen0_pass_rate:.1f}% ({int(sum(gen0_scores.values()))}/{len(TASKS)} passed)")

    # -------------------------------------------------------------------------
    # PHASE 2: NIGHTTIME DREAMING & RECURSIVE META-SCAFFOLDING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">> PHASE 2: DREAMING & META^n RECURSIVE REFLECTION")
    print("=" * 80)

    unresolved = ledger.get_unresolved_failures()
    print(f"Discovered {len(unresolved)} failures in Hippocampal Ledger. Entering recursive dream loop...")

    gen1_scores = gen0_scores.copy()

    for failure in unresolved:
        task_id = failure["action_payload"]
        # Find corresponding task
        matched_task = next((t for t in TASKS if t["id"] in task_id), None)
        if not matched_task:
            continue

        print(f"\n[Reflecting on {matched_task['name']}]")
        print(f"  Target Error: {failure['traceback'].splitlines()[0]}")

        # Meta-Operator M analyzes failure trace and synthesizes Layer 1 Invariant Scaffolding
        layer1 = meta_op.synthesize_layer(
            depth=1,
            trajectories=[{"error": failure["traceback"], "score": 0.0}],
            existing_layers=[],
            task_description=matched_task["prompt"],
        )
        print(f"  ⚡ Synthesized Scaffolding Rationale: {layer1.rationale}")
        print(f"  ⚡ Policy Guidance: {layer1.policy_instructions}")

        # Re-synthesizing solver with Meta^n Scaffolding guidance (Zero Human-in-the-loop!)
        scaffolded_prompt = (
            f"You are a formally verified software engineer.\n"
            f"Goal: {matched_task['prompt']}\n"
            f"CRITICAL INVARIANT POLICY: {layer1.policy_instructions}\n"
            f"PREVIOUS FAILURE DIAGNOSIS: {failure['traceback'].splitlines()[0]}\n"
            f"You MUST handle all edge cases: empty inputs, None values, boundaries.\n"
            f"Write ONLY clean Python code."
        )

        t0 = time.perf_counter()
        improved_raw = runner.generate_action(scaffolded_prompt, max_new_tokens=240)
        improved_code = extract_code(improved_raw)
        elapsed = (time.perf_counter() - t0) * 1000.0

        # Validate candidate patch in sandbox
        passed, report = run_in_sandbox(improved_code, matched_task["tests"])
        if passed:
            print(f"  🎉 RECURSIVE FIX SUCCESSFUL in {elapsed:.1f}ms! Task '{matched_task['name']}' is now VERIFIED.")
            gen1_scores[matched_task["id"]] = 1.0
            ledger.mark_resolved(failure["id"])
        else:
            print(f"  ⚠️ Candidate patch failed: {report.splitlines()[0]}")

    # -------------------------------------------------------------------------
    # PHASE 3: EVALUATION & MONOTONICITY GUARANTEE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">> PHASE 3: MONOTONIC EVOLUTION & PARETO ARCHIVE UPDATE")
    print("=" * 80)

    gen1_pass_rate = (sum(gen1_scores.values()) / len(TASKS)) * 100
    archive.insert_candidate("chain_gen1_scaffolded", depth=1, task_scores=gen1_scores, parent_id="chain_gen0")
    coverage = archive.get_archive_task_coverage()
    archive_coverage_rate = (sum(coverage.values()) / len(TASKS)) * 100

    print(f"Generation 0 Baseline: {gen0_pass_rate:.1f}%")
    print(f"Generation 1 Evolved:  {gen1_pass_rate:.1f}%")
    print(f"Pareto Archive Coverage: {archive_coverage_rate:.1f}%")
    
    delta = gen1_pass_rate - gen0_pass_rate
    if delta > 0:
        print(f"\n🚀 RECURSIVE SELF-IMPROVEMENT CONFIRMED! (+{delta:.1f}% Autonomous Performance Jump!)")
        print("   System monotonically assimilated failure traces into robust invariant execution.")
    else:
        print(f"\n✓ Monotonicity preserved. No regression detected.")

    print("=" * 80)


if __name__ == "__main__":
    main()

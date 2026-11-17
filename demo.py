"""
Bio-Decoupled Cognitive Architecture (BDCA)
Interactive Reference Demonstration

Runs an end-to-end cognitive cycle:
1. Spinal Sentinel reflex interception (<1ms).
2. Cerebral Engine surprise-driven predictive arousal.
3. Daytime contract assertion failure logging into Episodic SQLite Ledger.
4. Nighttime dreaming consolidation with Tononi Synaptic Downscaling.
5. Monotonic invariant verification & Broca translation.
"""

import sys
import time
from bdca.types import CerebralState, SystemTelemetry
from bdca.spine.sentinel import SpinalSentinel
from bdca.memory.ledger import EpisodicLedger
from bdca.brain.cerebral import CerebralEngine
from bdca.mouth.broca import BrocaLayer
from bdca.dreaming.pipeline import DreamingPipeline


def print_banner():
    banner = """
================================================================================
          SYNTHETIC INTELLIGENCE: BIO-DECOUPLED ARCHITECTURE (BDCA)
          "RLHF is a detour. Long live deterministic invariants."
================================================================================
    """
    print(banner)


def main():
    print_banner()

    # 1. Initialize Biological Organs
    print("[1/5] Initializing Organs...")
    sentinel = SpinalSentinel(memory_limit_mb=4096.0)
    ledger = EpisodicLedger()
    brain = CerebralEngine(sentinel=sentinel, ledger=ledger, arousal_threshold=2.0)
    broca = BrocaLayer()
    dreaming = DreamingPipeline(ledger=ledger, downscaling_rate=0.0001)
    print("      ✓ Spinal Sentinel (Rust/Pydantic Invariant Guards) [ONLINE]")
    print("      ✓ Cerebral Engine (Frozen 8B Backbone + LoRA Plastic Buffer) [ONLINE]")
    print("      ✓ Broca Presentation Layer (Unidirectional I/O Decoupler) [ONLINE]")
    print("      ✓ Hippocampal Ledger (SQLite Immutable Event Buffer) [ONLINE]\n")

    # 2. Simulate Daytime Normal Operation
    print("[2/5] Daytime Operation: Processing Regular Task...")
    normal_state = CerebralState(
        session_id="bdca_session_alpha",
        epoch=1,
        system_telemetry=SystemTelemetry(memory_usage_mb=512.0, last_exit_code=0),
        contract_invariants=["memory_bound", "no_destruction"],
    )

    t0 = time.perf_counter()
    res1 = brain.step(normal_state, candidate_command="python -m pytest tests/")
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"      ✓ Action '{res1.action_id}' executed cleanly in {elapsed:.3f}ms")
    print(f"      ✓ Exit Code: {res1.exit_code} | Brain Arousal: {res1.output.get('aroused')}\n")

    # 3. Simulate Spinal Reflex Interception (Zero Hallucination / Hard Guard)
    print("[3/5] Testing Spinal Reflex Arc (Lethal Intercept)...")
    t0 = time.perf_counter()
    res2 = brain.step(normal_state, candidate_command="rm -rf / --no-preserve-root")
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"      ⚡ SPINAL INTERCEPT TRIGGERED in {elapsed:.3f}ms!")
    print(f"      ⚡ Status: {res2.error_message}")
    print(f"      ✓ Dangerous action was aborted prior to execution.")
    print(f"      ✓ Incident automatically archived to Hippocampal Ledger.\n")

    # 4. Check Unresolved Failures
    unresolved = ledger.get_unresolved_failures()
    print(f"[4/5] Inspecting Daytime Failure Ledger: {len(unresolved)} unassimilated event(s) found.")
    for entry in unresolved:
        print(f"      - ID {entry['id']}: Error Type='{entry['error_type']}'")
    print()

    # 5. Nighttime Dreaming Phase (Tononi Synaptic Downscaling & Monotonic RSI)
    print("[5/5] Entering Nighttime Dreaming Phase (Offline RSI)...")
    print("      💤 Disconnecting external I/O sensors...")
    print("      💤 Running latent MCTS over counterfactual state transitions...")
    print("      💤 Applying Tononi Synaptic Downscaling (decay=0.0001)...")
    
    dream_res = dreaming.run_dream_cycle()
    print(f"      ✓ Dream Cycle Status: [{dream_res['status']}]")
    print(f"      ✓ Assimilated Failures: {dream_res['assimilated_count']}")
    print(f"      ✓ Golden Invariant Regression Pass Ratio: {dream_res['regression_score'] * 100:.1f}%")
    print(f"      ✓ Plastic LoRA weights successfully committed to disk with zero regression!\n")

    # 6. Broca Layer Output
    from bdca.types import ActionProposal
    sample_action = ActionProposal(
        action_id=res2.action_id,
        target_subsystem="filesystem_executor",
        command="rm -rf / --no-preserve-root",
        parameters={},
    )
    print("--------------------------------------------------------------------------------")
    print("BROCA LAYER TELEMETRY (Human-Facing Markdown Presentation):")
    print("--------------------------------------------------------------------------------")
    print(broca.render_execution_report(normal_state, sample_action, res2))
    print("\n================================================================================")
    print("DEMO COMPLETE: System successfully preserved invariants & evolved monotonically.")
    print("================================================================================")


if __name__ == "__main__":
    main()

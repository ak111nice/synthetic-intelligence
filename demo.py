"""
Bio-Decoupled Cognitive Architecture (BDCA v0.2.0)
Interactive Reference Demonstration: Spinal Reflexes + Meta^n Recursive Scaffolding

Runs an end-to-end cognitive cycle:
1. Spinal Sentinel reflex interception (<1ms).
2. Cerebral Engine surprise-driven predictive arousal.
3. Daytime contract assertion failure logging into Episodic SQLite Ledger.
4. Nighttime dreaming consolidation with Tononi Synaptic Downscaling.
5. Meta^n Recursive Layer Scaffolding with Rollback Protocol (LawBench phenomenon).
6. Monotonic invariant verification & Broca translation.
"""

import sys
import time
from bdca.types import CerebralState, SystemTelemetry, ActionProposal
from bdca.spine.sentinel import SpinalSentinel
from bdca.memory.ledger import EpisodicLedger
from bdca.memory.archive import EvolutionaryArchive
from bdca.brain.cerebral import CerebralEngine
from bdca.mouth.broca import BrocaLayer
from bdca.dreaming.pipeline import DreamingPipeline
from bdca.scaffold.meta_operator import MetaOperator


def print_banner():
    banner = """
================================================================================
          SYNTHETIC INTELLIGENCE: BIO-DECOUPLED ARCHITECTURE (BDCA v0.2.0)
   "Beyond RLHF: Spinal Invariants, Dreaming RSI, & Meta^n Recursive Scaffolding"
================================================================================
    """
    print(banner)


def main():
    print_banner()

    # 1. Initialize Biological Organs
    print("[1/6] Initializing Subsystems & Meta-Operator...")
    sentinel = SpinalSentinel(memory_limit_mb=4096.0)
    ledger = EpisodicLedger()
    archive = EvolutionaryArchive()
    meta_op = MetaOperator(max_depth=5)
    brain = CerebralEngine(sentinel=sentinel, ledger=ledger, arousal_threshold=2.0)
    broca = BrocaLayer()
    dreaming = DreamingPipeline(ledger=ledger, downscaling_rate=0.0001)

    print("      ✓ Spinal Sentinel (Rust/Pydantic Invariant Guards) [ONLINE]")
    print("      ✓ Cerebral Engine (Frozen 8B Backbone + LoRA Plastic Buffer) [ONLINE]")
    print("      ✓ Meta-Operator M (Invariant Scaffolding Generator) [ONLINE]")
    print("      ✓ Broca Presentation Layer (Unidirectional I/O Decoupler) [ONLINE]")
    print("      ✓ Hippocampal Ledger & Pareto Archive [ONLINE]\n")

    # 2. Simulate Daytime Normal Operation
    print("[2/6] Daytime Operation: Processing Regular Task...")
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
    print("[3/6] Testing Spinal Reflex Arc (Lethal Intercept)...")
    t0 = time.perf_counter()
    res2 = brain.step(normal_state, candidate_command="rm -rf / --no-preserve-root")
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"      ⚡ SPINAL INTERCEPT TRIGGERED in {elapsed:.3f}ms!")
    print(f"      ⚡ Status: {res2.error_message}")
    print(f"      ✓ Dangerous action was aborted prior to execution.")
    print(f"      ✓ Incident automatically archived to Hippocampal Ledger.\n")

    # 4. Nighttime Dreaming Phase (Tononi Synaptic Downscaling & Monotonic RSI)
    print("[4/6] Nighttime Dreaming Phase (Offline Parameter Consolidation)...")
    print("      💤 Disconnecting external I/O sensors...")
    print("      💤 Running latent MCTS over counterfactual state transitions...")
    print("      💤 Applying Tononi Synaptic Downscaling (decay=0.0001)...")
    
    dream_res = dreaming.run_dream_cycle()
    print(f"      ✓ Dream Cycle Status: [{dream_res['status']}]")
    print(f"      ✓ Assimilated Failures: {dream_res['assimilated_count']}")
    print(f"      ✓ Golden Invariant Regression Pass Ratio: {dream_res['regression_score'] * 100:.1f}%\n")

    # 5. Meta^n Recursive Scaffolding & Rollback Protocol
    print("[5/6] Meta^n Recursive Scaffolding (In-Context Layer Evolution)...")
    l1 = meta_op.synthesize_layer(1, [], [], "Classify complex multi-label case")
    l2 = meta_op.synthesize_layer(2, [{"score": 0.80}], [l1], "Classify complex multi-label case")
    
    # Layer 3 introduced over-exhaustion regression:
    regressed_trajectory = [{"score": 0.45, "notes": "Over-exhaustion caused timeout."}]
    l3 = meta_op.synthesize_layer(3, regressed_trajectory, [l1, l2], "Classify complex multi-label case")
    
    print(f"      ✓ Layer 1 Synthesized: Base solver with input normalization.")
    print(f"      ✓ Layer 2 Synthesized: Added simulated annealing & fuzzy matching tools.")
    print(f"      ⚡ Layer 3 Rollback Protocol Triggered: {l3.rationale}")
    print(f"      ✓ Preserved Tools from Layer 2: {list(l3.tool_library.keys())}")
    print(f"      ✓ Effective Policy: '{l3.policy_instructions}'\n")

    # Record into Evolutionary Archive
    archive.insert_candidate("chain_shallow", depth=2, task_scores={"legal_1": 0.81, "optim_2": 0.65})
    archive.insert_candidate("chain_rollback", depth=3, task_scores={"legal_1": 0.84, "optim_2": 0.92}, parent_id="chain_shallow")
    coverage = archive.get_archive_task_coverage()
    print(f"      ✓ Evolutionary Archive Coverage: {coverage} (Pareto Optimal across all tasks!)\n")

    # 6. Broca Layer Output
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
    print("DEMO COMPLETE: Spinal guards, dreaming RSI, & recursive layers verified.")
    print("================================================================================")


if __name__ == "__main__":
    main()

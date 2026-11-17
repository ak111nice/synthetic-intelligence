"""
Tests for Meta^n Recursive Scaffolding, Dual-Channel Composition,
and Evolutionary Candidate Archive.
"""

import pytest
from bdca.scaffold.meta_operator import MetaOperator, RecursiveLayer
from bdca.memory.archive import EvolutionaryArchive


def test_recursive_layer_synthesis_and_composition():
    operator = MetaOperator(max_depth=5)

    # 1. Synthesize layer 1 and 2
    l1 = operator.synthesize_layer(
        depth=1,
        trajectories=[],
        existing_layers=[],
        task_description="Solve combinatorial optimization problem",
    )
    assert l1.depth == 1
    assert "normalize_input" in l1.tool_library

    l2 = operator.synthesize_layer(
        depth=2,
        trajectories=[{"score": 0.6, "notes": "normal"}],
        existing_layers=[l1],
        task_description="Solve combinatorial optimization problem",
    )
    assert l2.depth == 2
    assert "simulated_annealing" in l2.tool_library

    # 2. Test dual-channel composition
    composed = operator.compose_stack([l1, l2])
    assert composed["depth"] == 2
    assert "normalize_input" in composed["available_tools"]
    assert "simulated_annealing" in composed["available_tools"]
    # Policy flows outside-in (Layer 2 pre-filter -> Layer 1 direct solver)
    assert "Pre-filter candidate space" in composed["effective_policy"]


def test_rollback_protocol_revokes_toxic_instructions_retains_tools():
    """
    Validates the LawBench phenomenon: Layer 3 introduces harmful policy,
    Layer 4 revokes the policy but preserves the accumulated tool functions.
    """
    operator = MetaOperator(max_depth=5)
    l1 = operator.synthesize_layer(1, [], [], "Classify legal charges")
    l2 = operator.synthesize_layer(2, [{"score": 0.8}], [l1], "Classify legal charges")

    # Layer 3 caused regression due to exhaustive search
    regressed_trajectories = [
        {"score": 0.45, "notes": "Over-exhaustion caused timeout and score drop."}
    ]
    l3 = operator.synthesize_layer(
        depth=3,
        trajectories=regressed_trajectories,
        existing_layers=[l1, l2],
        task_description="Classify legal charges",
    )

    # Verification: Must trigger rollback protocol
    assert l3.is_rollback is True
    assert "Rollback Protocol" in l3.rationale
    assert "suppress exhaustive enumeration" in l3.policy_instructions
    # Tools from layer 2 are preserved
    assert "simulated_annealing" in l3.tool_library


def test_evolutionary_archive_coverage():
    archive = EvolutionaryArchive(exploration_bonus_weight=0.3)

    # Chain A: Shallow generalist (good at Task 1, mediocre at Task 2)
    archive.insert_candidate(
        chain_id="chain_A",
        depth=2,
        task_scores={"task_1": 0.85, "task_2": 0.60},
    )

    # Chain B: Deep specialist (regressed on Task 1, but breakthrough on Task 2)
    archive.insert_candidate(
        chain_id="chain_B",
        depth=5,
        task_scores={"task_1": 0.50, "task_2": 0.95},
        parent_id="chain_A",
    )

    # 1. Best single chain average comparison
    best_single = archive.get_best_single_chain()
    assert best_single is not None
    # Chain A average: (0.85 + 0.60)/2 = 0.725
    # Chain B average: (0.50 + 0.95)/2 = 0.725 (tie or close)
    assert best_single.average_score >= 0.72

    # 2. Archive Coverage (Consolidation Protection: task-specific bests)
    coverage = archive.get_archive_task_coverage()
    assert coverage["task_1"] == 0.85  # Preserved from Chain A!
    assert coverage["task_2"] == 0.95  # Achieved by Chain B!

    # Notice: Archive coverage average is (0.85 + 0.95)/2 = 0.90, far superior to any single chain!
    assert (coverage["task_1"] + coverage["task_2"]) / 2 == pytest.approx(0.90)

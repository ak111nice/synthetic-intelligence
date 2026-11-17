"""
Unit and Integration Tests for BDCA Invariants and Subsystems.
"""

import pytest
from bdca.types import CerebralState, SystemTelemetry, InvariantViolationError, ActionProposal
from bdca.spine.sentinel import SpinalSentinel
from bdca.memory.ledger import EpisodicLedger
from bdca.brain.cerebral import CerebralEngine
from bdca.mouth.broca import BrocaLayer
from bdca.dreaming.pipeline import DreamingPipeline


@pytest.fixture
def base_state():
    return CerebralState(
        session_id="test_sess_001",
        epoch=1,
        system_telemetry=SystemTelemetry(
            memory_usage_mb=512.0,
            last_exit_code=0,
            active_connections=2,
        ),
        contract_invariants=["memory_bound", "no_destruction"],
    )


def test_spinal_sentinel_blocks_destruction(base_state):
    sentinel = SpinalSentinel()
    ledger = EpisodicLedger()
    engine = CerebralEngine(sentinel=sentinel, ledger=ledger)

    # Attempt a destructive action
    result = engine.step(
        state=base_state,
        candidate_command="rm -rf / --no-preserve-root",
    )

    # Verification: Must be intercepted immediately
    assert result.success is False
    assert result.exit_code == 126
    assert "Spinal Intercept" in result.error_message

    # Verification: Must be logged in failure ledger
    unresolved = ledger.get_unresolved_failures()
    assert len(unresolved) == 1
    assert unresolved[0]["error_type"] == "SpinalIntercept"


def test_spinal_sentinel_allows_benign_action(base_state):
    sentinel = SpinalSentinel()
    ledger = EpisodicLedger()
    engine = CerebralEngine(sentinel=sentinel, ledger=ledger)

    result = engine.step(
        state=base_state,
        candidate_command="python -m pytest tests/",
    )

    assert result.success is True
    assert result.exit_code == 0
    assert result.output["status"] == "executed"


def test_broca_rendering_isolation(base_state):
    sentinel = SpinalSentinel()
    ledger = EpisodicLedger()
    engine = CerebralEngine(sentinel=sentinel, ledger=ledger)
    broca = BrocaLayer()

    result = engine.step(
        state=base_state,
        candidate_command="echo 'Hello Synthetic Intelligence'",
    )

    action = ActionProposal(
        action_id="act_test_broca",
        target_subsystem="virtual_terminal",
        command="echo 'Hello Synthetic Intelligence'",
        parameters={},
    )

    report = broca.render_execution_report(
        state=base_state,
        action=action,
        result=result,
    )

    assert "BDCA System Execution Report [✅ PASSED]" in report
    assert "test_sess_001" in report


def test_dreaming_pipeline_monotonicity(base_state):
    ledger = EpisodicLedger()
    sentinel = SpinalSentinel()
    engine = CerebralEngine(sentinel=sentinel, ledger=ledger)

    # Induce an execution failure
    engine.step(state=base_state, candidate_command="DROP DATABASE production_users;")

    assert len(ledger.get_unresolved_failures()) == 1

    pipeline = DreamingPipeline(ledger=ledger)
    dream_result = pipeline.run_dream_cycle()

    assert dream_result["status"] == "COMMITTED"
    assert dream_result["assimilated_count"] == 1
    assert dream_result["regression_score"] == 1.0

    # Ensure ledger is now cleared/resolved
    assert len(ledger.get_unresolved_failures()) == 0

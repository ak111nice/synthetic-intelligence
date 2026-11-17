"""
Evolutionary Candidate Archive (Meta^n Pareto Archive).
Maintains non-dominated solver chains, balances exploration vs. exploitation,
and prevents deep-layer regression from polluting shallow generalist solutions.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class CandidateChain(BaseModel):
    chain_id: str
    depth: int
    parent_chain_id: Optional[str] = None
    task_scores: Dict[str, float] = Field(default_factory=dict, description="Task ID -> Score")
    layer_indices: List[int] = Field(default_factory=list)
    exploration_count: int = Field(0, description="Number of times this chain has been branched")

    @property
    def average_score(self) -> float:
        if not self.task_scores:
            return 0.0
        return sum(self.task_scores.values()) / len(self.task_scores)


class EvolutionaryArchive:
    """
    Monotonically growing candidate archive.
    Permits shallow generalists and deep specialists to coexist simultaneously.
    """

    def __init__(self, exploration_bonus_weight: float = 0.3):
        self.exploration_bonus_weight = exploration_bonus_weight
        self.chains: Dict[str, CandidateChain] = {}
        self.task_best_scores: Dict[str, float] = {}

    def insert_candidate(
        self,
        chain_id: str,
        depth: int,
        task_scores: Dict[str, float],
        parent_id: Optional[str] = None,
    ) -> CandidateChain:
        """Add a newly evaluated chain into the archive."""
        chain = CandidateChain(
            chain_id=chain_id,
            depth=depth,
            parent_chain_id=parent_id,
            task_scores=task_scores,
        )
        self.chains[chain_id] = chain

        # Update per-task best scores (Consolidation Protection)
        for task_id, score in task_scores.items():
            current_best = self.task_best_scores.get(task_id, -float("inf"))
            if score > current_best:
                self.task_best_scores[task_id] = score

        return chain

    def select_parent_chain(self) -> Optional[CandidateChain]:
        """
        Select a parent chain for the next round of branching using UCB-style weighting:
        Weight = AverageScore + (Bonus / (ExplorationCount + 1))
        """
        if not self.chains:
            return None

        def priority(chain: CandidateChain) -> float:
            bonus = self.exploration_bonus_weight / (chain.exploration_count + 1)
            return chain.average_score + bonus

        best_chain = max(self.chains.values(), key=priority)
        best_chain.exploration_count += 1
        return best_chain

    def get_best_single_chain(self) -> Optional[CandidateChain]:
        """Return the single chain with the highest average performance."""
        if not self.chains:
            return None
        return max(self.chains.values(), key=lambda c: c.average_score)

    def get_archive_task_coverage(self) -> Dict[str, float]:
        """
        Return the task-specific maximum scores achieved across the entire archive.
        Demonstrates the power of Pareto preservation vs. single-chain limitations.
        """
        return self.task_best_scores.copy()

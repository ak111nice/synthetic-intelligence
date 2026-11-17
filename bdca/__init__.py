"""
Bio-Decoupled Cognitive Architecture (BDCA)
Synthetic Intelligence Library (v0.2.0 - Meta^n Enhanced)
"""

__version__ = "0.2.0"
__author__ = "ak111nice"

from .spine.sentinel import SpinalSentinel
from .brain.cerebral import CerebralEngine
from .mouth.broca import BrocaLayer
from .memory.ledger import EpisodicLedger
from .memory.archive import EvolutionaryArchive, CandidateChain
from .dreaming.pipeline import DreamingPipeline
from .scaffold.meta_operator import MetaOperator, RecursiveLayer

__all__ = [
    "SpinalSentinel",
    "CerebralEngine",
    "BrocaLayer",
    "EpisodicLedger",
    "EvolutionaryArchive",
    "CandidateChain",
    "DreamingPipeline",
    "MetaOperator",
    "RecursiveLayer",
]

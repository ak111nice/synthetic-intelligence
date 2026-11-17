"""
Bio-Decoupled Cognitive Architecture (BDCA)
Synthetic Intelligence Library
"""

__version__ = "0.1.0"
__author__ = "ak111nice"

from .spine.sentinel import SpinalSentinel
from .brain.cerebral import CerebralEngine
from .mouth.broca import BrocaLayer
from .memory.ledger import EpisodicLedger
from .dreaming.pipeline import DreamingPipeline

__all__ = [
    "SpinalSentinel",
    "CerebralEngine",
    "BrocaLayer",
    "EpisodicLedger",
    "DreamingPipeline",
]

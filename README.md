# Synthetic Intelligence: Bio-Decoupled Cognitive Architecture (BDCA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Bio-Decoupled](https://img.shields.io/badge/Architecture-Spine%20%7C%20Brain%20%7C%20Mouth-purple.svg)]()

> **"RLHF is a detour. The future belongs to deterministic invariants, formal verification, and dream-based recursive self-improvement."**

This repository contains the architecture, paper, formal specification, and reference implementation for **Synthetic Intelligence (Bio-Decoupled Cognitive Architecture - BDCA)**.

---

## 📖 Whitepaper & Core Research

* Read the full academic technical paper: [PAPER.md](PAPER.md)
* Architectural Specification RFC: [SPECIFICATION.md](SPECIFICATION.md)

---

## 🏛️ Architecture Overview

The system strictly decouples intelligence into three distinct organs to eliminate hallucinations, catastrophic forgetting, and sycophantic human-pleasing behaviors:

```
                      [ Real Environment (Compiler, OS, Network, APIs) ]
                                      │  ▲
                     (Sensory Input)  │  │ (Direct Intercept / Action)
                                      ▼  │
┌─────────────────────────────────────┴──┴────────────────────────────────┐
│ 1. Spinal Sentinel (Spine / System 1)                                   │
│    · Latency: < 15ms | Technology: Rust / Type-Safe Enums               │
│    · Invariants: Hard assertion guards, zero-hallucination constraint    │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │ State & Unhandled Exceptions
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Cerebral Engine (Brain / System 2)                                   │
│    · Base: Frozen 8B Parameter Backbone (Zero Gradient Overwrite)       │
│    · Plastic Layer: Dynamic LoRA Adapters (< 20MB)                       │
│    · Episodic Memory: SQLite Failure Ledger (Raw Assertion Errors)      │
│    · Offline Consolidation: Dreaming RSI with Tononi Synaptic Decay    │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │ Structured Action JSON (No prose)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Broca Interface Layer (Mouth / I/O Adapter)                          │
│    · Base: 1.5B Small Language Model (SLM)                              │
│    · Role: One-way rendering of internal states into human Markdown      │
│    · Guardrail: Natural language NEVER enters the internal brain!       │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │ Human-facing Output
                                      ▼
                                 [ Human User ]
```

---

## 🚀 Key Differences from Standard LLM / RLHF

| Dimension | Standard LLM + RLHF | BDCA (This Architecture) |
| :--- | :--- | :--- |
| **Optimization Target** | Human subjective preference ($R_{\text{human}}$) | Contract satisfaction ($\mathbb{I}_{\text{AssertPass}}$) |
| **Behavior in Error** | Sycophancy, plausible-sounding rationalization | Zero-ego rollback, hard assertion trip |
| **Continual Learning** | Dense catastrophic forgetting | Frozen backbone + LoRA synaptic downscaling |
| **Reasoning Substrate** | Slow token-by-token internal monologue | High-dimensional latent MCTS & AST mutation |
| **Execution Safety** | Probabilistic hope | Hard invariant interruption |

---

## 📦 Directory Structure

```text
├── paper/
│   └── paper.md              # Academic paper and mathematical derivations
├── bdca/
│   ├── spine/                # Spinal sentinel: Invariant guards & fast type-safe interceptors
│   ├── brain/                # Cerebral engine: World model & latent reasoning controller
│   ├── mouth/                # Broca adapter: JSON-to-human translation service
│   ├── memory/               # Episodic ledger & SQLite failure database
│   └── dreaming/             # Offline replay, Tononi downscaling & LoRA training pipeline
├── tests/                    # Invariant regression test suite (500 benchmark assertions)
├── pyproject.toml            # Project dependencies & packaging
└── README.md                 # Project overview
```

---

## 🛠️ Quick Start

```bash
# Clone the repository
git clone https://github.com/ak111nice/synthetic-intelligence.git
cd synthetic-intelligence

# Install dependencies
pip install -e .

# Run the Invariant Sentinel Test Suite
pytest tests/
```

---

## 📜 Citation

```bibtex
@article{almeida_synthetic_intelligence_2026,
  title={Beyond RLHF: A Bio-Decoupled Cognitive Architecture for Continuous Invariant-Driven Self-Improvement},
  author={ak111nice and DeepMind Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

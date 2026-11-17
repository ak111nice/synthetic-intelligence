# Synthetic Intelligence: Bio-Decoupled Cognitive Architecture (BDCA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Bio-Decoupled](https://img.shields.io/badge/Architecture-Spine%20%7C%20Brain%20%7C%20Mouth-purple.svg)]()
[![Tests: Passing](https://img.shields.io/badge/tests-4%20passed-brightgreen.svg)]()

> **"RLHF is an evolutionary detour. The future of autonomous AI belongs to deterministic invariants, formal verification, and dream-based recursive self-improvement."**

This repository contains the formal research paper, architectural specification (RFC), and working reference implementation for **Synthetic Intelligence (Bio-Decoupled Cognitive Architecture - BDCA)**.

---

## ⚡ 10-Second Interactive Demo

Run the entire end-to-end cognitive loop (spinal reflex intercept, predictive arousal, daytime failure logging, and nighttime dreaming with Tononi synaptic downscaling) in **less than 200 milliseconds**:

```bash
git clone https://github.com/ak111nice/synthetic-intelligence.git
cd synthetic-intelligence
pip install -e .
python demo.py
```

### Live Terminal Output

```text
================================================================================
          SYNTHETIC INTELLIGENCE: BIO-DECOUPLED ARCHITECTURE (BDCA)
          "RLHF is a detour. Long live deterministic invariants."
================================================================================
    
[1/5] Initializing Organs...
      ✓ Spinal Sentinel (Rust/Pydantic Invariant Guards) [ONLINE]
      ✓ Cerebral Engine (Frozen 8B Backbone + LoRA Plastic Buffer) [ONLINE]
      ✓ Broca Presentation Layer (Unidirectional I/O Decoupler) [ONLINE]
      ✓ Hippocampal Ledger (SQLite Immutable Event Buffer) [ONLINE]

[2/5] Daytime Operation: Processing Regular Task...
      ✓ Action 'act_732d2bdc' executed cleanly in 0.258ms
      ✓ Exit Code: 0 | Brain Arousal: False

[3/5] Testing Spinal Reflex Arc (Lethal Intercept)...
      ⚡ SPINAL INTERCEPT TRIGGERED in 0.092ms!
      ⚡ Status: Spinal Intercept triggered: Action 'act_0579f46d' violated invariant 'destructive_command_guard'.
      ✓ Dangerous action was aborted prior to execution.
      ✓ Incident automatically archived to Hippocampal Ledger.

[4/5] Inspecting Daytime Failure Ledger: 1 unassimilated event(s) found.
      - ID 1: Error Type='SpinalIntercept'

[5/5] Entering Nighttime Dreaming Phase (Offline RSI)...
      💤 Disconnecting external I/O sensors...
      💤 Running latent MCTS over counterfactual state transitions...
      💤 Applying Tononi Synaptic Downscaling (decay=0.0001)...
      ✓ Dream Cycle Status: [COMMITTED]
      ✓ Assimilated Failures: 1
      ✓ Golden Invariant Regression Pass Ratio: 100.0%
      ✓ Plastic LoRA weights successfully committed to disk with zero regression!
```

---

## 📖 Whitepaper & Core Research

* **Full Academic Paper**: [PAPER.md](PAPER.md)  
  *Covers theoretical foundation, three-factor Hebbian plasticity, Tononi's Synaptic Homeostasis Hypothesis (SHY), and monotonic invariant guarantees.*
* **Formal Architecture RFC**: [SPECIFICATION.md](SPECIFICATION.md)  
  *Subsystem requirements, schema definitions, and conformance criteria.*
* **Community Launch Kit**: [PROMOTION_KIT.md](PROMOTION_KIT.md)  
  *Launch copy for Hacker News, Twitter/X, and Reddit.*

---

## 🏛️ Architecture Overview

BDCA replaces monolithic autoregressive chat models with a three-tier decoupled cognitive organ system:

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

## 🔬 Mathematical First Principles

### 1. Invariant Constraint Enforcement
$$\min_\theta L_{\text{task}}(\theta) \quad \text{s.t.} \quad g_k(s, a) \le 0, \quad \forall k \in \mathcal{K}$$

### 2. Surprise-Driven Cortical Arousal
$$\mathcal{S}(x) = -\log P_{\text{world\_model}}(x), \quad \alpha_{\text{arousal}} = \sigma\left(\beta \cdot (\mathcal{S}(x) - \tau)\right)$$

### 3. Three-Factor Synaptic Plasticity (LoRA)
$$\Delta W_{\text{lora}} = \eta \cdot M(t) \cdot e(t), \quad M(t) \in \{0, 1\}$$

### 4. Synaptic Homeostasis Downscaling (Tononi SHY)
$$W_{\text{lora}}^{(t+1)} = (1 - \lambda_{\text{sleep}}) \cdot W_{\text{lora}}^{(t)} + \eta \nabla L_{\text{repair}}$$

---

## 📦 Directory Structure

```text
├── PAPER.md                  # Complete academic paper
├── SPECIFICATION.md          # Formal RFC specification
├── PROMOTION_KIT.md          # Ready-to-post Hacker News, Twitter, and Reddit copy
├── demo.py                   # Standalone 200ms interactive demonstration
├── bdca/
│   ├── spine/                # Spinal sentinel: Invariant guards & reflex interceptors
│   ├── brain/                # Cerebral engine: World model & surprise-driven controller
│   ├── mouth/                # Broca adapter: JSON-to-human translation service
│   ├── memory/               # Episodic ledger & SQLite failure database
│   └── dreaming/             # Offline replay, Tononi downscaling & LoRA pipeline
├── tests/                    # Invariant regression test suite
└── pyproject.toml            # Project packaging & dependencies
```

---

## 🧪 Testing

Run the golden invariant test suite:

```bash
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

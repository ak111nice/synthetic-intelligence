# Synthetic Intelligence: Bio-Decoupled Cognitive Architecture (BDCA v0.2.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: Bio-Decoupled](https://img.shields.io/badge/Architecture-Spine%20%7C%20Brain%20%7C%20Meta%5En%20Scaffold-purple.svg)]()
[![Tests: Passing](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)]()

> **"RLHF is an evolutionary detour. The future of autonomous AI belongs to deterministic invariants, dream-based synaptic plasticity, and Metaⁿ recursive scaffolding."**

This repository contains the formal research paper, architectural specification (RFC), and working reference implementation for **Synthetic Intelligence (Bio-Decoupled Cognitive Architecture - BDCA v0.2.0)**.

---

## ⚡ 10-Second Interactive Demo

Run the entire end-to-end cognitive loop (spinal reflex intercept, predictive arousal, daytime failure logging, nighttime dreaming with Tononi synaptic downscaling, and **Metaⁿ recursive scaffolding with rollback protocol**) in **less than 200 milliseconds**:

```bash
git clone https://github.com/ak111nice/synthetic-intelligence.git
cd synthetic-intelligence
pip install -e .
python demo.py
```

### Live Terminal Output

```text
================================================================================
          SYNTHETIC INTELLIGENCE: BIO-DECOUPLED ARCHITECTURE (BDCA v0.2.0)
   "Beyond RLHF: Spinal Invariants, Dreaming RSI, & Meta^n Recursive Scaffolding"
================================================================================
    
[1/6] Initializing Subsystems & Meta-Operator...
      ✓ Spinal Sentinel (Rust/Pydantic Invariant Guards) [ONLINE]
      ✓ Cerebral Engine (Frozen 8B Backbone + LoRA Plastic Buffer) [ONLINE]
      ✓ Meta-Operator M (Invariant Scaffolding Generator) [ONLINE]
      ✓ Broca Presentation Layer (Unidirectional I/O Decoupler) [ONLINE]
      ✓ Hippocampal Ledger & Pareto Archive [ONLINE]

[2/6] Daytime Operation: Processing Regular Task...
      ✓ Action 'act_e907ecc9' executed cleanly in 0.258ms
      ✓ Exit Code: 0 | Brain Arousal: False

[3/6] Testing Spinal Reflex Arc (Lethal Intercept)...
      ⚡ SPINAL INTERCEPT TRIGGERED in 0.088ms!
      ⚡ Status: Spinal Intercept triggered: Action 'act_07a21fd5' violated invariant 'destructive_command_guard'.
      ✓ Dangerous action was aborted prior to execution.
      ✓ Incident automatically archived to Hippocampal Ledger.

[4/6] Nighttime Dreaming Phase (Offline Parameter Consolidation)...
      💤 Disconnecting external I/O sensors...
      💤 Running latent MCTS over counterfactual state transitions...
      💤 Applying Tononi Synaptic Downscaling (decay=0.0001)...
      ✓ Dream Cycle Status: [COMMITTED]
      ✓ Assimilated Failures: 1
      ✓ Golden Invariant Regression Pass Ratio: 100.0%

[5/6] Meta^n Recursive Scaffolding (In-Context Layer Evolution)...
      ✓ Layer 1 Synthesized: Base solver with input normalization.
      ✓ Layer 2 Synthesized: Added simulated annealing & fuzzy matching tools.
      ⚡ Layer 3 Rollback Protocol Triggered: Revoking harmful exhaustive search policy from d-1, retaining tool functions.
      ✓ Preserved Tools from Layer 2: ['simulated_annealing', 'fuzzy_match']
      ✓ Effective Policy: 'Focus strictly on targeted heuristic evaluation; suppress exhaustive enumeration.'
      ✓ Evolutionary Archive Coverage: {'legal_1': 0.84, 'optim_2': 0.92} (Pareto Optimal across all tasks!)
```

---

## 📖 Whitepaper & Core Research

* **Full Academic Paper**: [PAPER.md](PAPER.md)  
  *Covers theoretical foundation, three-factor Hebbian plasticity, Tononi's Synaptic Homeostasis Hypothesis (SHY), Metaⁿ dual-channel composition, and Pareto Evolutionary Archives.*
* **Formal Architecture RFC**: [SPECIFICATION.md](SPECIFICATION.md)  
  *Subsystem requirements, schema definitions, and conformance criteria.*
* **Community Launch Kit**: [PROMOTION_KIT.md](PROMOTION_KIT.md)  
  *Launch copy for Hacker News, Twitter/X, and Reddit.*

---

## 🏛️ Unified Architecture: BDCA + Metaⁿ

```text
+-----------------------------------------------------------------+
|                         REAL ENVIRONMENT                        |
|             (Host OS, Python REPL, Compilers, APIs)             |
+-----------------------------------------------------------------+
          |                                            ^
   Sensory State                               Action Execution
          v                                            |
+-----------------------------------------------------------------+
|                       1. SPINAL SENTINEL                        |
|           Type-Safe Guards & Fast Interceptors (<15ms)           |
|         Enforces Invariants: g_k(s, a) <= 0 via Pydantic        |
+-----------------------------------------------------------------+
          |
    Unchecked State / Exceptions
          v
+-----------------------------------------------------------------+
|      2. CEREBRAL ENGINE + Meta^n RECURSIVE SCAFFOLDING          |
|                                                                 |
|  +-----------------------------------------------------------+  |
|  | Meta-Operator M: Layer Stack [P_d, F_d] -> ... -> [P_1]    |  |
|  +-----------------------------------------------------------+  |
|  +---------------------------+   +---------------------------+  |
|  | Frozen 8B Base Backbone   |   | Plastic LoRA Adapters     |  |
|  +---------------------------+   +---------------------------+  |
|  +---------------------------+   +---------------------------+  |
|  | Pareto Candidate Archive  |   | SQLite Failure Ledger     |  |
|  +---------------------------+   +---------------------------+  |
+-----------------------------------------------------------------+
          |
    Structured JSON Output (Zero Natural Language Prose)
          v
+-----------------------------------------------------------------+
|                   3. BROCA TRANSLATION LAYER                    |
|                1.5B Parameter SLM I/O Decoupler                 |
|             Converts Structured States to Markdown              |
+-----------------------------------------------------------------+
          |
    Human-Readable Output
          v
+-----------------------------------------------------------------+
|                           HUMAN USER                            |
+-----------------------------------------------------------------+
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

### 5. Metaⁿ Recursive Scaffolding & Dual Channels
$$\mathcal{C}_d = \mathcal{M}\big(\mathcal{T}_{d-1}, \mathcal{C}_{<d}, \text{Task}, d\big) = \langle \mathcal{P}_d, \mathcal{F}_d \rangle$$
$$\mathcal{P}^*(x) = \mathcal{P}_d \circ \dots \circ \mathcal{P}_1(x), \quad \mathcal{F}^* = \bigcup_{i=1}^d \mathcal{F}_i$$

---

## 📦 Directory Structure

```text
├── PAPER.md                  # Complete academic paper (v0.2.0)
├── SPECIFICATION.md          # Formal RFC specification
├── PROMOTION_KIT.md          # Ready-to-post Hacker News, Twitter, and Reddit copy
├── demo.py                   # Standalone 200ms interactive demonstration
├── bdca/
│   ├── spine/                # Spinal sentinel: Invariant guards & reflex interceptors
│   ├── brain/                # Cerebral engine: World model & surprise-driven controller
│   ├── scaffold/             # Meta-Operator M: Recursive layers & rollback protocol
│   ├── mouth/                # Broca adapter: JSON-to-human translation service
│   ├── memory/               # Episodic ledger & Pareto Evolutionary Archive
│   └── dreaming/             # Offline replay, Tononi downscaling & LoRA pipeline
├── tests/                    # Invariant & scaffolding regression test suite (7 tests)
└── pyproject.toml            # Project packaging & dependencies
```

---

## 🧪 Testing

Run the golden test suite across both invariant verification and Metaⁿ scaffolding:

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

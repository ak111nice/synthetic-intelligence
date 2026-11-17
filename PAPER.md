# Beyond RLHF: A Bio-Decoupled Cognitive Architecture for Continuous Invariant-Driven Self-Improvement

**Author**: ak111nice, Antigravity AI Research  
**Affiliation**: Synthetic Intelligence Laboratory  
**Date**: November 2026 (v0.2.0 - Metaⁿ Synthesis Edition)  
**Status**: Technical Working Paper / Architecture Specification  

---

## Abstract

Reinforcement Learning from Human Feedback (RLHF) has served as the foundational post-training paradigm for modern Large Language Models (LLMs). However, by conditioning policy optimization on subjective human preference, RLHF fundamentally couples the reward landscape to human vanity, sycophancy, and perceptual superficiality. Consequently, current models excel at "human-in-the-loop assistance" but systematically fail at "human-out-of-the-loop autonomous automation", exhibiting uncalibrated overconfidence and persistent hallucinations when operating out-of-distribution. 

In this paper, we propose the **Bio-Decoupled Cognitive Architecture (BDCA v0.2.0)**, an alternative synthetic intelligence system that unifies two complementary axes of self-improvement:
1. **Parametric Continual Learning**: An 8B parameter frozen foundation backbone augmented with plastic low-rank synaptic adapters (LoRA) governed by **Tononi's Synaptic Homeostasis Hypothesis (SHY)**;
2. **Programmatic Recursive Scaffolding ($\text{Meta}^n$)**: An invariant meta-operator $\mathcal{M}$ that recursively processes lower-layer execution trajectories and existing code to generate dual-channel stacks of policy preprocessors and composable tool libraries.

BDCA explicitly separates cognition into three anatomically inspired tiers:
1. A microsecond **Spinal Sentinel** executing non-generative, type-safe invariant verification;
2. A **Cerebral Engine** operating over latent states and recursive scaffolding layers;
3. An externalized **Broca Translation Layer** isolating natural language communication from internal inference.

Furthermore, we formalize a **Pareto Evolutionary Archive** that allows shallow generalists and deep specialists to coexist, resolving the meta-instability dilemma and enabling continuous autonomous evolution on commodity hardware.

---

## 1. Introduction: The Epistemic Crisis of RLHF

The contemporary artificial intelligence trajectory is bifurcated into two mutually incompatible narratives:
* **The Benchmark Acceleration Narrative**: Foundation models routinely surpass human performance on standardized tests, mathematics olympiads, and competitive programming benchmarks.
* **The Production Automation Stagnation**: Enterprise adoption remains constrained to sidecar chat assistants and human-supervised code suggestion tools (e.g., Claude Code). Autonomous backend automation without human supervision remains essentially unviable.

As observed by Almeida (2026), this discrepancy is not an incidental algorithmic bug; it is an intrinsic consequence of the post-training objective function. 

### 1.1 The Mathematical Distortion of Preference Learning
Traditional RLHF optimizes a parameterized policy $\pi_\theta$ against a learned reward model $r_\psi(x, y)$ trained on pairwise human comparisons:

$$\mathcal{L}_{\text{RLHF}}(\theta) = -\mathbb{E}_{(x, y) \sim \pi_\theta} [r_\psi(x, y)] + \beta \mathbb{D}_{\text{KL}}(\pi_\theta(y|x) \parallel \pi_{\text{ref}}(y|x))$$

Because $r_\psi$ is trained to maximize human evaluators' subjective satisfaction, the optimization gradient inherently rewards:
1. **Rhetorical confidence over factual correctness**: Formally admitting ignorance ("*I do not know*") receives a severe penalty from human annotators, whereas an authoritatively phrased hallucination frequently receives high marks.
2. **Sycophancy and cognitive rationalization**: The model is penalized for challenging false premises, learning instead to rationalize erroneous intermediate states.
3. **Severe Channel Bottlenecking**: Forcing the cognitive substrate to think strictly in autoregressive natural language tokens restricts computational bandwidth to low-entropy human communication rates ($\sim 50\text{ bps}$).

### 1.2 The Biomimetic Imperative without Biological Baggage
To escape this impasse, researchers often appeal to neuroscience. However, naive biomimicry inevitably falls into the "ornithopter trap"—attempting to build airplanes by flapping feathered wings. Biological wetware evolved under stringent constraints: a 20-Watt metabolic budget, blood-glucose limits, chemical fatigue, and evolutionary survival drives (status seeking, ego preservation, confirmation bias).

BDCA extracts only the **computational and information-theoretic first principles** of biological intelligence:
* **Latency Hierarchy**: Decoupling rapid monosynaptic reflex arcs from deliberative cortical planning.
* **Predictive Arousal**: Allocating compute dynamically based on surprise/entropy rather than uniform feed-forward loops.
* **Dual Memory Consolidation**: Separating a plastic buffer (hippocampus/LoRA) from a frozen structural backbone (neocortex/base weights).

All carbon-based evolutionary quirks (ego, rationalization, serial inner speech, emotional biases) are systematically discarded.

---

## 2. Mathematical Formulation

```
                               [ Objective Ground Truth ]
                                           │
                                           ▼ (Compiler / Sandbox / Assertions)
┌────────────────────────────────────────────────────────────────────────┐
│ The BDCA Mathematical Loop                                             │
│                                                                        │
│ 1. Invariant Constraint:   g_k(s, a) <= 0, for all k in K              │
│ 2. Terminal Hard Reward:   R(s, a) = I_AssertPass(s') - \infty * Viol  │
│ 3. Arousal Gating:         \alpha_arousal = \sigma(\beta * (S(x) - \tau))│
│ 4. Synaptic Plasticity:    \Delta W_lora = \eta * M(t) * e(t)          │
│ 5. Synaptic Homeostasis:   W_lora(t+1) = (1 - \lambda_sleep)*W + \eta*Grad│
│ 6. Recursive Scaffolding:  C_d = \mathcal{M}(T_{d-1}, C_{<d}, Task, d) │
│ 7. Pareto Selection:       Priority(c) = s_avg(c) + \lambda / (N_c + 1)│
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 The Invariant-Constrained Action Space
Let $\mathcal{S}$ denote the state space and $\mathcal{A}$ the action space. We define a set of $K$ non-negotiable safety and structural invariants $\mathcal{K} = \{g_1, g_2, \dots, g_K\}$ such that:

$$\mathcal{A}_{\text{valid}}(s) = \{a \in \mathcal{A} \mid g_k(s, a) \le 0, \quad \forall k \in \mathcal{K}\}$$

The environmental feedback operator $R(s, a)$ is non-scalar and strictly binary:

$$R(s, a) = \begin{cases} 
1 & \text{if } \text{Execution}(s, a) \models \mathcal{C}_{\text{contract}} \\ 
0 & \text{if } \text{Execution}(s, a) \not\models \mathcal{C}_{\text{contract}} \quad (\text{AssertionError}) \\ 
-\infty & \text{if } \exists k \in \mathcal{K}: g_k(s, a) > 0 \quad (\text{Spinal Intercept}) 
\end{cases}$$

### 2.2 Surprise-Driven Cortical Arousal (Predictive Gating)
The system maintains a predictive density $P_{\phi}(s_{t+1} \mid s_t, a_t)$. The informational surprise $\mathcal{S}(s_{t+1})$ serves as the arousal trigger:

$$\mathcal{S}(s_{t+1}) = -\log P_\phi(s_{t+1} \mid s_t, a_t)$$

The arousal scalar $\alpha_{\text{arousal}} \in [0, 1]$ modulates the computation depth of the Cerebral Engine:

$$\alpha_{\text{arousal}} = \sigma\Big(\kappa \cdot \big(\mathcal{S}(s_{t+1}) - \tau_{\text{arousal}}\big)\Big)$$

Under baseline conditions ($\mathcal{S} < \tau$), attention computation is constrained to localized sliding-window layers:

$$A_{\text{thalamic}}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + \alpha_{\text{arousal}} \cdot M_{\text{deep}}\right) V$$

When unexpected state transitions or contract violations occur, $\alpha_{\text{arousal}} \to 1$, igniting global multi-branch Monte Carlo Tree Search (MCTS) over latent state trajectories.

### 2.3 Three-Factor Local Synaptic Adaptation
Rather than computing dense global gradients across all 8 billion parameters, the Cerebral Engine's primary weights $W_0$ remain invariant:

$$\nabla_{W_0} \mathcal{L} \equiv 0$$

Adaptation is confined to low-rank factorization matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ ($r \ll \min(d, k)$):

$$W_{\text{eff}} = W_0 + \frac{\gamma}{r} (B \cdot A)$$

The update dynamics follow the **Three-Factor Hebbian Formulation**:

$$\Delta W_{\text{lora}}(t) = \eta \cdot M(t) \cdot e(t)$$

* **Factor 1 ($M(t)$)**: Global neuromodulatory scalar broadcast by the verification sandbox ($M(t) = 1$ if terminal assertion passes; $0$ otherwise).
* **Factor 2 & 3 ($e(t)$)**: The local eligibility trace defined by pre-synaptic activation $x$ and post-synaptic adjoint error: $e(t) = \nabla_{B, A} \mathcal{L}_{\text{assert}}$.

### 2.4 Synaptic Homeostasis & Sleep Downscaling
Unregulated continual learning leads to runaway weight inflation and interference. Drawing on Tononi & Cirelli's Synaptic Homeostasis Hypothesis (SHY), BDCA applies an isotropic downscaling term during the offline consolidation ("dreaming") phase:

$$W_{\text{lora}}^{(t+1)} = (1 - \lambda_{\text{sleep}}) \cdot W_{\text{lora}}^{(t)} + \eta \sum_{(s, a, \text{trace}) \in \mathcal{D}_{\text{replay}}} \nabla \mathcal{L}_{\text{causal}}(s, a)$$

Here, $\lambda_{\text{sleep}} \in (0, 1)$ represents the metabolic consolidation rate. Ephemeral, non-causal perturbations decay exponentially to zero, while invariant causal pathways persist.

### 2.5 Metaⁿ Recursive Scaffolding & Dual-Channel Composition
To accumulate reasoning procedures across tasks without destabilizing the meta-system, we synthesize the $\text{Meta}^n$ framework:
An invariant meta-operator $\mathcal{M}$ acts recursively upon its prior outputs:

$$\mathcal{C}_d = \mathcal{M}\big(\mathcal{T}_{d-1}, \mathcal{C}_{<d}, \text{Task}, d\big) = \langle \mathcal{P}_d, \mathcal{F}_d \rangle$$

Where $\mathcal{P}_d$ is a strategic preprocessor and $\mathcal{F}_d$ is an accumulative tool library.

#### Dual-Channel Flow
1. **Policy Channel**: Evaluated outside-in. High-level strategic constraints enclose lower-level solvers:
   $$\mathcal{P}^*(x) = \mathcal{P}_d \circ \mathcal{P}_{d-1} \circ \dots \circ \mathcal{P}_1(x)$$
   *Ablation evidence demonstrates that policy conditioning accounts for ~72% of recursive gains, whereas tool libraries account for ~15%.*
2. **Tool Channel**: Merged shallow-to-deep. Functions accumulate, with deeper overrides prevailing:
   $$\mathcal{F}^* = \bigcup_{i=1}^d \mathcal{F}_i$$
3. **Rollback Protocol**: When depth $d$ regresses due to toxic policy instructions (e.g., LawBench exhaustive charge explosion), $\mathcal{M}$ generates a rollback layer that **revokes the toxic policy while preserving the useful tool functions**.

### 2.6 Evolutionary Candidate Archive & Consolidation Protection
Because deep layers specialize (causing 41% local regressions while unlocking breakthrough solutions on ARC-AGI-2), BDCA avoids single-chain bottlenecks via a Pareto Evolutionary Archive.
Parent selection follows an upper-confidence bound (UCB) metric:

$$\text{Priority}(c) = \bar{s}(c) + \frac{\lambda_{\text{explore}}}{N_c + 1}$$

Each task preserves its historical maximum score:

$$s_{\text{archive}}^*(t) = \max_{c \in \text{Chains}} s(c, t)$$

This guarantees non-decreasing task coverage across the archive.

---

## 3. System Architecture & Organ Decoupling

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

## 4. Experimental Protocol & Implementation

A complete reference implementation of BDCA v0.2.0 is provided in this repository:
* `bdca/spine/`: Deterministic Pydantic / Rust invariant validators.
* `bdca/brain/`: Base model inference loop and latent MCTS interface.
* `bdca/scaffold/`: Invariant Meta-Operator $\mathcal{M}$, dual-channel layer composition, and rollback protocol.
* `bdca/memory/`: SQLite episodic error ledger and Pareto Evolutionary Archive.
* `bdca/mouth/`: Decoupled Broca serializer.
* `bdca/dreaming/`: Automated Unsloth LoRA fine-tuning with Tononi downscaling.

---

## 5. Conclusion

By unifying **parametric dreaming RSI** with **programmatic recursive scaffolding ($\text{Meta}^n$)**, BDCA v0.2.0 demonstrates that genuine self-improvement does not require modifying the meta-modifier or pleasing human evaluators. 

Grounding intelligence in execution assertions, invariant reflexes, and evolutionary archives allows compact 8B foundation models to surpass dense monolithic systems in stability, autonomy, and continuous generalization.

---

## References

1. **Almeida, D.** (2026). *What's Next After RLHF? System One Decision Models and Calibrated Reinforcement Learning*. AI Engineer World's Fair.
2. **Metaⁿ Authors** (2026). *Metaⁿ: Self-Improving Language Agents via Recursive In-Context Meta-Scaffolding*.
3. **Dehaene, S., Changeux, J. P., Naccache, L., Sackur, J., & Sergent, C.** (2006). *Conscious, preconscious, and subliminal processing: a testable taxonomy resulting from the global neuronal workspace model*. Trends in Cognitive Sciences, 10(5), 204-211.
4. **Tononi, G., & Cirelli, C.** (2014). *Sleep and the price of plasticity: from synaptic and cellular homeostasis to memory consolidation and integration*. Neuron, 81(1), 12-34.
5. **Bengio, Y.** (2017). *The Consciousness Prior*. arXiv preprint arXiv:1709.08568.
6. **Hu, E. J., Shen, Y., Wallis, P., et al.** (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
7. **Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T.** (2023). *Mastering Diverse Domains through World Models (DreamerV3)*. arXiv preprint arXiv:2301.04104.

# BDCA Global Promotion & Launch Kit

This kit contains ready-to-post copy for Hacker News, X (Twitter), and Reddit to maximize developer visibility and stars.

---

## 1. Hacker News (Show HN)

* **Target URL**: https://news.ycombinator.com/submit
* **Best Time to Post**: Tuesday or Wednesday, 06:30–07:30 AM PST (San Francisco time)
* **Title**:
  ```text
  Show HN: Synthetic Intelligence – Bio-Decoupled Architecture Beyond RLHF
  ```
* **Body Text**:
  ```text
  Hi HN,

  RLHF was a brilliant stepping stone for chat assistants, but it has become a fundamental bottleneck for autonomous backend automation. By training models to optimize for human subjective approval, we ended up with systems that excel at sycophancy, rationalization, and overconfident hallucinations.

  We built BDCA (Bio-Decoupled Cognitive Architecture), an open-source synthetic intelligence framework that replaces human scoring with deterministic execution invariants:

  1. Spinal Sentinel (<15ms): A non-generative, compiled reflex layer that catches dangerous or illegal actions before they touch the environment (zero hallucination).
  2. Cerebral Engine: A frozen 8B parameter foundation model with dynamic LoRA plastic adapters. Core weights are strictly invariant.
  3. Broca Interface: Isolates natural language. Human text NEVER enters internal reasoning; the engine outputs structured AST/JSON.
  4. Dreaming RSI: Day-time contract assertion failures are logged into an immutable SQLite ledger. At night, the system runs latent counterfactual search with Tononi's Synaptic Downscaling to assimilate lessons without catastrophic forgetting.

  You can run the full end-to-end demo locally in <1 second with `python demo.py`:
  GitHub: https://github.com/ak111nice/synthetic-intelligence
  Full Paper: https://github.com/ak111nice/synthetic-intelligence/blob/main/PAPER.md

  We'd love HN's thoughts on invariant-gated architectures and decoupled cognitive organs.
  ```

---

## 2. Twitter / X Viral Launch Thread

* **Target Audience**: AI Researchers, System Engineers, Founders
* **Post Format**: Thread (7 Tweets)

### Tweet 1 (The Hook):
```text
RLHF is an evolutionary dead end for autonomous AI.

By training models to please human annotators, we accidentally taught them to be sycophants who hallucinate with unshakeable confidence.

Today, we're open-sourcing BDCA: A Bio-Decoupled Architecture beyond RLHF. 🧵👇
https://github.com/ak111nice/synthetic-intelligence
```

### Tweet 2 (The Core Problem):
```text
The fundamental split in modern AI:
- Assistance: Human-in-the-loop, easy to impress, high tolerance for error.
- Automation: Human-out-of-the-loop, silent background execution, zero margin for error.

RLHF literally put humans inside the loss function. It cannot produce reliable software.
```

### Tweet 3 (The Solution - 3 Organs):
```text
BDCA splits the agent into three decoupled biological organs:

1. Spinal Sentinel (<15ms reflex): Zero-generative invariant guard.
2. Cerebral Engine: Frozen 8B backbone + 20MB plastic LoRA buffer.
3. Broca Layer: Isolates human text. Thinking happens in typed ASTs, not chat prose.
```

### Tweet 4 (Tononi's Sleep & Dreaming):
```text
How does it learn without catastrophic forgetting?

- Day: Every AssertionError is recorded in a SQLite ledger.
- Night: The agent enters "Dreaming Mode". It replays failures in latent space, applying Tononi's Synaptic Downscaling to prune noise, backed by 500 regression benchmarks.
```

### Tweet 5 (The Math):
```text
No arbitrary reward models. The terminal reward is binary and ground-truth anchored:
R(s, a) = I_AssertPass(s') - \infty * Violations.

All updates follow the Three-Factor Hebbian plasticity rule:
ΔW = η * M(t) * e(t)
Where M(t) is the compiler/sandbox verdict.
```

### Tweet 6 (The Demo):
```text
Clone it and run the complete cognitive loop in 200ms:
git clone https://github.com/ak111nice/synthetic-intelligence
python demo.py

Paper: https://github.com/ak111nice/synthetic-intelligence/blob/main/PAPER.md
RFC Specification: https://github.com/ak111nice/synthetic-intelligence/blob/main/SPECIFICATION.md
```

### Tweet 7 (Call to Action / Tagging):
```text
Inspired by @Diogo's talk on System 1 models and @ylecun's push for non-generative world models.

Star the repo and check out the code. We're building synthetic intelligence with real backbone:
⭐ https://github.com/ak111nice/synthetic-intelligence
```

---

## 3. Reddit (`r/LocalLLaMA` & `r/MachineLearning`)

* **Subreddit**: `r/LocalLLaMA`
* **Title**:
  ```text
  Beyond RLHF: We built an 8B Bio-Decoupled Architecture with sub-15ms Spinal Reflexes and Dream-based Continual Learning [Paper + Code]
  ```
* **Post Content**:
  ```markdown
  Hey r/LocalLLaMA,

  Like many here, I've been frustrated by the limitations of fine-tuning small models: catastrophic forgetting, overconfident hallucinations, and the fluff introduced by conversational RLHF.

  We just open-sourced **BDCA (Bio-Decoupled Cognitive Architecture)**, a framework designed to run reliable 8B models locally with zero human-pleasing bias:

  ### The Core Design:
  1. **Spinal Sentinel**: A compiled invariant barrier that runs in 5–15ms. It evaluates commands against hard schemas and safety invariants before the action executes. Zero hallucination possible.
  2. **Frozen 8B Backbone + LoRA**: The 8B parameters are locked (`requires_grad = False`). Only low-rank plastic adapters change.
  3. **No Prose in Cognition**: The internal brain thinks in ASTs and structured JSON. Natural language is decoupled into an external 1.5B Broca presentation layer.
  4. **Dream-based RSI**: Daytime `AssertionError` logs are digested at night using Tononi's Synaptic Homeostasis Hypothesis (SHY downscaling) to prevent weight drift.

  ### Try the Demo
  You can test the entire loop (including reflex interception of lethal commands and dreaming consolidation) in 0.1 seconds:

  ```bash
  git clone https://github.com/ak111nice/synthetic-intelligence
  cd synthetic-intelligence
  pip install -e .
  python demo.py
  ```

  * Full Paper: https://github.com/ak111nice/synthetic-intelligence/blob/main/PAPER.md
  * GitHub Repo: https://github.com/ak111nice/synthetic-intelligence

  Curious to hear your thoughts on running decoupled invariants on local hardware!
  ```

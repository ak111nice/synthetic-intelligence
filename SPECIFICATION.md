# BDCA Architecture Specification (RFC-001)

**Status**: Standard Track (v1.0.0)  
**Authors**: ak111nice, Antigravity AI Research  
**Category**: Cognitive Systems / AI Systems Engineering  

---

## 1. Scope & Purpose

This specification defines the functional interfaces, data models, state-space representations, and operational requirements for the **Bio-Decoupled Cognitive Architecture (BDCA)**.

Implementations conforming to this RFC MUST satisfy all conformance criteria denoted by RFC 2119 keywords (`MUST`, `MUST NOT`, `REQUIRED`, `SHALL`, `SHALL NOT`, `SHOULD`).

---

## 2. Core Subsystem Definitions

### 2.1 The Spinal Sentinel (`bdca.spine`)
1. The Sentinel MUST evaluate every proposed action before execution.
2. The Sentinel MUST execute with sub-20ms latency.
3. The Sentinel MUST NOT generate natural language text.
4. If an invariant rule is violated, the Sentinel MUST terminate execution immediately and emit a typed `InvariantViolationException`.

### 2.2 The Cerebral Engine (`bdca.brain`)
1. The Cerebral Engine MUST use a frozen base model ($W_0$) whose gradient computation is permanently disabled (`requires_grad = False`).
2. All online adaptation MUST be restricted to low-rank auxiliary tensors ($B \times A$).
3. The Cerebral Engine MUST output valid JSON complying with the requested schema.
4. The Cerebral Engine MUST NOT generate conversational prose or user-directed apologetic text.

### 2.3 The Broca Interface (`bdca.mouth`)
1. The Broca Layer MUST NOT connect directly to the execution runtime or tool dispatchers.
2. The Broca Layer MUST operate strictly as a unidirectional presentation transformer (JSON $\to$ Markdown).

### 2.4 The Episodic Memory Ledger (`bdca.memory`)
1. Failed executions MUST be recorded in an immutable, append-only SQLite or DuckDB ledger.
2. Each record MUST contain: `timestamp`, `state_hash`, `action_payload`, `error_type`, and `traceback`.

---

## 3. Data Schemas

### 3.1 State Schema (`bdca.types.State`)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CerebralState",
  "type": "object",
  "properties": {
    "session_id": { "type": "string" },
    "epoch": { "type": "integer" },
    "system_telemetry": {
      "type": "object",
      "properties": {
        "memory_usage_mb": { "type": "number" },
        "active_connections": { "type": "integer" },
        "last_exit_code": { "type": "integer" }
      },
      "required": ["memory_usage_mb", "last_exit_code"]
    },
    "contract_invariants": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["session_id", "epoch", "system_telemetry", "contract_invariants"]
}
```

### 3.2 Action Proposal Schema (`bdca.types.Action`)
```json
{
  "title": "CerebralAction",
  "type": "object",
  "properties": {
    "action_id": { "type": "string" },
    "target_subsystem": { "type": "string" },
    "command": { "type": "string" },
    "parameters": { "type": "object" },
    "expected_invariant": { "type": "string" }
  },
  "required": ["action_id", "target_subsystem", "command", "parameters"]
}
```

---

## 4. The Monotonicity Invariant Suite

Implementations MUST maintain a regression test suite consisting of a minimum of 50 deterministic test vectors. 

During the Dreaming Phase:
$$\text{MonotonicityRatio} = \frac{\text{Passed Invariants}(W_{k+1})}{\text{Passed Invariants}(W_k)}$$

If $\text{MonotonicityRatio} < 1.0$, the deployment harness MUST execute an automated rollback.

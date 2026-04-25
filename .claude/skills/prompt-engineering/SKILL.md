---
name: prompt-engineering
description: "Prompt engineering patterns for writing agent prompts and skill files — constraint budgets, register shifting, HARD GATE patterns, anti-personas. Use when writing or reviewing agents/*.md or skills/*/SKILL.md. TRIGGER when: agent prompt, skill file, prompt engineering, model-tier compensation, HARD GATE, prompt quality. DO NOT TRIGGER when: user-facing docs, README, CHANGELOG, config files."
allowed-tools: [Read, Grep, Glob]
---

# Prompt Engineering Patterns

Compact, actionable patterns for writing agent prompts and skill files. See [docs/PROMPT-ENGINEERING.md](../../docs/PROMPT-ENGINEERING.md) for full rationale and academic citations.

---

## Constraint Budget (MOSAIC finding)

| Range | Compliance |
|-------|-----------|
| 1–6 | Reliable |
| 7–15 | Unpredictable |
| >15 | Degraded |

**Rule**: Max 6 hard constraints per section. Hard gates at the END of sections (Claude recency effect).

---

## Register Shifting Checklist

Before finalizing any hard gate or enforcement directive, verify the verbs:

- [ ] Tier 1 (deep): `audit`, `verify`, `critically analyse`, `enumerate` — use for HARD GATEs
- [ ] Tier 2 (balanced): `review`, `check`, `describe` — use for standard task instructions
- [ ] Tier 3 (minimal): `look at`, `mention`, `note` — advisory only; never in enforcement

**FORBIDDEN**: Tier 3 verbs in REQUIRED or FORBIDDEN lists.

---

## Persona Anti-Pattern

**FORBIDDEN for analytical agents** (reviewers, auditors, analysts, planners):
- ❌ `You are a [role]...` persona prefixes
- ❌ `Act as an expert in...` framing
- ❌ `Imagine you are...` setups

**Source**: PRISM (arxiv:2603.18507) — expert personas reduce knowledge-retrieval accuracy for analytical tasks.

**Correct pattern**: Role description + behavioral directive.

```
# BAD
You are a senior security engineer. Review this code.

# GOOD
Perform a security audit. Enumerate all CWE categories present.
```

---

## HARD GATE Pattern Template

```markdown
### HARD GATE: [Topic]

**[One-sentence scope statement]**

**REQUIRED** (all must be met):
- Specific action A
- Specific action B

**FORBIDDEN** — You MUST NOT do any of the following:
- ❌ You MUST NOT [prohibited action]
- ❌ You MUST NOT [prohibited action]

**Why**: [One sentence — increases compliance]
```

---

## Model-Tier Compensation Template

```xml
<model-tier-compensation tier="[haiku|sonnet|opus]">
## Model-Tier Behavioral Constraints ([Tier Name])

- Do NOT [specific tendency to suppress].
- ALWAYS [specific required behavior].
</model-tier-compensation>
```

Rules:
- Include `tier` attribute for automated validation
- No meta-commentary about model weaknesses (model-visible; may reinforce behavior)
- Actionable directives only — positive framing where possible
- Place at TOP of agent prompt before role description

---

## Self-Refine Loop (when quality matters)

Use for planning, security scans, and architecture tasks. 2–3 passes optimal (NeurIPS 2023).

```markdown
## Pass 1: GENERATE
[Initial output]

## Pass 2: FEEDBACK
Critically analyse Pass 1. Enumerate:
- Gaps (missing)
- Errors (wrong)
- Ambiguities (unclear)

## Pass 3: REFINE
Apply Pass 2 feedback. Produce final output.
```

**Anti-pattern**: "Review your answer and improve it" — Tier 3, will be ignored. Use Tier 1: "Critically analyse for gaps and enumerate missing cases."

---

## Stick+Carrot Pattern

Every block (hook or agent) MUST include a required next action:

```
# STICK only (bad)
BLOCKED: Cannot edit agents/*.md outside the pipeline.

# STICK + CARROT (good)
BLOCKED: Cannot edit agents/*.md outside the pipeline.
REQUIRED NEXT ACTION: Use /implement to run the pipeline with your changes.
```

Hook JSON format:
```json
{
  "decision": "block",
  "reason": "What failed and why.",
  "requiredNextAction": "Specific step to take next."
}
```

---

## Anti-Patterns Summary

| Anti-Pattern | Why Wrong | Fix |
|-------------|-----------|-----|
| `>15 constraints` per section | MOSAIC: degraded compliance | Split into subsections ≤6 each |
| Persona prefix for analytical tasks | PRISM: reduces accuracy | Role description + directive |
| Hedge words in hard gates | Weakens enforcement | Delete or replace with MUST |
| Prose-buried constraints | Low compliance | Convert to numbered list |
| Meta-commentary about model weaknesses | May reinforce behavior | Reframe as behavioral directive |
| Block without carrot | Model loops | Add REQUIRED NEXT ACTION |

---

## Cross-Reference

See [docs/PROMPT-ENGINEERING.md](../../docs/PROMPT-ENGINEERING.md) for:
- Full MOSAIC constraint budget analysis
- PRISM persona findings detail
- Self-Refine NeurIPS 2023 citations
- XML sectioning rationale
- Complete anti-pattern catalog

# <feature> — spec

Replace every `<angle bracket>`. Delete a section that is genuinely empty rather than
writing "N/A". Target: one screen.

## 1. Ask — verbatim

> <Paste the request unedited: typos, tone and all. Never reword it. Never tick it.>

This is a quotation, not a worklist. It is what review checks against — paraphrase here and
the review only proves the spec agrees with the code it produced.

## 2. Decision

<What gets built, in 2-4 sentences. Name the one design choice that was not obvious and say
why the alternative lost. Cite code as `path.py:123`, do not re-describe it.>

## 3. Requirements

Literal. Every threshold carries its number **and its direction**.

- [ ] **<label>:** <"upsample the prediction, never downsample the GT" — not "handle resolution">
- [ ] **<label>:** <...>

## 4. Done when

Each item measurable, and naming the metric it is measured by.

- [ ] **<claim>** — <metric> <number>, baseline <number>. Confound avoided: <what the obvious
      metric would have flattered, and by how much>.
      **Verify:** `tests/test_<x>.py` — <the assertion that fails if this breaks>

## 5. Out of scope (explicit)

- <what must NOT be built, changed or deleted. Name files or directories.>

<!-- What the reviewers check:
  - §1 unticked and unedited, always.
  - Every §4 item has a **Verify:** clause naming a real file and a real assertion.
    A clause that cannot fail is a finding.
  - Rank by classaware_f1 (score_mode=detector), never crop top-1.
  - Every number names its iteration; every timing names the machine and its load. -->

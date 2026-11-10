# branch 42 — frozen MaskDINO segmenter + separate crop classifier

A frozen **class-agnostic** MaskDINO supplies instance masks; `cropcls/` crops each mask and
classifies it. The live spec is [spec_cropcls.md](spec_cropcls.md).

**This branch is not the reclassification line of work.** The `CLAUDE.md` in the main checkout
(`/home/janick.bilang/dev/MaskDINO/CLAUDE.md`) describes phase-2 reclassification and says
*"Do not propose or reason about classifier-only retraining as the active setup."* That rule
does **not** apply here — on this branch a separate classifier **is** the work. Read that file
for MaskDINO internals, not for what is live.

## Environment — the trap that will cost you a run

This is a **git worktree**. Run everything from `/home/janick.bilang/dev/MaskDINO-Unet`.

- **Interpreter: `/home/janick.bilang/dev/MaskDINO/.venv/bin/python`** — in the *sibling*
  checkout. There is **no `.venv` in this worktree**, so the repo's usual
  `./.venv/bin/python` does not exist here.
- **`maskdino` is not installed in that venv.** `import maskdino` and `import cropcls` resolve
  only through the current directory. So tests run **only** as

  ```bash
  cd /home/janick.bilang/dev/MaskDINO-Unet
  /home/janick.bilang/dev/MaskDINO/.venv/bin/python -m pytest tests/ -q
  ```

  The `-m` form is what puts cwd on `sys.path`. Any other form is an **ImportError, not a test
  failure** — do not read it as a broken test.
- **Scripts in `tools/` need `PYTHONPATH=.`** for the same reason: a script's `sys.path[0]` is
  its own directory, so `python tools/anything.py` dies on `import maskdino`. This affects
  every tool here, not just new ones. Their docstrings predate the worktree and omit it.
- Other interpreters on this host are dead ends: `miniforge3/envs/myenv` has torch but no
  pytest; `/usr/bin/pytest` has no torch; putting `/usr/lib/python3/dist-packages` on
  `PYTHONPATH` shadows the venv's numpy and breaks the torch import.
- `testpaths = ["tests"]`, flat `tests/`, no `conftest.py`. Lint: `ruff`.
- **`export SURGICAL_IMAGES_ROOT=/home/janick.bilang/training/images_lambdablade-6000` before
  anything that imports `maskdino` on this host.** `images_root.py` globs `~/training/images_*`
  and **raises** on more than one match; `dbe-cia-sl19-01` has three
  (`images_babylambda1-2080`, `images_dgx-a100`, `images_lambdablade-6000`), and the variable is
  set neither in the environment nor in `~/.bashrc`. So a bare `import maskdino` dies with
  `RuntimeError: several image roots match ~/training/images_*` - at import, before any test or
  training code runs. The old `training/images` symlink is **gone**; this replaced it.
- The roots were once **different renders of the same scene sets**, and older notes say to
  tell them apart by the frame-name digest. That no longer holds for the set in use:
  measured 2026-11-03, `images_dgx-a100/train_phase1_pool_set_a_mm` and
  `images_lambdablade-6000/train_phase1_pool_set_a_mm` have the **same 1024 frame names and
  sizes and the same bytes** (3 files hashed in full), digest `0d90fc4eb1bf` for both.
  `images_babylambda1-2080` has no phase-1 pool at all. So the digest **cannot** say which
  root a run used - it only catches the pool being replenished, which does change frame names.
  A run's `config.json` records only the dataset *name*, so what identifies the data is
  `crops/<split>/meta.json`, which pins `pool_realpath` as well as the digest. Digests quoted
  in older docs (`d0cd49821f66`, `abe2ae25810c`) match nothing on disk now.
- **Datasets are registered by folder name**, so the sets are `train_phase1_pool_set_a_mm` and
  `val_set_a_mm`. The older names `train_seta_mm` / `val_seta_mm` appear all over the specs and
  results and are **not registered any more**; `configs/cropcls/phase1_seta.yaml` and
  `phase1_seta_type2.yaml` still name them and will raise
  `Dataset 'train_seta_mm' is not registered!`. The six `mask4_*` configs use the new names.
- Classifier entry point: `train_cropcls.py`. Runs land in `runs/<name>/`.
- **Expect this warning and do not act on it:** `MultiScaleDeformableAttention CUDA op is not
  compiled; using the pure-PyTorch fallback`. The fallback is **deliberate** — the main
  checkout's CLAUDE.md records that the compiled kernel corrupts the CUDA stream under this
  torch/CUDA build (surfacing as "invalid resource handle" on the next unrelated CUDA call,
  uncatchable by try/except). **Do not run `sh make.sh` to "fix" it.** It does mean any timing
  measured here is on the slow path.

## Read, do not re-derive

| file | what it settles |
|---|---|
| [spec_cropcls.md](spec_cropcls.md) | the requirements, the DoD, and the explicit out-of-scope list |
| [cropcls/README.md](cropcls/README.md) | the invariant → why → test table, and "Traps that cost real time" |
| `cropcls-baseline-comparison.md` | the 1176-vs-1114 matched-mask confound, worth +7.75 points of recall for free |
| `cropcls-results.md` §5 | "Methodology notes, including what I got wrong" — read before quoting any number |
| `maskdino/evaluation/hungarian_evaluator_metrics.md` | the exact meaning of every emitted metric key |
| `MASKDINO_ARCHITECTURE.md` | the code↔paper map |

## Facts that are easy to get wrong

- **Rank runs by `classaware_f1`** from the baseline's own `HungarianInstanceEvaluator` with
  `score_mode=detector`. **Crop top-1 is a training monitor only** — it scores only crops that
  passed `IoU >= 0.5 AND runner_up < 0.2`, i.e. clean isolated instances, and it has misled in
  *both* directions. It was used to rank runs and retracted each time.
- **Detector scores are sigmoid, not softmax**, and `score = objectness * class_prob` couples
  localization to classifier confidence — an undertrained classifier can make good masks vanish
  below `SCORE_THRESH`, depressing even the *class-agnostic* metric by 21 points. Use
  `score_mode="detector"` to read localization alone.
- **Always check the leaf config**, never conclude from a base config.
- **Name the machine and its load for every timing.** The same detector forward is 629 ms/frame
  on a contended Quadro RTX 6000 and 125 ms/frame on an idle RTX 2080 Ti.
- `/home` quota is tight and checkpoints are ~500 MB. Keep `checkpoint_period` coarse.
- DDP: collective-running hooks must be ordered identically on every rank.

## The review gate

Changes to `cropcls/` go through three passes, in this order. They are disjoint by design — do
not let one do another's job.

1. **Pre-flight**, before any code: `spec-conformance` (`mode: preflight`) on the verbatim ask
   plus the spec amendment. **No code until BLOCK findings are cleared.**
2. **`/code-review high`** on the diff — correctness and reuse.
3. **Post-flight**: `spec-conformance` (`mode: postflight`) **and** `claim-verifier`, invoked in
   one message so they run in parallel.

Both reviewers are **report-only** (no `Write`/`Edit`). Every finding ends in exactly one of:
fix the code; fix the spec, recorded *inline in the DoD item*; or escalate to the user. There is
no "silently accept".

The reviewers' source of truth is a **verbatim** `requirements_*.md` at the repo root — the
user's words pasted unedited, typos intact. Without one, `spec-conformance` is instructed to
BLOCK, because a spec written by the same session that wrote the code proves only
self-consistency.

## Worktree hygiene

**Never bare `git stash` / `git stash pop`** — the stash stack is shared with the main checkout
and every other worktree, and other sessions may be using it. Prefer a temporary WIP commit.

"""Server-specific root of the rendered HDF5 image sets.

Each training server keeps the same sets under its own ~/training/images_<server>
(images_dgx-a100, images_babylambda1-2080, images_lambdablade-6000, ...); the rest
of every dataset path is identical. Resolution order:

1. $SURGICAL_IMAGES_ROOT, if set (must be an existing directory).
2. The one ~/training/images_* directory on this machine.
3. Several of them -> error naming the candidates; never a silent pick.
   None -> None (the caller skips registration with a warning).

A server that needs the override sets it once at the TOP of ~/.bashrc - above the
`case $- in ... return` guard, so non-interactive ssh/nohup/tmux runs see it too:
    export SURGICAL_IMAGES_ROOT=/home/janick.bilang/training/images_<server>

Stdlib only, so shell scripts can run it by path without `import maskdino`:
    IMAGES="$(./.venv/bin/python maskdino/data/datasets/images_root.py)"
"""
import glob
import os
import sys

ENV_VAR = "SURGICAL_IMAGES_ROOT"
_PATTERN = "~/training/images_*"


def resolve_images_root():
    override = os.environ.get(ENV_VAR)
    if override:
        if not os.path.isdir(override):
            raise FileNotFoundError(f"{ENV_VAR}={override} is not a directory")
        return override
    # expanded per call, not at import, so tests can point HOME elsewhere
    candidates = sorted(
        p for p in glob.glob(os.path.expanduser(_PATTERN)) if os.path.isdir(p)
    )
    if len(candidates) > 1:
        raise RuntimeError(
            f"several image roots match {_PATTERN}: {candidates} - pick one with "
            f"`export {ENV_VAR}=<dir>` (e.g. at the top of ~/.bashrc)"
        )
    return candidates[0] if candidates else None


if __name__ == "__main__":
    root = resolve_images_root()
    if root is None:
        sys.exit(f"no {_PATTERN} directory found; set {ENV_VAR}")
    print(root)

#!/usr/bin/env python
"""One-shot: re-index a checkpoint's class head into a different label space.

Every MaskDINO instance-seg checkpoint has exactly four ``num_classes``-shaped
tensors::

    sem_seg_head.predictor.class_embed.weight   (N, hidden_dim)
    sem_seg_head.predictor.class_embed.bias     (N,)
    sem_seg_head.predictor.label_enc.weight     (N, hidden_dim)
    criterion.empty_weight                      (N + 1,)   # buffer, CE loss only

This copies each *source* row into the *target* row that carries the same
canonical instrument category_id (e.g. "adapter-11" == category_id 11), leaves
genuinely new target rows at a fresh ``nn.Linear`` / ``nn.Embedding`` init, and
stamps the target :class:`~maskdino.ClassMapping` into the output ``.pth``. Use
it to turn a converged phase-1 model into the starting point for a phase-2
classifier-retrain whose head has a different width / class set.

The source checkpoint's own label space is read from ``--src-mapping`` (a
``class_mapping.json`` / run dir / ``.pth``) or, by default, from the embedded
mapping in ``--src`` itself.

Example (adapt the converged set-B phase-1 run for the 16-class reclass phase)::

    ./.venv/bin/python tools/adapt_classifier_checkpoint.py \\
        --src runs/multi_class_setb_live_pool_real_size_4/model_final.pth
"""
import argparse
import os
import sys

import torch
from torch import nn

_CLASS_EMBED_W = "sem_seg_head.predictor.class_embed.weight"
_CLASS_EMBED_B = "sem_seg_head.predictor.class_embed.bias"
_LABEL_ENC_W = "sem_seg_head.predictor.label_enc.weight"
_EMPTY_WEIGHT = "criterion.empty_weight"


def adapt_class_head(sd, src_canonical_to_row, cm_target, *, seed=0, eos_coef=0.1):
    """In place: resize the four num_classes-shaped tensors of ``sd`` to
    ``cm_target``'s label space, copying rows by canonical id. Returns
    ``(carried_canonical_ids, fresh_canonical_ids)``.
    """
    for k in (_CLASS_EMBED_W, _CLASS_EMBED_B, _LABEL_ENC_W):
        if k not in sd:
            raise KeyError(f"state dict is missing '{k}'")

    n_tgt = cm_target.num_classes
    hidden = sd[_CLASS_EMBED_W].shape[1]
    n_src = sd[_CLASS_EMBED_W].shape[0]
    max_row = max(src_canonical_to_row.values(), default=-1)
    if max_row >= n_src:
        raise ValueError(
            f"source row {max_row} out of range for a {n_src}-wide head "
            f"(canonical->row map: {src_canonical_to_row})"
        )

    # Fresh templates with the model's own default init (class_embed is a plain
    # nn.Linear - no prior-prob bias; label_enc a plain nn.Embedding).
    torch.manual_seed(seed)
    new_w = nn.Linear(hidden, n_tgt).weight.detach().clone()
    new_b = nn.Linear(hidden, n_tgt).bias.detach().clone()
    new_e = nn.Embedding(n_tgt, hidden).weight.detach().clone()

    src_w, src_b, src_e = sd[_CLASS_EMBED_W], sd[_CLASS_EMBED_B], sd[_LABEL_ENC_W]
    carried, fresh = [], []
    for j in range(n_tgt):
        c = cm_target.canonical_id(j)
        row = src_canonical_to_row.get(c)
        if row is None:
            fresh.append(c)
            continue
        carried.append(c)
        new_w[j] = src_w[row]
        new_b[j] = src_b[row]
        new_e[j] = src_e[row]

    sd[_CLASS_EMBED_W] = new_w
    sd[_CLASS_EMBED_B] = new_b
    sd[_LABEL_ENC_W] = new_e
    ew = torch.ones(n_tgt + 1)
    ew[-1] = eos_coef
    sd[_EMPTY_WEIGHT] = ew
    return sorted(carried), sorted(fresh)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, help="source checkpoint (.pth)")
    ap.add_argument("--dst", help="output path (default: <src dir>/model_final_reclass<N>.pth)")
    ap.add_argument(
        "--target-mapping",
        default="reclass_train",
        help="class_mapping.json / run dir / .pth / dataset name for the TARGET "
        "label space (default: reclass_train)",
    )
    ap.add_argument(
        "--src-mapping",
        help="source label space (class_mapping.json / run dir / .pth / dataset "
        "name); default: read the mapping embedded in --src",
    )
    ap.add_argument(
        "--eos-coef",
        type=float,
        default=0.1,
        help="criterion.empty_weight[-1] (cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT)",
    )
    ap.add_argument("--seed", type=int, default=0, help="fresh-row init seed")
    args = ap.parse_args()

    sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
    import maskdino  # noqa: F401  - registers datasets so a name resolves
    from maskdino import load_class_mapping, write_class_mapping_sidecar

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    n_src = sd[_CLASS_EMBED_W].shape[0] if _CLASS_EMBED_W in sd else None

    src_map = load_class_mapping(args.src_mapping or args.src)
    src_c2r = dict(src_map.thing_dataset_id_to_contiguous_id)
    cm_t = load_class_mapping(args.target_mapping)

    try:
        carried, fresh = adapt_class_head(
            sd, src_c2r, cm_t, seed=args.seed, eos_coef=args.eos_coef
        )
    except (KeyError, ValueError) as exc:
        sys.exit(f"{args.src}: {exc}")

    dst = args.dst or os.path.join(
        os.path.dirname(os.path.abspath(args.src)),
        f"model_final_reclass{cm_t.num_classes}.pth",
    )
    # Drop trainer/iteration/optimizer/scheduler: phase-2 starts fresh at iter 0.
    torch.save({"model": sd, "class_mapping": cm_t.state_dict()}, dst)
    sidecar = write_class_mapping_sidecar(os.path.dirname(os.path.abspath(dst)), cm_t)

    print(f"src  {args.src}   class head {n_src} -> {cm_t.num_classes}")
    print(f"     class_embed.weight -> {tuple(sd[_CLASS_EMBED_W].shape)}")
    print(f"     empty_weight       -> {tuple(sd[_EMPTY_WEIGHT].shape)}  [-1]={args.eos_coef}")
    print(f"carried {len(carried)} rows by canonical id: {carried}")
    print(f"fresh   {len(fresh)} rows (new classes)   : {fresh}")
    print(f"wrote {dst}")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()

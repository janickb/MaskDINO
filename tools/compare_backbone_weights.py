#!/usr/bin/env python
"""Check whether a ResNet backbone actually stayed frozen between two checkpoints.

Compares the ``backbone.*`` tensors of two detectron2 ``DetectionCheckpointer``
``.pth`` files (default: the shared finetuning init checkpoint vs. a trained
``model_final.pth``), grouped by ResNet stage (``stem``/``res2``.../``res5``) —
the granularity ``MODEL.BACKBONE.FREEZE_AT`` actually operates at. Stages whose
weights are bit-identical are reported as frozen; stages that changed get a
quantified drift (max abs diff, relative L2 change, cosine similarity).

Optionally (``--cka``) also computes a linear-CKA functional-similarity score
per stage from a folder of sample images, by building two ResNet backbones,
loading each checkpoint's weights into one, running both on the same image
batch, and comparing globally-average-pooled activations.

Examples::

    python tools/compare_backbone_weights.py \\
        --finetuned runs/multi_class_seta_frozen_backbone/model_final.pth

    python tools/compare_backbone_weights.py \\
        --finetuned runs/multi_class_seta_unfrozen_backbone/model_final.pth \\
        --cka --image-dir /path/to/sample_images
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

_STAGES = ("stem", "res2", "res3", "res4", "res5")
_DEFAULT_ORIGINAL = (
    "models/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"
)
_DEFAULT_CONFIG = (
    "configs/coco/instance-segmentation/"
    "maskdino_R50_surgical_tools_finetune_multiclass_seta_frozen_backbone.yaml"
)


def load_state_dict(path):
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - report cleanly, don't crash
        sys.exit(f"{path}: could not be loaded as a torch checkpoint ({exc})")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(sd, dict) or not all(torch.is_tensor(v) for v in sd.values()):
        sys.exit(f"{path}: does not look like a detectron2 checkpoint state dict")
    return sd


def filter_keys(sd, prefix, all_keys):
    if all_keys:
        return dict(sd)
    return {k: v for k, v in sd.items() if k.startswith(prefix)}


def stage_of(key, prefix):
    if not key.startswith(prefix):
        return "other"
    rest = key[len(prefix):].split(".", 1)[0]
    return rest if rest in _STAGES else "other"


def compare_tensor(a, b):
    result = {
        "shape_a": tuple(a.shape),
        "shape_b": tuple(b.shape),
        "shape_mismatch": tuple(a.shape) != tuple(b.shape),
        "bit_identical": False,
        "identical": False,
        "max_abs_diff": float("nan"),
        "rel_l2": float("nan"),
        "cosine_sim": float("nan"),
    }
    if result["shape_mismatch"]:
        return result

    result["bit_identical"] = a.dtype == b.dtype and torch.equal(a, b)
    af, bf = a.float(), b.float()
    result["identical"] = torch.equal(af, bf)
    if result["identical"]:
        result["max_abs_diff"] = 0.0
        result["rel_l2"] = 0.0
        result["cosine_sim"] = 1.0
        return result

    diff = (af - bf).flatten()
    a_norm = af.flatten().norm().item()
    result["max_abs_diff"] = diff.abs().max().item()
    result["rel_l2"] = diff.norm().item() / a_norm if a_norm > 0 else diff.norm().item()
    result["cosine_sim"] = F.cosine_similarity(af.flatten(), bf.flatten(), dim=0).item()
    return result


def compare_backbones(sd_a, sd_b, prefix):
    common = sorted(set(sd_a) & set(sd_b))
    rows = []
    for key in common:
        row = {"key": key, "stage": stage_of(key, prefix)}
        row.update(compare_tensor(sd_a[key], sd_b[key]))
        rows.append(row)
    return {
        "rows": rows,
        "missing_in_b": sorted(set(sd_a) - set(sd_b)),
        "missing_in_a": sorted(set(sd_b) - set(sd_a)),
    }


def summarize_by_stage(rows):
    summary = {}
    for stage in sorted({r["stage"] for r in rows}, key=lambda s: (_STAGES + ("other",)).index(s)):
        stage_rows = [r for r in rows if r["stage"] == stage]
        changed = [r for r in stage_rows if not r["identical"]]
        summary[stage] = {
            "total": len(stage_rows),
            "n_identical": len(stage_rows) - len(changed),
            "n_changed": len(changed),
            "mean_rel_l2_changed": (
                sum(r["rel_l2"] for r in changed if not r["shape_mismatch"])
                / max(1, sum(1 for r in changed if not r["shape_mismatch"]))
                if any(not r["shape_mismatch"] for r in changed)
                else float("nan")
            ),
            "max_abs_diff": max((r["max_abs_diff"] for r in changed if not r["shape_mismatch"]), default=0.0),
            "worst_key": max(
                (r for r in changed if not r["shape_mismatch"]),
                key=lambda r: r["rel_l2"],
                default=None,
            ),
        }
        summary[stage]["worst_key"] = (
            summary[stage]["worst_key"]["key"] if summary[stage]["worst_key"] else None
        )
    return summary


def linear_cka(x, y):
    """Kornblith et al. 2019 linear CKA. x: (n, p1), y: (n, p2)."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (y.t() @ x).norm(p="fro") ** 2
    denom = (x.t() @ x).norm(p="fro") * (y.t() @ y).norm(p="fro")
    return (hsic / denom).item() if denom > 0 else float("nan")


def build_and_load_backbone(config_file, opts, backbone_sd, prefix, device):
    from detectron2.config import get_cfg
    from detectron2.modeling.backbone import build_backbone
    from detectron2.projects.deeplab import add_deeplab_config
    from maskdino import add_maskdino_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()

    backbone = build_backbone(cfg)
    stripped = {
        k[len(prefix):]: v for k, v in backbone_sd.items() if k.startswith(prefix)
    }
    missing, unexpected = backbone.load_state_dict(stripped, strict=False)
    if missing:
        print(f"  [warn] backbone missing keys not covered by checkpoint: {missing}")
    if unexpected:
        print(f"  [warn] checkpoint keys not used by backbone: {unexpected}")
    backbone.eval().to(device)
    return backbone, cfg


def load_and_preprocess_images(image_dir, num_images, resize_size, pixel_mean, pixel_std, input_format, seed):
    paths = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
    )
    if not paths:
        sys.exit(f"--image-dir {image_dir}: no .jpg/.jpeg/.png images found")
    if len(paths) < num_images:
        print(f"  [warn] only {len(paths)} images found in {image_dir}, requested {num_images}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:num_images]

    from PIL import Image

    mean = torch.tensor(pixel_mean).view(3, 1, 1)
    std = torch.tensor(pixel_std).view(3, 1, 1)
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((resize_size, resize_size), Image.BILINEAR)
        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        if input_format == "BGR":
            arr = arr.flip(0)
        tensors.append((arr - mean) / std)
    return torch.stack(tensors, dim=0)


def extract_pooled_features(backbone, images, device):
    with torch.no_grad():
        feats = backbone(images.to(device))
    return {stage: t.mean(dim=[2, 3]).cpu() for stage, t in feats.items()}


def run_cka(config_file, opts, sd_a, sd_b, prefix, image_dir, num_images, resize_size, device, seed):
    backbone_a, cfg = build_and_load_backbone(config_file, opts, sd_a, prefix, device)
    backbone_b, _ = build_and_load_backbone(config_file, opts, sd_b, prefix, device)

    images = load_and_preprocess_images(
        image_dir, num_images, resize_size,
        cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD, cfg.INPUT.FORMAT, seed,
    )
    feats_a = extract_pooled_features(backbone_a, images, device)
    feats_b = extract_pooled_features(backbone_b, images, device)

    scores = {}
    for stage in _STAGES[1:]:  # ResNet.forward only returns res2..res5
        if stage in feats_a and stage in feats_b:
            scores[stage] = linear_cka(feats_a[stage], feats_b[stage])
    return scores


def print_report(original, finetuned, compare_result, stage_summary, cka_scores, *, fmt, output, verbose):
    if fmt == "json":
        payload = {
            "original": original,
            "finetuned": finetuned,
            "rows": compare_result["rows"],
            "missing_in_a": compare_result["missing_in_a"],
            "missing_in_b": compare_result["missing_in_b"],
            "stage_summary": stage_summary,
            "cka": cka_scores,
        }
        text = json.dumps(payload, indent=2, default=str)
        if output:
            with open(output, "w") as f:
                f.write(text)
            print(f"wrote {output}")
        else:
            print(text)
        return

    lines = []
    lines.append(f"original : {original}")
    lines.append(f"finetuned: {finetuned}")
    n_compared = len(compare_result["rows"])
    lines.append(f"backbone tensors compared: {n_compared}")
    if compare_result["missing_in_a"]:
        lines.append(f"keys only in finetuned (missing in original): {compare_result['missing_in_a']}")
    if compare_result["missing_in_b"]:
        lines.append(f"keys only in original (missing in finetuned): {compare_result['missing_in_b']}")
    lines.append("")
    lines.append(f"{'stage':6} {'total':>6} {'identical':>10} {'changed':>8} {'mean_rel_l2':>12} {'max_abs_diff':>13}  worst_key")
    for stage, s in stage_summary.items():
        mean_l2 = "-" if s["n_changed"] == 0 else f"{s['mean_rel_l2_changed']:.4g}"
        max_diff = "-" if s["n_changed"] == 0 else f"{s['max_abs_diff']:.4g}"
        worst = s["worst_key"] or "-"
        lines.append(f"{stage:6} {s['total']:>6} {s['n_identical']:>10} {s['n_changed']:>8} {mean_l2:>12} {max_diff:>13}  {worst}")
    lines.append("")
    for stage, s in stage_summary.items():
        if s["n_changed"] == 0:
            lines.append(f"  {stage:5}: IDENTICAL (frozen)")
        else:
            lines.append(
                f"  {stage:5}: DRIFTED: {s['n_changed']}/{s['total']} tensors changed, "
                f"mean rel L2={s['mean_rel_l2_changed']:.4g}"
            )

    if verbose:
        lines.append("")
        lines.append("per-tensor detail:")
        for r in compare_result["rows"]:
            if r["shape_mismatch"]:
                lines.append(f"  {r['key']}: SHAPE MISMATCH {r['shape_a']} vs {r['shape_b']}")
            elif r["identical"]:
                lines.append(f"  {r['key']}: identical")
            else:
                lines.append(
                    f"  {r['key']}: rel_l2={r['rel_l2']:.4g} max_abs_diff={r['max_abs_diff']:.4g} "
                    f"cosine_sim={r['cosine_sim']:.4g}"
                )

    total_changed = sum(s["n_changed"] for s in stage_summary.values())
    if total_changed == 0:
        lines.append("")
        lines.append(f"verdict: backbone fully frozen ({n_compared}/{n_compared} tensors bit-identical).")
    else:
        lines.append("")
        lines.append(f"verdict: backbone changed ({total_changed}/{n_compared} tensors drifted).")

    if cka_scores:
        lines.append("")
        lines.append("CKA (linear, functional similarity per stage):")
        for stage, score in cka_scores.items():
            lines.append(f"  {stage}: {score:.4f}")

    text = "\n".join(lines)
    print(text)
    if output:
        with open(output, "w") as f:
            f.write(text + "\n")
        print(f"wrote {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default=_DEFAULT_ORIGINAL, help="'before' checkpoint (.pth)")
    ap.add_argument("--finetuned", required=True, help="'after' checkpoint (.pth)")
    ap.add_argument("--key-prefix", default="backbone.", help="key prefix to compare (default: backbone.)")
    ap.add_argument("--all-keys", action="store_true", help="compare the full state dict, not just --key-prefix")
    ap.add_argument("--format", choices=["table", "json"], default="table")
    ap.add_argument("--output", default=None, help="also write the report to this path")
    ap.add_argument("--verbose", action="store_true", help="print one row per tensor, not just the stage summary")

    ap.add_argument("--cka", action="store_true", help="also compute a linear-CKA score per stage")
    ap.add_argument("--image-dir", default=None, help="folder of sample images for --cka")
    ap.add_argument("--num-images", type=int, default=32)
    ap.add_argument("--resize-size", type=int, default=512, help="square resize side (aspect ratio not preserved)")
    ap.add_argument("--config-file", default=_DEFAULT_CONFIG, help="config used only for backbone architecture / pixel norm")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="extra cfg overrides, forwarded to merge_from_list")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cka and not args.image_dir:
        ap.error("--cka requires --image-dir")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but torch.cuda.is_available() is False")

    sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
    import maskdino  # noqa: F401

    sd_a = load_state_dict(args.original)
    sd_b = load_state_dict(args.finetuned)

    keys_a = filter_keys(sd_a, args.key_prefix, args.all_keys)
    keys_b = filter_keys(sd_b, args.key_prefix, args.all_keys)
    result = compare_backbones(keys_a, keys_b, args.key_prefix)
    stage_summary = summarize_by_stage(result["rows"])

    cka_scores = None
    if args.cka:
        try:
            cka_scores = run_cka(
                args.config_file, args.opts, sd_a, sd_b, args.key_prefix,
                args.image_dir, args.num_images, args.resize_size, args.device, args.seed,
            )
        except RuntimeError as exc:
            sys.exit(f"CKA failed - checkpoints likely aren't backbone-compatible: {exc}")

    print_report(
        args.original, args.finetuned, result, stage_summary, cka_scores,
        fmt=args.format, output=args.output, verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

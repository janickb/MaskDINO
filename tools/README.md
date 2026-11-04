We use ResNet and Swin as the backbone in our model.

* `convert-torchvision-to-d2.py`

Tool to convert torchvision pre-trained weights for D2.

```
wget https://download.pytorch.org/models/resnet101-63fe2227.pth
python tools/convert-torchvision-to-d2.py resnet101-63fe2227.pth R-101.pkl
```

* `convert-pretrained-swin-model-to-d2.py`

Tool to convert Swin Transformer pre-trained weights for D2.

```
pip install timm

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_small_patch4_window7_224.pth swin_small_patch4_window7_224.pkl

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_base_patch4_window12_384_22k.pth swin_base_patch4_window12_384_22k.pkl

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_large_patch4_window12_384_22k.pth swin_large_patch4_window12_384_22k.pkl
```


* `analyze_model.py`

Tool to analyze model parameters and flops.

Usage for semantic segmentation (ADE20K only, use with caution!):

```
python tools/analyze_model.py --num-inputs 1 --tasks flop --use-fixed-input-size --config-file CONFIG_FILE
```

Note that, for semantic segmentation (ADE20K only), we use a dummy image with fixed size that equals to `cfg.INPUT.CROP.SIZE[0] x cfg.INPUT.CROP.SIZE[0]`.
Please do not use `--use-fixed-input-size` for calculating FLOPs on other datasets like Cityscapes!

Usage for panoptic and instance segmentation:

```
python tools/analyze_model.py --num-inputs 100 --tasks flop --config-file CONFIG_FILE
```

Note that, for panoptic and instance segmentation, we compute the average flops over 100 real validation images.


* `compare_backbone_weights.py`

Tool to check whether a ResNet backbone actually stayed frozen (or how much it
drifted) between two checkpoints, grouped by ResNet stage (stem/res2/res3/res4/res5).
Optionally computes a linear-CKA functional-similarity score per stage from a
folder of sample images.

```
python tools/compare_backbone_weights.py --finetuned runs/multi_class_seta_frozen_backbone/model_final.pth

python tools/compare_backbone_weights.py --finetuned runs/multi_class_seta_unfrozen_backbone/model_final.pth --cka --image-dir /path/to/sample_images
```


* `score_predictions.py`, `export_predictions_jsonl.py`

One scorer for every model in a comparison, whatever framework produced the
predictions (MaskDINO, Mask2Former, Ultralytics YOLO, ...). `score_predictions.py`
reads a JSON Lines file with one line per frame (HDF5 basename, class names, scores,
full-resolution COCO RLE masks; the exact format is in its docstring) and runs
`HungarianInstanceEvaluator` on it. The GT is prepared as `train_net.setup()` prepares
`DATASETS.TEST`: frame-edge-truncated tools become ignore regions, and the
`MIN_VISIBILITY` filter applies. The score threshold is swept over 0.10 to 0.90, both
class-agnostic and class-aware. The output dir gets `metrics.json` (the numbers to
report), `score_settings.json` and the `instance_matching_*` artifacts. The scorer
fails, listing every problem, if a frame is missing or listed twice, or if a class
name, score or mask size is wrong.

`export_predictions_jsonl.py` writes that format from a detectron2 eval's
`inference/instances_predictions.pth`, so MaskDINO and Mask2Former runs are scored
through the same path as the others.

```
./.venv/bin/python tools/export_predictions_jsonl.py --dataset val_set_a_mm \
    --predictions runs/<run>/inference/instances_predictions.pth \
    --output runs/<run>/predictions/val_set_a_mm.jsonl
./.venv/bin/python tools/score_predictions.py --dataset val_set_a_mm \
    --predictions runs/<run>/predictions/val_set_a_mm.jsonl \
    --output-dir runs/<run>/score/val_set_a_mm
```

Pass the `*_singleclass` dataset name for a single-class model. Set `--min-visibility`
to the compared recipe's `INPUT.MIN_VISIBILITY` (the default 0.0 is phase 1's). A
strided in-training eval (`TEST_SAMPLE_STRIDE: 4`) needs `--sample-stride 4`. Other
models' runs call the scorer from this repo's venv as a command, never as an import.

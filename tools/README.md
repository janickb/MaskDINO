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


* `export_attention.py`

Tool to export the decoder's cross-attention per detection - "where did the model
look when it detected this instrument?" - as `<frame>.attn.hdf5` sidecars that
hdf5_viewer's Attention tab reads (see `maskdino/visualization/attention.py`).
MaskDINO's cross-attention is deformable, so what is stored is the sparse set of
weighted sample points per head/level/layer plus each layer's reference box, not
a dense attention map; the heatmap is rasterized at display time.

`--preview` additionally writes a PNG per frame (image, then one heatmap panel
per detection with its mask outlined) so an export can be checked without the
viewer.

```
python tools/export_attention.py \
    --config-file configs/coco/instance-segmentation/maskdino_R50_surgical_tools_finetune_multiclass_seta_frozen_backbone.yaml \
    --input "$HOME/training/images_lambdablade-6000/val_set_a_mm/*.hdf5" \
    --output /tmp/attn --preview \
    --opts MODEL.WEIGHTS runs/20260915_multi_class_seta_frozen/model_final.pth

# everything the model saw, including low-confidence queries, plus the dense
# per-query response maps (bulky - fp16 at 1/4 resolution, one per detection)
python tools/export_attention.py ... --confidence-threshold 0.0 --max-detections 20 --dense
```

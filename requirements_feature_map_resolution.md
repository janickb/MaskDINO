# Feature-map resolution vs input resolution for the crop classifier — raw ask

Asked 2026-10-30, in conversation. Pasted unedited: typos, tone and all.
This file is the ground truth the reviewers check against. Do not reword it.

---

Origin of the thread (same conversation, earlier):

> can i use a larger resnet like resnet 448 instaed of 224

> could we use ArcFace, CosFace, or Triplet Loss to push forcep03 / 04 furhter apart?

The actual ask:

> we saw that higher resolution in the imagenet helps the f1 score. which is strange since we
> only upsample the low resolution image and segmentation map. Why does this happen, could we
> argue that the feature maps are of higher resolution and this helps the network, would a
> resnet50 with kernel 3*3x3 instead of 1*7x7 help on smal resolution (224)?

> how about this one?
>
> resnet50_cifar

> alright then lets test it

---

## Decision (2026-10-31, from the user)

The verbatim section above is unchanged and remains the ground truth. This section records
only what was decided, so a later reader does not think the unbuilt parts were forgotten.

- `resnet50_cifar` **does not exist** in timm 1.0.28 (no model with "cifar" in its name), so
  "lets test it" could not be executed literally.
- Asked which mechanism to test instead, the user chose **`resnet50d` only**: the real timm
  model with the 3x 3x3 deep stem and ImageNet weights. No code change, so the `cropcls/`
  review gate does not apply to this work.
- **Not built, deliberately:** the `output_stride` plumbing (dilated stages 3-4) that would
  have tested the feature-map-resolution hypothesis directly. The "why does upsampling help"
  question is therefore answered by analysis + measurement, not by an ablation.
- The user chose **one seed per arm**, with the noise floor flagged rather than replicated:
  seed variance on `classaware_f1` is 0.82 points (`cropcls-results.md` section 5), so any
  gap below ~1.6 points is not rankable.
- Pretrained tag is **pinned to `.a1_in1k` on both trunks**. Bare `resnet50d` defaults to
  `ra2_in1k` while `resnet50` defaults to `a1_in1k`, which would have confounded the stem
  change with a different pretraining recipe.

**Out of scope (explicit):** `output_stride` / dilation, any stem that cannot load the
ImageNet weights, seed replicates, changes to `cropcls/`, and the ArcFace/CosFace/Triplet
question quoted in the origin section (already answered in conversation: the training set is
separated at CE 1e-4, so margin losses have no headroom).

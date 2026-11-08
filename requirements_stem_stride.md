# Stem stride for the crop classifier — raw ask

Asked 2026-11-05, in conversation. Pasted unedited: typos, tone and all.
This file is the ground truth the reviewers check against. Do not reword it.

---

Origin of the thread (same conversation, earlier). The user had been told that
`output_stride=16` would test the "why does 224 -> 448 help" question, and rejected it:

> would you recommend to increase the pixel size further, since it laready helped once when
> going from 224 to 448

> what do you mean with output_stride=16

> but img reduction still happens the same in the first few layers compared when doing
> outputstride 16, so the fine detail in early layer is also still lost

The objection is correct and was verified: `output_stride=16` changes only `layer4`
(7 -> 14 at c224); `conv1`, `maxpool`, `layer1`, `layer2` and `layer3` are unchanged, while
c448 doubles every one of them. So `output_stride` cannot test the early-detail mechanism.

The actual ask:

> alright lets try the stem idea

> only test it with convnext-tiny

---

The same question, from the user, in the earlier round that
`requirements_feature_map_resolution.md` records verbatim:

> would a resnet50 with kernel 3*3x3 instead of 1*7x7 help on smal resolution (224)?

That round could not execute it (`resnet50_cifar` does not exist in timm 1.0.28) and listed
the mechanism as out of scope. This round builds it, on convnext_tiny only.

# Segmentation mask as a 4th input channel — raw ask

Asked 2026-10-31, in conversation. Pasted unedited: typos, tone and all.
This file is the ground truth the reviewers check against. Do not reword it.

---

> i have different network i usage right now
> resnet50
> resnet50d
> and convnext_tiny
>
> i want to create a new experiment series where i add a fourth layer (the segmentaiton mask) to
> the input of the network such that the first layer of each network has an additional channel.
> Which is initialized with zeros.

> coding wise i was thinking about something like this:
>
> ```python
> # Load pretrained model
> weights = ConvNeXt_Tiny_Weights.DEFAULT
> model = convnext_tiny(weights=weights)
>
> # Replace the first convolution
> old_conv = model.features[0][0]
>
> new_conv = nn.Conv2d(
>     in_channels=4,
>     out_channels=old_conv.out_channels,
>     kernel_size=old_conv.kernel_size,
>     stride=old_conv.stride,
>     padding=old_conv.padding,
>     bias=(old_conv.bias is not None)
> )
>
> # Copy pretrained RGB weights
> with torch.no_grad():
>     new_conv.weight[:, :3, :, :] = old_conv.weight
>     new_conv.weight[:, 3:4, :, :] = 0
>
>     if old_conv.bias is not None:
>         new_conv.bias.copy_(old_conv.bias)
>
> # Replace convolution
> model.features[0][0] = new_conv
> ```

---

## Decisions (2026-11-02, from the user)

The verbatim section above is unchanged and remains the ground truth. This section records
only what was decided when asked, so a later reader does not think the unbuilt parts were
forgotten.

Asked because the sketch is **torchvision** while `cropcls/` builds every trunk through
**timm**, and because `blackout: true` — the setting all five current arms use — already zeroes
RGB outside the mask, which makes a 4th channel nearly redundant with it.

- **`blackout: false` only.** The series runs with the real background kept, so the mask channel
  is the only thing identifying the instance pixels. This is the arm where the channel carries
  information; `spec_cropcls.md` notes `blackout: false` "acts as a distractor", and an explicit
  mask channel is precisely the fix for that. A `blackout: true` 4-channel arm was offered and
  declined.
- **Mask channel is `0.0 / 1.0`, appended after normalization.** `pixel_mean`/`pixel_std` stay
  3-element RGB, so the existing configs and the tests that pin those stats are untouched.
  Unit scale keeps the channel's gradient in the same range as normalized RGB (~±2), so the
  zero-initialized weights learn at a comparable rate. `-1/+1` and `0/255` with 4-element stats
  were offered and declined.
- **Six configs: 3 trunks x {224, 448}.** `resnet50`, `resnet50d`, `convnext_tiny` at both crop
  sizes, so the channel's effect can be read at both resolutions.
- **A 3-channel checkpoint is NOT loadable into a 4-channel run.** Out of scope deliberately:
  strict `load_state_dict` already raises a first-conv size mismatch, which is loud and
  unambiguous. Every arm starts from ImageNet. Adapting a checkpoint on load (copy RGB, zero
  the mask channel) was offered and declined.

**Out of scope (explicit):** checkpoint adaptation across the flag; a `blackout: true`
4-channel arm; feeding an *undilated* mask in channel 3 while the blackout uses the dilated one
(that needs `letterbox_crop` to return a second mask); any change to `maskdino/`.

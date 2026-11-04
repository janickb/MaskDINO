# Crop padding — verbatim ask

The user's words, pasted unedited, typos intact. This is a quotation, not a worklist.
Never reworded, never ticked. Asked 2026-11-04.

> in extract_crop.py how are the crops generated? i dont want them thight but they should ad a
> border of 10 percent on each side for the classifier to redetect lost pixels

> what is margin then doing, is this fucntion then not obsolete?

> why are we then not using margin to store the crops with a border?

> okey folder size does not seem to be that important it should raise by 1.2x1.2 so *1.44.
> lets adapt the spec_stati_crops.md. eithe r we keep margin and then calculate the 10 % per
> side and set margin new or we discard margin and just go with % size

On the knob, asked as a choice between keeping `margin` as an absolute floor alongside a new
percentage knob or replacing it outright, the user chose **replace it with `pad_frac`**.

On which mask regime the border is for, offered as `blackout: true` + dilation, or
`blackout: false` + `mask_channel`, or both as two arms:

> just extend the whole image by the 10%, we will save the whole image in training it can be
> decided what is kept and what not

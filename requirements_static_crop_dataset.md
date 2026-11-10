# Static crop dataset for the classifier — raw ask

Asked 2026-10-26/27, in conversation. Pasted unedited: typos, tone and all.
This file is the ground truth the reviewers check against. Do not reword it.

---

> but the whole architecture seems a bit crooked, how about we generate crop images on the fly
> and train a classifer from it like we did with the segmenter?

> lets first test the version where we use the current train images to produce train images for
> the classifer and then feed classifier training from this dataset such that we can just apply
> augmentation rotation and batch normalization like in a static training

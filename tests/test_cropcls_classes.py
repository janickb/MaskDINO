"""The class count comes from the dataset, so its two failure modes must name a cause."""

import pytest
from detectron2.data import MetadataCatalog

from cropcls.data import dataset_class_names


def test_registered_dataset_yields_its_thing_classes():
    MetadataCatalog.get("cropcls_test_eight").thing_classes = [f"c{i}" for i in range(8)]
    assert len(dataset_class_names("cropcls_test_eight")) == 8


def test_unregistered_dataset_points_at_the_images_symlink():
    """Empty metadata means registration skipped, which on a new host is the symlink."""
    with pytest.raises(ValueError, match="images. symlink"):
        dataset_class_names("cropcls_test_unregistered")


def test_singleclass_registration_is_rejected():
    """The *_singleclass sets exist for the class-agnostic detector, not the classifier."""
    MetadataCatalog.get("cropcls_test_one").thing_classes = ["instrument"]
    with pytest.raises(ValueError, match="singleclass"):
        dataset_class_names("cropcls_test_one")

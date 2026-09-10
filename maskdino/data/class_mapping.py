# Copyright (c) Facebook, Inc. and its affiliates.
"""Compact ("effective") class label space for the surgical HDF5 datasets.

Every scene_generator ``.hdf5`` file embeds an ``instrument_classes`` list whose
*list index is the canonical category_id*: index 0 is always ``"background"``,
and any category_id with no live object at render time gets a ``"unused_<i>"``
placeholder (see ``sgdata.coco.instrument_classes_from_config``). The stock
MaskDINO surgical loader fed that raw category_id straight in as the model class
index, which forced ``MODEL.SEM_SEG_HEAD.NUM_CLASSES`` up to
``max(category_id) + 1`` with dead rows for ``background`` / every ``unused_*``.

``derive_class_mapping`` drops ``background`` + every ``unused_*`` slot and packs
what is left into a contiguous ``0..N-1`` label space, ordered by canonical
category_id. ``ClassMapping`` carries that space plus enough metadata to invert
it - the canonical category_id and human name behind every contiguous slot - and
serialises through both ``json`` and ``torch.save``, so it can ride inside every
checkpoint (see ``train_net.py``) and a ``class_mapping.json`` sidecar.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CLASS_MAPPING_FILENAME = "class_mapping.json"
_STATE_FORMAT_VERSION = 1
_BACKGROUND_NAME = "background"
_UNUSED_PREFIX = "unused_"


class ClassMapping:
    """Bidirectional map between a model's contiguous class ids (``0..N-1``) and a
    dataset's canonical instrument category_ids / names.

    Build it with :func:`derive_class_mapping` (from an ``instrument_classes``
    list), :meth:`from_metadata` (from a registered dataset) or
    :meth:`from_state_dict` (from a checkpoint / sidecar).
    """

    def __init__(self, contiguous_to_canonical: dict[int, dict[str, Any]]):
        self.contiguous_to_canonical: dict[int, dict[str, Any]] = {
            int(i): {"category_id": int(v["category_id"]), "name": str(v["name"])}
            for i, v in contiguous_to_canonical.items()
        }
        n = len(self.contiguous_to_canonical)
        if set(self.contiguous_to_canonical) != set(range(n)):
            raise ValueError(
                f"contiguous ids must be exactly 0..{n - 1}, got "
                f"{sorted(self.contiguous_to_canonical)}"
            )
        self.num_classes: int = n
        self.thing_classes: list[str] = [
            self.contiguous_to_canonical[i]["name"] for i in range(n)
        ]
        self.thing_dataset_id_to_contiguous_id: dict[int, int] = {
            self.contiguous_to_canonical[i]["category_id"]: i for i in range(n)
        }
        if len(self.thing_dataset_id_to_contiguous_id) != n:
            raise ValueError(
                f"duplicate canonical category_id in {self.contiguous_to_canonical}"
            )

    # -- lookups -----------------------------------------------------------------
    def canonical_id(self, contiguous_id: int) -> int:
        return self._entry(contiguous_id)["category_id"]

    def canonical_name(self, contiguous_id: int) -> str:
        return self._entry(contiguous_id)["name"]

    def contiguous_id(self, canonical_id: int) -> int:
        try:
            return self.thing_dataset_id_to_contiguous_id[int(canonical_id)]
        except KeyError:
            raise KeyError(
                f"canonical category_id {canonical_id} is not in this mapping; "
                f"known {sorted(self.thing_dataset_id_to_contiguous_id)}"
            ) from None

    def _entry(self, contiguous_id: int) -> dict[str, Any]:
        try:
            return self.contiguous_to_canonical[int(contiguous_id)]
        except KeyError:
            raise KeyError(
                f"contiguous id {contiguous_id} out of range; "
                f"known 0..{self.num_classes - 1}"
            ) from None

    # -- (de)serialisation -----------------------------------------------------
    def state_dict(self) -> dict:
        """JSON- and pickle-safe. String keys so it round-trips through both
        ``json.dump`` and ``torch.save``."""
        return {
            "format_version": _STATE_FORMAT_VERSION,
            "entries": {
                str(i): dict(self.contiguous_to_canonical[i])
                for i in range(self.num_classes)
            },
        }

    def to_json_dict(self) -> dict:
        return self.state_dict()

    def load_state_dict(self, state: dict) -> None:
        """In-place - the Detectron2 checkpointable protocol."""
        self.__dict__.update(ClassMapping.from_state_dict(state).__dict__)

    @classmethod
    def from_state_dict(cls, state: dict) -> "ClassMapping":
        if not isinstance(state, dict) or "entries" not in state:
            raise ValueError(f"not a ClassMapping state dict: {type(state)!r}")
        return cls({int(k): v for k, v in state["entries"].items()})

    @classmethod
    def from_json_dict(cls, d: dict) -> "ClassMapping":
        return cls.from_state_dict(d)

    @classmethod
    def from_instrument_classes(cls, instrument_classes: list[str]) -> "ClassMapping":
        return derive_class_mapping(instrument_classes)

    @classmethod
    def from_metadata(cls, metadata) -> "ClassMapping":
        """Rebuild from a registered dataset's ``MetadataCatalog`` entry. Uses
        ``thing_dataset_id_to_contiguous_id`` when present, otherwise assumes the
        legacy identity space (contiguous id == canonical category_id)."""
        thing_classes = list(getattr(metadata, "thing_classes", []) or [])
        if not thing_classes:
            raise ValueError(
                f"metadata {getattr(metadata, 'name', metadata)!r} has no thing_classes"
            )
        d2c = dict(getattr(metadata, "thing_dataset_id_to_contiguous_id", {}) or {})
        if d2c:
            contig_to_canonical_id = {int(c): int(k) for k, c in d2c.items()}
        else:
            contig_to_canonical_id = {i: i for i in range(len(thing_classes))}
        return cls(
            {
                i: {"category_id": contig_to_canonical_id[i], "name": thing_classes[i]}
                for i in range(len(thing_classes))
            }
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ClassMapping)
            and other.contiguous_to_canonical == self.contiguous_to_canonical
        )

    def __repr__(self) -> str:
        return (
            f"ClassMapping(num_classes={self.num_classes}, "
            f"thing_classes={self.thing_classes})"
        )


def derive_class_mapping(instrument_classes: list[str]) -> ClassMapping:
    """``instrument_classes`` (list index == canonical category_id, index 0 ==
    ``"background"``, gaps == ``"unused_<i>"``) -> a compact :class:`ClassMapping`
    over just the real instrument classes, ordered by canonical category_id.
    """
    if not instrument_classes:
        raise ValueError("instrument_classes is empty")
    if instrument_classes[0] != _BACKGROUND_NAME:
        logger.warning(
            "instrument_classes[0] is %r, expected %r; dropping index 0 regardless",
            instrument_classes[0],
            _BACKGROUND_NAME,
        )
    # instrument_classes[i] is the name for canonical category_id i: index 0 is
    # "background", a gap in the id space is "unused_<i>", the rest are real.
    effective: list[tuple[int, str]] = []
    for category_id, name in enumerate(instrument_classes):
        if category_id == 0:  # background
            continue
        if name.startswith(_UNUSED_PREFIX):  # placeholder for an unused id slot
            continue
        effective.append((category_id, name))
    if not effective:
        raise ValueError(
            f"no effective classes in {instrument_classes!r} (all background/unused_*)"
        )

    # already ascending (enumerate order); pack into contiguous ids 0..N-1
    return ClassMapping(
        {
            contiguous_id: {"category_id": category_id, "name": name}
            for contiguous_id, (category_id, name) in enumerate(effective)
        }
    )


def remap_gt_category_ids(annotations: list[dict], cm: ClassMapping) -> None:
    """In place: rewrite each annotation's raw canonical ``category_id`` to its
    contiguous id. Raises ``ValueError`` on an id with no effective-class slot -
    a genuine dataset/config inconsistency, fail fast rather than silently drop
    a training target (background 0 is already filtered upstream, ``unused_*``
    slots never spawn instances)."""
    d2c = cm.thing_dataset_id_to_contiguous_id
    for ann in annotations:
        raw = ann["category_id"]
        try:
            ann["category_id"] = d2c[int(raw)]
        except KeyError:
            raise ValueError(
                f"annotation category_id={raw} has no effective-class slot "
                f"(known canonical ids {sorted(d2c)}); instrument_classes / render "
                f"config drift?"
            ) from None


def apply_class_mapping_to_metadata(name: str, cm: ClassMapping):
    """Set the compact class space (+ a ``class_mapping_entries`` marker) on a
    registered dataset's metadata. Superset of what the surgical registrars set
    today, so re-registration with an identical mapping is a no-op."""
    from detectron2.data import MetadataCatalog

    md = MetadataCatalog.get(name)
    md.set(
        thing_classes=list(cm.thing_classes),
        thing_dataset_id_to_contiguous_id=dict(cm.thing_dataset_id_to_contiguous_id),
        evaluator_type="coco",
        class_mapping_entries=cm.state_dict(),
    )
    return md


def write_class_mapping_sidecar(output_dir: str, cm: ClassMapping) -> str:
    """Write ``<output_dir>/class_mapping.json``. Idempotent."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, CLASS_MAPPING_FILENAME)
    with open(path, "w") as fh:
        json.dump(cm.to_json_dict(), fh, indent=2, sort_keys=True)
    return path


def load_class_mapping(source: str) -> ClassMapping:
    """Recover a :class:`ClassMapping`, trying in order: an explicit ``.pth``'s
    embedded ``class_mapping`` key; a ``class_mapping.json`` sidecar (``source``
    may be that file, a directory holding it, or a checkpoint's directory); the
    newest ``model_*.pth`` under a run dir; a registered dataset's metadata.
    """
    tried: list[str] = []

    def _from_ckpt(path: str) -> ClassMapping | None:
        import torch

        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - fall through to the next tier
            logger.warning("load_class_mapping: could not read %s: %s", path, exc)
            return None
        if isinstance(obj, dict) and "class_mapping" in obj:
            return ClassMapping.from_state_dict(obj["class_mapping"])
        return None

    if isinstance(source, str) and source.endswith(".pth") and os.path.isfile(source):
        tried.append(source)
        cm = _from_ckpt(source)
        if cm is not None:
            return cm

    json_candidates: list[str] = []
    if isinstance(source, str):
        if source.endswith(".json"):
            json_candidates.append(source)
        if os.path.isdir(source):
            json_candidates.append(os.path.join(source, CLASS_MAPPING_FILENAME))
        json_candidates.append(
            os.path.join(os.path.dirname(source), CLASS_MAPPING_FILENAME)
        )
    for jp in json_candidates:
        tried.append(jp)
        if os.path.isfile(jp):
            with open(jp) as fh:
                return ClassMapping.from_json_dict(json.load(fh))

    if isinstance(source, str) and os.path.isdir(source):
        hits = sorted(glob.glob(os.path.join(source, "**", "model_*.pth"), recursive=True))
        for path in reversed(hits):
            tried.append(path)
            cm = _from_ckpt(path)
            if cm is not None:
                return cm

    try:
        from detectron2.data import MetadataCatalog

        return ClassMapping.from_metadata(MetadataCatalog.get(source))
    except Exception:  # noqa: BLE001
        pass

    raise FileNotFoundError(
        f"could not resolve a ClassMapping from {source!r}; tried {tried} and MetadataCatalog"
    )


def set_num_classes_from_metadata(cfg, dataset_name: str) -> None:
    """Set ``cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES`` from a compact-mapped surgical
    dataset's metadata. No-op for any dataset not registered through
    :func:`apply_class_mapping_to_metadata` (stock COCO / ADE / panoptic configs
    keep their YAML value). ``cfg`` must be unfrozen.

    - sentinel ``-1`` (or ``None``)  -> set to the derived count
    - ``1`` (single-class mode)       -> left alone (warns)
    - any other mismatch              -> ``ValueError`` (a stale/wrong literal)
    """
    from detectron2.data import MetadataCatalog

    md = MetadataCatalog.get(dataset_name)
    if not md.get("class_mapping_entries", None):
        return
    n = len(md.thing_classes)
    cur = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    if cur in (-1, None):
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = n
    elif cur == 1:
        logger.warning(
            "MODEL.SEM_SEG_HEAD.NUM_CLASSES pinned to 1 (single-class mode); dataset "
            "%r has %d effective classes - not overriding.",
            dataset_name,
            n,
        )
    elif cur != n:
        raise ValueError(
            f"MODEL.SEM_SEG_HEAD.NUM_CLASSES={cur} in config but dataset "
            f"{dataset_name!r} has {n} effective classes: {list(md.thing_classes)}"
        )

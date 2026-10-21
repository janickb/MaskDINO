"""Inspection/visualization helpers that are not part of train or eval.

Deliberately not re-exported from `maskdino/__init__.py`: nothing in the
training or evaluation path imports this, and the viewer imports the submodule
directly (`from maskdino.visualization.attention import read_attention`).
"""

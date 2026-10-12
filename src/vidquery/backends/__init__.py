"""Inference backends: ``pytorch``, ``onnxruntime`` and ``tensorrt`` behind one interface."""

from __future__ import annotations

from vidquery.backends.base import BACKEND_KINDS, BackendSpec, InferenceBackend

__all__ = ["BACKEND_KINDS", "BackendSpec", "InferenceBackend"]

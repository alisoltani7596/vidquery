"""Cross-backend numerical parity checks.

Each candidate backend is compared against a reference (PyTorch FP32 by default) on the same
preprocessed inputs. Metrics:

* ``max_abs`` / ``mean_abs`` and an ``allclose(atol, rtol)`` check on raw outputs,
* ``min_cosine`` over per-sample output vectors (what matters for retrieval),
* for the detector, ``det_f1``: agreement of the *decoded* person boxes (IoU >= 0.5), which is
  robust to harmless reorderings inside NMS-free heads.

Default tolerances depend on the candidate's precision and can be overridden.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from vidquery.backends.base import BackendSpec, InferenceBackend
from vidquery.config import Settings
from vidquery.models.detector import box_iou, decode_detections, preprocess_letterbox
from vidquery.models.encoder import preprocess_images
from vidquery.models.registry import Component, build_tokenizer, create_backend, image_params
from vidquery.types import Detection

log = logging.getLogger(__name__)

DEFAULT_TEXTS = [
    "a man in a blue shirt walking left",
    "a woman pushing a stroller",
    "person wearing a red jacket",
    "a child running",
    "someone carrying a backpack",
    "two people talking",
    "a person riding a bicycle",
    "an old man with a cane",
]


class Tolerance(BaseModel):
    atol: float | None = None
    rtol: float | None = None
    min_cosine: float | None = None
    min_det_f1: float | None = None


DEFAULT_TOLERANCES: dict[str, Tolerance] = {
    "fp32": Tolerance(atol=1e-3, rtol=1e-3, min_cosine=0.9999, min_det_f1=0.99),
    "fp16": Tolerance(min_cosine=0.999, min_det_f1=0.95),
    "int8": Tolerance(min_cosine=0.98, min_det_f1=0.85),
}


class ParityResult(BaseModel):
    reference: str
    candidate: str
    description: str
    max_abs: float
    mean_abs: float
    allclose: bool | None
    min_cosine: float
    mean_cosine: float
    det_f1: float | None = None
    tolerance: Tolerance
    passed: bool
    failures: list[str]


class ParityReport(BaseModel):
    component: str
    model: str
    num_samples: int
    results: list[ParityResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_markdown(self) -> str:
        rows = [
            f"### Parity: `{self.component}` ({self.model}), {self.num_samples} samples",
            "",
            "| candidate | vs | max abs | mean abs | min cos | det F1 | pass |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.results:
            f1 = "-" if r.det_f1 is None else f"{r.det_f1:.4f}"
            verdict = "yes" if r.passed else "NO: " + "; ".join(r.failures)
            rows.append(
                f"| {r.description} | {r.reference} | {r.max_abs:.3e} | {r.mean_abs:.3e} | "
                f"{r.min_cosine:.6f} | {f1} | {verdict} |"
            )
        return "\n".join(rows)


def synthetic_images(n: int, size: tuple[int, int] = (480, 640), seed: int = 0) -> list[np.ndarray]:
    """Noise plus coloured rectangles: deterministic inputs when no real frames are given."""
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(n):
        img = rng.integers(60, 200, size=(*size, 3), dtype=np.uint8)
        for _ in range(int(rng.integers(1, 5))):
            x1, y1 = int(rng.integers(0, size[1] - 40)), int(rng.integers(0, size[0] - 80))
            w, h = int(rng.integers(20, 80)), int(rng.integers(40, 200))
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
        images.append(img)
    return images


def _cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = a.reshape(a.shape[0], -1).astype(np.float64)
    b2 = b.reshape(b.shape[0], -1).astype(np.float64)
    num = (a2 * b2).sum(1)
    den = np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-30), 1.0)


def detection_f1(
    ref: Sequence[Sequence[Detection]], out: Sequence[Sequence[Detection]], iou: float = 0.5
) -> float:
    matched = total = 0
    for r, o in zip(ref, out, strict=True):
        total += len(r) + len(o)
        if not r or not o:
            continue
        ious = box_iou(np.array([d.box for d in r]), np.array([d.box for d in o]))
        pairs = sorted(
            ((float(ious[i, j]), i, j) for i, j in zip(*np.nonzero(ious >= iou), strict=True)),
            reverse=True,
        )
        used_r: set[int] = set()
        used_o: set[int] = set()
        for _, i, j in pairs:  # greedy one-to-one matching, best IoU first
            if i in used_r or j in used_o:
                continue
            used_r.add(i)
            used_o.add(j)
            matched += 2
    return 1.0 if total == 0 else matched / total


def compare(ref: np.ndarray, out: np.ndarray, tol: Tolerance) -> tuple[dict[str, Any], list[str]]:
    if ref.shape != out.shape:
        return (
            {
                "max_abs": float("inf"),
                "mean_abs": float("inf"),
                "allclose": False,
                "min_cosine": -1.0,
                "mean_cosine": -1.0,
            },
            [f"shape {out.shape} != {ref.shape}"],
        )
    diff = np.abs(ref.astype(np.float64) - out.astype(np.float64))
    cos = _cosines(ref, out)
    metrics: dict[str, Any] = {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
        "allclose": None,
    }
    failures: list[str] = []
    if tol.atol is not None or tol.rtol is not None:
        ok = bool(np.allclose(out, ref, atol=tol.atol or 0.0, rtol=tol.rtol or 0.0))
        metrics["allclose"] = ok
        if not ok:
            failures.append(f"allclose(atol={tol.atol}, rtol={tol.rtol}) failed")
    if tol.min_cosine is not None and metrics["min_cosine"] < tol.min_cosine:
        failures.append(f"min cosine {metrics['min_cosine']:.6f} < {tol.min_cosine}")
    return metrics, failures


def _prepare_inputs(
    settings: Settings,
    component: Component,
    images: Sequence[np.ndarray] | None,
    texts: Sequence[str] | None,
    batch_size: int,
) -> tuple[list[np.ndarray], list[Any]]:
    """Preprocessed input batches (+ letterbox metadata for the detector)."""
    batches: list[np.ndarray] = []
    metas: list[Any] = []
    if component == "text_encoder":
        tok = build_tokenizer(settings)
        texts = list(texts or DEFAULT_TEXTS)
        for i in range(0, len(texts), batch_size):
            batches.append(tok(texts[i : i + batch_size]))
            metas.append(None)
        return batches, metas
    imgs = list(images) if images is not None else synthetic_images(16, seed=settings.seed)
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i : i + batch_size]
        if component == "detector":
            arr, meta = preprocess_letterbox(chunk, settings.detector.input_size)
            batches.append(arr)
            metas.append(meta)
        else:
            size, mean, std = image_params(settings)
            batches.append(preprocess_images(chunk, size, mean, std))
            metas.append(None)
    return batches, metas


def _run(backend: InferenceBackend, batches: list[np.ndarray]) -> np.ndarray:
    outs = [backend(b)[0] for b in batches]
    return np.concatenate(outs, axis=0)


def run_parity(
    settings: Settings,
    component: Component,
    candidates: Sequence[BackendSpec | str],
    *,
    reference: BackendSpec | str = "pytorch:fp32",
    device: str = "cpu",
    reference_device: str | None = None,
    images: Sequence[np.ndarray] | None = None,
    texts: Sequence[str] | None = None,
    batch_size: int = 4,
    tolerances: dict[str, Tolerance] | None = None,
) -> ParityReport:
    tols = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    ref_spec = BackendSpec.parse(reference) if isinstance(reference, str) else reference
    batches, metas = _prepare_inputs(settings, component, images, texts, batch_size)

    ref_backend = create_backend(settings, component, ref_spec, reference_device or device)
    ref_out = _run(ref_backend, batches)
    ref_backend.close()
    del ref_backend
    gc.collect()

    def decode(raw: np.ndarray) -> list[list[Detection]]:
        dets: list[list[Detection]] = []
        offset = 0
        cfg = settings.detector
        for meta in metas:
            n = len(meta)
            dets.extend(
                decode_detections(
                    raw[offset : offset + n],
                    meta,
                    conf_threshold=cfg.conf_threshold,
                    iou_threshold=cfg.iou_threshold,
                    person_class=cfg.person_class,
                    max_detections=cfg.max_detections,
                )
            )
            offset += n
        return dets

    ref_dets = decode(ref_out) if component == "detector" else None
    results: list[ParityResult] = []
    for cand in candidates:
        spec = BackendSpec.parse(cand) if isinstance(cand, str) else cand
        tol = tols[spec.precision]
        try:
            backend = create_backend(settings, component, spec, device)
        except (FileNotFoundError, RuntimeError, ImportError, ValueError) as exc:
            log.error("Skipping %s: %s", spec, exc)
            results.append(
                ParityResult(
                    reference=str(ref_spec),
                    candidate=str(spec),
                    description=str(spec),
                    max_abs=float("inf"),
                    mean_abs=float("inf"),
                    allclose=None,
                    min_cosine=-1.0,
                    mean_cosine=-1.0,
                    tolerance=tol,
                    passed=False,
                    failures=[f"unavailable: {exc}"],
                )
            )
            continue
        out = _run(backend, batches)
        desc = backend.description
        backend.close()
        del backend
        gc.collect()
        metrics, failures = compare(ref_out, out, tol)
        det_f1 = None
        if ref_dets is not None and out.shape == ref_out.shape:
            det_f1 = detection_f1(ref_dets, decode(out))
            if tol.min_det_f1 is not None and det_f1 < tol.min_det_f1:
                failures.append(f"det F1 {det_f1:.4f} < {tol.min_det_f1}")
            # Raw dense-head scores below threshold can drift harmlessly; boxes are what count.
            if spec.precision != "fp32":
                failures = [f for f in failures if not f.startswith("min cosine")]
        results.append(
            ParityResult(
                reference=str(ref_spec),
                candidate=str(spec),
                description=desc,
                det_f1=det_f1,
                tolerance=tol,
                passed=not failures,
                failures=failures,
                **metrics,
            )
        )
    return ParityReport(
        component=component,
        model=settings.detector.model if component == "detector" else settings.encoder.model,
        num_samples=int(ref_out.shape[0]),
        results=results,
    )

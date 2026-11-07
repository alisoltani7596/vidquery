"""Cross-backend numerical parity checks.

Each candidate backend is compared against a reference (PyTorch FP32 by default) on the same
preprocessed inputs. Metrics:

* encoders: ``max_abs`` / ``mean_abs``, an ``allclose(atol, rtol)`` check on raw outputs and
  ``min_cosine`` over per-sample embeddings (what matters for retrieval);
* detector: ``det_f1``, agreement of the *decoded* person boxes (IoU >= 0.5, score threshold from
  the config), plus the largest box and score error over matched high-confidence detections.
  The raw head output is not compared elementwise: an NMS-free head returns its top-k
  detections sorted by score, so near-ties reorder rows without changing any detection.

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


# Box/score errors are reported over matched detections whose reference score is at least this
# (or the detector's conf_threshold, if higher): near-threshold boxes jitter more and matter less.
HIGH_CONFIDENCE = 0.5


class ParityResult(BaseModel):
    reference: str
    candidate: str
    description: str
    max_abs: float | None = None
    mean_abs: float | None = None
    allclose: bool | None = None
    min_cosine: float | None = None
    mean_cosine: float | None = None
    det_f1: float | None = None
    det_matched: int | None = None
    max_box_err: float | None = None
    max_score_err: float | None = None
    tolerance: Tolerance
    passed: bool
    failures: list[str]


class ParityReport(BaseModel):
    component: str
    model: str
    num_samples: int
    results: list[ParityResult]
    note: str = ""

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_markdown(self) -> str:
        rows = [f"### Parity: `{self.component}` ({self.model}), {self.num_samples} samples", ""]
        if self.note:
            rows += [self.note, ""]
        if self.component == "detector":
            rows += [
                "| candidate | vs | det F1 | matched | max box err (px) | max score err | pass |",
                "|---|---|---|---|---|---|---|",
            ]
        else:
            rows += [
                "| candidate | vs | max abs | mean abs | min cos | pass |",
                "|---|---|---|---|---|---|",
            ]
        for r in self.results:
            verdict = "yes" if r.passed else "NO: " + "; ".join(r.failures)
            if self.component == "detector":
                cells = [
                    _fmt(r.det_f1, ".4f"),
                    _fmt(r.det_matched, "d"),
                    _fmt(r.max_box_err, ".2f"),
                    _fmt(r.max_score_err, ".4f"),
                ]
            else:
                cells = [_fmt(r.max_abs, ".3e"), _fmt(r.mean_abs, ".3e"), _fmt(r.min_cosine, ".6f")]
            rows.append(f"| {r.description} | {r.reference} | {' | '.join(cells)} | {verdict} |")
        return "\n".join(rows)


def _fmt(value: float | int | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


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


class DetectionMatch(BaseModel):
    f1: float
    matched: int
    max_box_err: float | None  # pixels, original image coordinates
    max_score_err: float | None


def match_detections(
    ref: Sequence[Sequence[Detection]],
    out: Sequence[Sequence[Detection]],
    iou: float = 0.5,
    high_conf: float = HIGH_CONFIDENCE,
) -> DetectionMatch:
    """Greedy one-to-one matching (best IoU first) of decoded detections, per image."""
    matched = total = 0
    box_errs: list[float] = []
    score_errs: list[float] = []
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
        for _, i, j in pairs:
            if i in used_r or j in used_o:
                continue
            used_r.add(i)
            used_o.add(j)
            matched += 1
            if r[i].score >= high_conf:
                box_errs.append(float(np.abs(np.subtract(r[i].box, o[j].box)).max()))
                score_errs.append(abs(r[i].score - o[j].score))
    return DetectionMatch(
        f1=1.0 if total == 0 else 2 * matched / total,
        matched=matched,
        max_box_err=max(box_errs) if box_errs else None,
        max_score_err=max(score_errs) if score_errs else None,
    )


def detection_f1(
    ref: Sequence[Sequence[Detection]], out: Sequence[Sequence[Detection]], iou: float = 0.5
) -> float:
    return match_detections(ref, out, iou).f1


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
    high_conf = max(settings.detector.conf_threshold, HIGH_CONFIDENCE)
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
        if ref_dets is not None:
            m = match_detections(ref_dets, decode(out), high_conf=high_conf)
            metrics: dict[str, Any] = {
                "det_f1": m.f1,
                "det_matched": m.matched,
                "max_box_err": m.max_box_err,
                "max_score_err": m.max_score_err,
            }
            failures: list[str] = []
            if tol.min_det_f1 is not None and m.f1 < tol.min_det_f1:
                failures.append(f"det F1 {m.f1:.4f} < {tol.min_det_f1}")
        else:
            metrics, failures = compare(ref_out, out, tol)
        results.append(
            ParityResult(
                reference=str(ref_spec),
                candidate=str(spec),
                description=desc,
                tolerance=tol,
                passed=not failures,
                failures=failures,
                **metrics,
            )
        )
    note = ""
    if component == "detector":
        note = (
            f"Judged by detection F1 (IoU >= 0.5, score >= {settings.detector.conf_threshold:g}); "
            f"box/score errors over matched detections with reference score >= {high_conf:g}."
        )
    return ParityReport(
        component=component,
        model=settings.detector.model if component == "detector" else settings.encoder.model,
        num_samples=int(ref_out.shape[0]),
        results=results,
        note=note,
    )

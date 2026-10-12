"""Command line interface.

    vidquery ingest <video> [--video-id ID] [--fps 2]
    vidquery query "<text>" [-k 10] [--verify]
    vidquery demo "<text>" --out demo.gif
    vidquery serve
    vidquery export onnx|trt ...
    vidquery parity --component image_encoder --backends onnxruntime tensorrt:fp16
    vidquery bench ... / vidquery eval ...   (same flags as scripts/benchmark.py, eval_retrieval.py)

Every command accepts ``--config file.yaml`` and repeated ``--set key.path=value`` overrides.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from vidquery.config import Settings, load_config, resolve_device

log = logging.getLogger("vidquery")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", help="YAML config (default: $VIDQUERY_CONFIG)")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. detector.backend=onnxruntime",
    )
    p.add_argument("--verbose", "-v", action="store_true")


def _settings(args: argparse.Namespace) -> Settings:
    return load_config(args.config, args.overrides)


def cmd_ingest(args: argparse.Namespace) -> int:
    from vidquery.pipeline import VidQuery

    engine = VidQuery(_settings(args))
    paths = [Path(p) for p in args.videos]
    if args.video_id and len(paths) > 1:
        raise SystemExit("--video-id only works with a single video")
    for path in paths:
        stats = engine.ingest(path, args.video_id, args.fps, replace=not args.no_replace)
        print(
            stats.model_dump_json(indent=2)
            if args.json
            else f"{stats.video_id}: {stats.sampled_frames} frames, {stats.crops} crops, "
            f"{stats.vectors} vectors in {stats.elapsed_s:.1f}s "
            f"({stats.ingest_fps:.1f} frames/s, {stats.realtime_factor:.1f}x realtime)"
        )
    engine.close()
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from vidquery.pipeline import VidQuery

    engine = VidQuery(_settings(args))
    result = engine.query(args.text, args.k, args.kind, args.verify, args.video or None)
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0
    if not result.segments:
        print("No results (is the index empty?)")
        return 1
    tag = " (VLM-verified)" if result.verified else ""
    print(f"Top {len(result.segments)} segments for {args.text!r}{tag}:")
    for i, seg in enumerate(result.segments, 1):
        vlm = f" vlm={seg.verifier_score:.2f}" if seg.verifier_score is not None else ""
        print(
            f"{i:2d}. {seg.video_id}  {seg.start:7.2f}s - {seg.end:7.2f}s  "
            f"score={seg.score:.4f}{vlm}  hits={seg.num_hits}"
        )
        for hit in seg.boxes()[: args.boxes]:
            assert hit.box is not None
            box = ", ".join(f"{v:.0f}" for v in hit.box)
            print(f"      t={hit.timestamp:7.2f}s box=[{box}] sim={hit.score:.4f}")
    engine.close()
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from vidquery.demo import render_demo
    from vidquery.pipeline import VidQuery

    engine = VidQuery(_settings(args))
    out, result = render_demo(
        engine,
        args.text,
        args.out,
        num_segments=args.segments,
        context_s=args.context,
        verify=args.verify,
    )
    print(f"Wrote {out} ({len(result.segments)} segment(s))")
    engine.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from vidquery.api import serve

    serve(_settings(args), args.host, args.port)
    return 0


def cmd_videos(args: argparse.Namespace) -> int:
    from vidquery.pipeline import VidQuery

    index = VidQuery(_settings(args)).index
    if index is None:
        print("Index is empty")
        return 0
    for v in index.videos.values():
        print(f"{v.video_id}\t{v.duration:.1f}s\t{v.fps:.2f}fps\t{v.width}x{v.height}\t{v.path}")
    print(f"{len(index)} vectors ({index.count('frame')} frames, {index.count('crop')} crops)")
    return 0


def cmd_export_onnx(args: argparse.Namespace) -> int:
    from vidquery.export.onnx_export import export_component

    settings = _settings(args)
    for comp in args.components:
        path = export_component(
            settings, comp, opset=args.opset, exporter=args.exporter, max_batch=args.max_batch
        )
        print(f"{comp}: {path}")
    return 0


def cmd_export_trt(args: argparse.Namespace) -> int:
    from vidquery.export import build_trt_engines

    settings = _settings(args)
    for comp in args.components:
        built = build_trt_engines(
            settings,
            comp,
            args.precisions,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            workspace_gb=args.workspace_gb,
            allow_tf32=args.allow_tf32,
            calib_sources=args.calib,
            calib_images=args.calib_images,
            int8_method=args.int8_method,
            timing_cache=Path(args.timing_cache) if args.timing_cache else None,
        )
        for precision, path in built.items():
            print(f"{comp} [{precision}]: {path}")
    return 0


def cmd_parity(args: argparse.Namespace) -> int:
    from vidquery.parity import DEFAULT_TOLERANCES, Tolerance, run_parity

    settings = _settings(args)
    device = resolve_device(args.device or settings.device)
    images = None
    if args.images:
        from vidquery.export.calibration import collect_frames

        images = collect_frames(args.images, args.num_images)
    tolerances: dict[str, Tolerance] = {}
    if args.tolerances:
        raw = json.loads(Path(args.tolerances).read_text())
        tolerances = {
            k: Tolerance(**{**DEFAULT_TOLERANCES[k].model_dump(), **v}) for k, v in raw.items()
        }
    reports = []
    ok = True
    for comp in args.components:
        report = run_parity(
            settings,
            comp,
            args.backends,
            reference=args.reference,
            device=device,
            images=images,
            batch_size=args.batch_size,
            tolerances=tolerances,
        )
        reports.append(report)
        ok &= report.passed
        print(report.to_markdown(), end="\n\n")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps([r.model_dump() for r in reports], indent=2))
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    from vidquery.models.registry import COMPONENTS

    parser = argparse.ArgumentParser(prog="vidquery", description="Text-to-video person search")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Index one or more videos")
    _add_common(p)
    p.add_argument("videos", nargs="+", help="Video files or image-sequence directories")
    p.add_argument("--video-id")
    p.add_argument("--fps", type=float, help="Sampling rate (default: ingest.sample_fps)")
    p.add_argument("--no-replace", action="store_true", help="Fail if the video is indexed")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", help="Search the index")
    _add_common(p)
    p.add_argument("text")
    p.add_argument("-k", type=int, default=None, help="Number of segments")
    p.add_argument("--kind", choices=["both", "crop", "frame"])
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--video", action="append", default=[], help="Restrict to video id(s)")
    p.add_argument("--boxes", type=int, default=3, help="Boxes printed per segment")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("demo", help="Write an annotated clip or GIF for a query")
    _add_common(p)
    p.add_argument("text")
    p.add_argument("--out", default="demo.gif", help=".gif, .mp4 or .avi")
    p.add_argument("--segments", type=int, default=1)
    p.add_argument("--context", type=float, default=0.5, help="Seconds around each segment")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=None)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("serve", help="Run the FastAPI server")
    _add_common(p)
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("videos", help="List indexed videos")
    _add_common(p)
    p.set_defaults(func=cmd_videos)

    export = sub.add_parser("export", help="Export models (onnx | trt)")
    esub = export.add_subparsers(dest="target", required=True)
    p = esub.add_parser("onnx", help="torch -> ONNX (dynamic batch)")
    _add_common(p)
    p.add_argument("--components", nargs="+", default=list(COMPONENTS), choices=list(COMPONENTS))
    p.add_argument("--exporter", choices=["dynamo", "torchscript"], default="dynamo")
    p.add_argument("--opset", type=int, help="Default: 18 (dynamo) / 17 (torchscript)")
    p.add_argument("--max-batch", type=int, default=64)
    p.set_defaults(func=cmd_export_onnx)

    p = esub.add_parser("trt", help="ONNX -> TensorRT engines")
    _add_common(p)
    p.add_argument(
        "--components", nargs="+", default=["detector", "image_encoder"], choices=list(COMPONENTS)
    )
    p.add_argument(
        "--precisions", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16", "int8"]
    )
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--workspace-gb", type=float, default=8.0)
    p.add_argument("--allow-tf32", action="store_true", help="Let fp32 engines use TF32")
    p.add_argument("--calib", nargs="+", default=[], help="Videos/dirs/images for INT8 calibration")
    p.add_argument("--calib-images", type=int, default=512)
    p.add_argument("--int8-method", choices=["calibrator", "modelopt"], default="calibrator")
    p.add_argument("--timing-cache", help="Path to a TensorRT timing cache (speeds up rebuilds)")
    p.set_defaults(func=cmd_export_trt)

    p = sub.add_parser("parity", help="Compare backends numerically")
    _add_common(p)
    p.add_argument(
        "--components", nargs="+", default=["detector", "image_encoder"], choices=list(COMPONENTS)
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=["onnxruntime"],
        help="Candidates, e.g. onnxruntime tensorrt:fp32 tensorrt:fp16 tensorrt:int8",
    )
    p.add_argument("--reference", default="pytorch:fp32")
    p.add_argument("--device")
    p.add_argument("--images", nargs="+", help="Videos/dirs/images to draw frames from")
    p.add_argument("--num-images", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--tolerances", help='JSON file, e.g. {"fp16": {"min_cosine": 0.998}}')
    p.add_argument("--out", help="Write the reports as JSON")
    p.set_defaults(func=cmd_parity)

    sub.add_parser("bench", help="Benchmark (see scripts/benchmark.py --help)", add_help=False)
    sub.add_parser(
        "eval", help="Retrieval eval (see scripts/eval_retrieval.py --help)", add_help=False
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bench":
        from vidquery.bench import main as bench_main

        return bench_main(argv[1:])
    if argv and argv[0] == "eval":
        from vidquery.evaluation import main as eval_main

        return eval_main(argv[1:])
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

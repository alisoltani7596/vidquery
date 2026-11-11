"""Qualitative geolocation on MOT17, compared with documented recording places where they exist.

``data/mot17_locations.json`` lists every sequence with ``status: known`` only when a source
states where it was filmed (with the URL and a quote); everything else is ``unknown`` with null
coordinates. Errors are reported for known sequences only.

    python scripts/geo_mot17.py --videos $SCRATCH/vidquery/data/MOT17/videos \
        --params results/geo-params.yaml --md results/geo-mot17.md --maps $VQ_RESULTS/geo-maps
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from vidquery.geo.geometry import haversine_km
from vidquery.geo.locate import GeoArea, LocateResult

log = logging.getLogger(__name__)

DEFAULT_LOCATIONS = Path(__file__).resolve().parents[3] / "data" / "mot17_locations.json"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".webm")


class LocationSource(BaseModel):
    url: str
    quote: str
    note: str | None = None


class SequenceLocation(BaseModel):
    sequence: str
    status: Literal["known", "unknown"]
    place: str | None = None
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    precision_km: float | None = Field(None, gt=0)
    scene: str | None = None
    sources: list[LocationSource] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SequenceLocation:
        has_coords = self.lat is not None or self.lon is not None
        if self.status == "known":
            if self.lat is None or self.lon is None or not self.place or not self.sources:
                raise ValueError(f"{self.sequence}: known needs place, lat, lon and a source")
        elif has_coords or self.place:
            raise ValueError(f"{self.sequence}: unknown locations must not carry a place or coords")
        return self


class LocationsFile(BaseModel):
    about: str
    checked: str
    sequences: list[SequenceLocation]


def load_locations(path: str | Path = DEFAULT_LOCATIONS) -> dict[str, SequenceLocation]:
    data = LocationsFile.model_validate_json(Path(path).read_text(encoding="utf-8"))
    out = {s.sequence: s for s in data.sequences}
    if len(out) != len(data.sequences):
        raise ValueError("duplicate sequence in locations file")
    return out


def compare(result: LocateResult, loc: SequenceLocation | None) -> dict[str, Any]:
    """Error and in-radius flag against a documented location (None when unknown)."""
    if loc is None or loc.status != "known" or result.estimate is None:
        return {"error_km": None, "inside_radius": None}
    assert loc.lat is not None and loc.lon is not None
    err = float(haversine_km(result.estimate.lat, result.estimate.lon, loc.lat, loc.lon))
    return {"error_km": round(err, 1), "inside_radius": err <= result.estimate.radius_km}


def _area(a: GeoArea | None) -> str:
    return "none" if a is None else f"{a.lat}, {a.lon} ± {a.radius_km:g} km"


def markdown(rows: Sequence[dict[str, Any]]) -> str:
    head = [
        "sequence",
        "documented place",
        "GeoCLIP estimate",
        "share",
        "alternatives",
        "VLM (country / city)",
        "frames used",
        "error km",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        res: LocateResult = r["result"]
        loc: SequenceLocation | None = r["location"]
        vlm = res.vlm
        vlm_text = "–"
        if vlm is not None:
            vlm_text = (
                f"{vlm.country or 'none'} / {vlm.city or '–'}"
                if vlm.frames_answered
                else "unavailable"
            )
        cells = [
            r["sequence"],
            loc.place if loc and loc.place else "unknown",
            _area(res.estimate),
            "–" if res.estimate is None else f"{res.estimate.share:.2f}",
            "; ".join(_area(a) for a in res.alternatives) or "–",
            vlm_text,
            f"{res.frames_used}/{res.frames_sampled}",
            "–" if r["error_km"] is None else f"{r['error_km']:.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Recording places are not documented for these sequences (see "
        "`data/mot17_locations.json`), so this table is qualitative: there is no ground truth to "
        "score against. Estimates are coarse areas, not pins.",
    ]
    return "\n".join(lines)


def find_videos(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        p
        for p in root.iterdir()
        if (p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        or (p.is_dir() and p.name.startswith("MOT"))
    )


def main(argv: Sequence[str] | None = None) -> int:
    from vidquery.config import load_config
    from vidquery.geo.aggregate import AggregateParams
    from vidquery.geo.evaluation import read_params
    from vidquery.geo.locate import GeoLocator
    from vidquery.geo.mapview import render_map_html

    p = argparse.ArgumentParser(description="Qualitative geolocation on MOT17")
    p.add_argument("--videos", required=True, type=Path, help="Directory of MOT17 videos")
    p.add_argument("--locations", type=Path, default=DEFAULT_LOCATIONS)
    p.add_argument("--params", help="Frozen params YAML from geo_eval.py tune")
    p.add_argument("--vlm", action="store_true", help="Also ask the VLM (verifier endpoint)")
    p.add_argument("--maps", type=Path, help="Write one HTML map per sequence here")
    p.add_argument("--md", default="results/geo-mot17.md")
    p.add_argument("--json")
    p.add_argument("--config", "-c")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = load_config(args.config, args.overrides)
    params = AggregateParams.from_config(settings.geo)
    if args.params:
        params = read_params(args.params, params)
    locations = load_locations(args.locations)
    locator = GeoLocator(settings)
    rows = []
    for video in find_videos(args.videos):
        seq = video.stem
        result = locator.locate(video, vlm=args.vlm, params=params)
        loc = locations.get(seq)
        rows.append({"sequence": seq, "result": result, "location": loc, **compare(result, loc)})
        if args.maps:
            args.maps.mkdir(parents=True, exist_ok=True)
            html = render_map_html(
                result,
                tiles_url=settings.geo.map_tiles_url,
                attribution=settings.geo.map_attribution,
            )
            (args.maps / f"{seq}.html").write_text(html, encoding="utf-8")
    locator.close()
    if not rows:
        raise SystemExit(f"No videos found in {args.videos}")
    md = markdown(rows)
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(md + "\n")
    if args.json:
        payload = [
            {
                "sequence": r["sequence"],
                "error_km": r["error_km"],
                "inside_radius": r["inside_radius"],
                "result": r["result"].model_dump(mode="json"),
            }
            for r in rows
        ]
        Path(args.json).write_text(json.dumps(payload, indent=2))
    print(md)
    return 0

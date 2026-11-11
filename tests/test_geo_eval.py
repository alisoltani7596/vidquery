"""Geolocation benchmark tooling: OSV-5M subset building, predict -> tune (val) -> test (once),
and the MOT17 locations file. Uses the mock geo backend and tiny generated images."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from vidquery.config import Settings
from vidquery.geo.aggregate import AggregateParams
from vidquery.geo.backends import MOCK_ANCHORS, MockGeoBackend
from vidquery.geo.evaluation import (
    build_clips,
    evaluate_clips,
    load_manifest,
    load_predictions,
    read_params,
    save_predictions,
    tune,
)
from vidquery.geo.evaluation import (
    main as geo_eval_main,
)
from vidquery.geo.mot17 import (
    DEFAULT_LOCATIONS,
    SequenceLocation,
    compare,
    load_locations,
)
from vidquery.geo.osv5m import (
    index_zip_members,
    make_subset,
    read_rows,
    split_sequences,
)
from vidquery.geo.osv5m import (
    main as osv_main,
)

#: Brightness per anchor for the mock backend (quartiles of the mean pixel value).
BRIGHTNESS = (30, 100, 160, 230)


def fake_osv(root: Path, n_seq: int = 60, per_seq: int = 3) -> None:
    """A tiny OSV-5M look-alike: test.csv + one zip of JPEGs whose brightness encodes the
    anchor city the mock backend will predict. Most images are labelled at their anchor; every
    fifth one is labelled 3,000 km away so the metrics are not trivially perfect."""
    (root / "images" / "test").mkdir(parents=True)
    rows = []
    with zipfile.ZipFile(root / "images" / "test" / "00.zip", "w") as zf:
        for s in range(n_seq):
            a = s % 4
            for j in range(per_seq):
                img_id = f"{1000 + s * per_seq + j}"
                lat, lon = MOCK_ANCHORS[a]
                if (s * per_seq + j) % 5 == 0:
                    lat = lat - 27.0 if lat > 0 else lat + 27.0
                lat += 0.01 * j  # one drive: a few hundred metres apart
                img = np.full((48, 64, 3), BRIGHTNESS[a], np.uint8)
                ok, buf = cv2.imencode(".jpg", img)
                assert ok
                zf.writestr(f"00/{img_id}.jpg", buf.tobytes())
                rows.append(
                    {
                        "id": img_id,
                        "latitude": lat,
                        "longitude": lon,
                        "thumb_original_url": "https://example.invalid/x",
                        "country": ["FR", "US", "JP", "AU"][a],
                        "sequence": f"seq{s:03d}",
                        "captured_at": str(1_600_000_000_000 + j),
                    }
                )
    with open(root / "test.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_split_by_sequence_is_disjoint_and_deterministic() -> None:
    split = split_sequences([f"s{i}" for i in range(100)], 0.3, seed=0)
    assert sum(v == "val" for v in split.values()) == 30
    assert split == split_sequences(reversed([f"s{i}" for i in range(100)]), 0.3, seed=0)
    assert split != split_sequences([f"s{i}" for i in range(100)], 0.3, seed=1)


def test_osv_subset_and_manifests(tmp_path: Path) -> None:
    fake_osv(tmp_path)
    index = index_zip_members([tmp_path / "images" / "test" / "00.zip"])
    assert len(index) == 180 and index["1000"][1] == "00/1000.jpg"
    rows = read_rows(tmp_path / "test.csv", set(index))
    images, seqs, summary = make_subset(rows, n_val=10, n_test=20, seed=0, seq_clips=5)
    val_seqs = {r["sequence"] for r in images if r["split"] == "val"}
    test_seqs = {r["sequence"] for r in images if r["split"] == "test"}
    assert len([r for r in images if r["split"] == "val"]) == 10
    assert len([r for r in images if r["split"] == "test"]) == 20
    assert not val_seqs & test_seqs and len(val_seqs) == 10  # one image per sequence
    seq_splits = {(r["clip"], r["split"]) for r in seqs}
    assert len({c for c, _ in seq_splits}) == len(seq_splits) == 10  # 5 clips per split
    assert summary["license"] == "CC BY-SA 4.0"
    assert make_subset(rows, 10, 20, seed=0, seq_clips=5)[0] == images

    assert (
        osv_main(["--out-dir", str(tmp_path), "--val", "10", "--test", "20", "--seq-clips", "5"])
        == 0
    )
    manifest = load_manifest(tmp_path / "manifest-images.jsonl")
    assert len(manifest) == 30 and Path(manifest[0]["zip"]).exists()
    assert manifest[0]["mode"] == "image" and manifest[0]["clip"] == manifest[0]["id"]
    assert json.loads((tmp_path / "subset.json").read_text())["images"] == {"val": 10, "test": 20}


def test_predict_tune_test_once(
    tmp_path: Path, mock_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_osv(tmp_path)
    assert (
        osv_main(["--out-dir", str(tmp_path), "--val", "20", "--test", "30", "--seq-clips", "6"])
        == 0
    )
    settings = mock_settings.model_copy(deep=True)
    settings.geo.backend = "mock"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(settings.model_dump(mode="json")))
    manifests = [
        "--manifest",
        str(tmp_path / "manifest-images.jsonl"),
        "--manifest",
        str(tmp_path / "manifest-sequences.jsonl"),
    ]
    preds = tmp_path / "preds.npz"
    assert (
        geo_eval_main(["predict", "-c", str(cfg), *manifests, "--out", str(preds), "--top-k", "8"])
        == 0
    )
    loaded = load_predictions(preds)
    assert len(loaded) >= 50 and next(iter(loaded.values())).lats.shape == (8,)

    params, val_md = tmp_path / "params.yaml", tmp_path / "val.md"
    grid = tmp_path / "grid.json"
    grid.write_text(
        json.dumps(
            {"top_k": [1, 4, 8], "cluster_km": [50, 500], "frame_weighting": ["uniform", "mass"]}
        )
    )
    code = geo_eval_main(
        [
            "tune",
            "-c",
            str(cfg),
            *manifests,
            "--preds",
            str(preds),
            "--grid",
            str(grid),
            "--params-out",
            str(params),
            "--md",
            str(val_md),
        ]
    )
    assert code == 0
    frozen = read_params(params)
    assert frozen.top_k in (1, 4, 8) and 0 < frozen.confidence_mass < 1
    assert params.read_text().startswith("# Frozen geolocation aggregation, tuned on split=val")
    assert "Selected by median error" in val_md.read_text()

    out = tmp_path / "eval.md"
    test_args = [
        "test",
        "-c",
        str(cfg),
        *manifests,
        "--preds",
        str(preds),
        "--params",
        str(params),
        "--md",
        str(out),
        "--json",
        str(tmp_path / "eval.json"),
    ]
    assert geo_eval_main(test_args) == 0
    text = out.read_text()
    assert "image: aggregation frozen on val" in text and "sequence: GeoCLIP top-1" in text
    assert "<25 km %" in text and "median km" in text
    results = json.loads((tmp_path / "eval.json").read_text())
    tuned = results["modes"]["image"]["tuned"]
    assert tuned["n"] == 30 and 0.6 <= tuned["acc@25km"] <= 0.95  # every 5th label is off
    # The test split is reported once.
    capsys.readouterr()
    assert geo_eval_main(test_args) == 2
    assert "reported once" in capsys.readouterr().out
    assert geo_eval_main([*test_args, "--force"]) == 0


def test_tune_uses_only_val_and_metrics(tmp_path: Path) -> None:
    backend = MockGeoBackend()
    imgs = [np.full((8, 8, 3), BRIGHTNESS[i % 4], np.uint8) for i in range(8)]
    preds = backend.predict(imgs, 6)
    ids = [str(i) for i in range(8)]
    save_predictions(tmp_path / "p.npz", ids, preds)
    loaded = load_predictions(tmp_path / "p.npz")
    rows = []
    for i in range(8):
        lat, lon = MOCK_ANCHORS[i % 4]
        rows.append(
            {
                "id": str(i),
                "lat": lat,
                "lon": lon,
                "split": "val" if i < 4 else "test",
                "clip": str(i),
                "mode": "image",
            }
        )
    # Corrupt the *test* truth: tuning must not see it.
    for r in rows[4:]:
        r["lat"] = -r["lat"]
    val = build_clips(rows, loaded, split="val")
    assert [c.split for c in val] == ["val"] * 4
    res = tune(val, AggregateParams(), {"top_k": [1, 6], "cluster_km": [100]})
    assert res.rows[0]["median_km"] < 30 and len(res.coverage_rows) == 7
    m = evaluate_clips(val, res.best)
    assert m["n"] == 4 and m["acc@25km"] == 1.0 and m["acc@1km"] >= 0.5
    test = evaluate_clips(build_clips(rows, loaded, split="test"), res.best)
    assert test["acc@2500km"] < 1.0  # the flipped hemisphere is far away
    with pytest.raises(ValueError):
        tune([], AggregateParams())


def test_mot17_locations_file() -> None:
    locs = load_locations(DEFAULT_LOCATIONS)
    assert sorted(locs) == [f"MOT17-{i:02d}" for i in range(1, 15)]
    for loc in locs.values():
        assert loc.sources and all(s.url.startswith("https://") and s.quote for s in loc.sources)
        if loc.status == "unknown":
            assert loc.lat is None and loc.lon is None and loc.place is None
    with pytest.raises(ValueError, match="known needs"):
        SequenceLocation(sequence="X", status="known", lat=1.0, lon=2.0)
    with pytest.raises(ValueError, match="must not carry"):
        SequenceLocation(sequence="X", status="unknown", lat=1.0, lon=2.0)


def test_mot17_compare(mock_settings: Settings, synthetic_video: Path) -> None:
    from vidquery.geo.locate import GeoLocator

    s = mock_settings.model_copy(deep=True)
    s.geo.backend = "mock"
    result = GeoLocator(s).locate(synthetic_video)
    assert result.estimate is not None
    unknown = SequenceLocation(sequence="MOT17-02", status="unknown")
    assert compare(result, unknown) == {"error_km": None, "inside_radius": None}
    known = SequenceLocation(
        sequence="X",
        status="known",
        place="somewhere",
        lat=result.estimate.lat,
        lon=result.estimate.lon,
        sources=[{"url": "https://example.org", "quote": "filmed here"}],
    )
    assert compare(result, known) == {"error_km": 0.0, "inside_radius": True}

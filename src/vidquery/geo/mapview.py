"""A small, self-contained HTML map of a ``LocateResult`` (Leaflet + OpenStreetMap tiles).

The estimate is drawn as a circle with its confidence radius and the alternatives as dashed
circles; the page opens on the estimate and the panel entries jump to each circle. There is
deliberately no pin marker and the zoom is capped, so the page shows an area rather than
inviting a reading of a precise point. Leaflet 1.9.4 (BSD-2-Clause) is loaded from unpkg with
subresource-integrity hashes; tiles are © OpenStreetMap contributors (ODbL), and heavy use
should point ``geo.map_tiles_url`` at a tile server whose usage policy allows it.
"""

from __future__ import annotations

import html
import json
from typing import Any

from vidquery.geo.locate import LocateResult

LEAFLET_VERSION = "1.9.4"
LEAFLET_CSS_SRI = "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
LEAFLET_JS_SRI = "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
MAX_ZOOM = 12  # city scale at most: the estimate is never finer than its radius

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@__LV__/dist/leaflet.css"
      integrity="__CSS_SRI__" crossorigin="">
<script src="https://unpkg.com/leaflet@__LV__/dist/leaflet.js"
        integrity="__JS_SRI__" crossorigin=""></script>
<style>
  :root { --fg: #1f2933; --muted: #52606d; --bg: #ffffff; --panel: rgba(255,255,255,.94);
          --est: #2563eb; --alt: #d97706; }
  @media (prefers-color-scheme: dark) {
    :root { --fg: #e4e7eb; --muted: #9aa5b1; --bg: #111418; --panel: rgba(24,28,33,.94); }
  }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--fg);
               font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  #map { position: absolute; inset: 0; }
  #panel { position: absolute; z-index: 1000; top: 12px; right: 12px; max-width: 340px;
           max-height: calc(100% - 48px); overflow: auto; padding: 12px 14px;
           background: var(--panel); border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,.25); }
  #panel h1 { font-size: 15px; margin: 0 0 6px; word-break: break-all; }
  #panel h2 { font-size: 13px; margin: 10px 0 4px; }
  #panel p, #panel li { margin: 2px 0; }
  #panel ul { padding-left: 18px; margin: 4px 0; }
  .muted { color: var(--muted); font-size: 12px; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
            margin-right: 6px; vertical-align: baseline; }
  @media (max-width: 520px) { #panel { left: 12px; right: 12px; max-width: none;
                              top: auto; bottom: 24px; max-height: 45%; } }
</style>
</head>
<body>
<div id="map"></div>
<div id="panel"></div>
<script type="application/json" id="geo-data">__DATA__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("geo-data").textContent);
  const r = data.result;
  const panel = document.getElementById("panel");
  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined && text !== null) e.textContent = text;
    if (cls) e.className = cls;
    panel.appendChild(e);
    return e;
  }
  function fmt(a) { return a.lat + ", " + a.lon + " \\u00b1 " + a.radius_km + " km"; }
  function pct(x) { return Math.round(100 * x) + "%"; }

  const map = L.map("map", { worldCopyJump: true, maxZoom: data.maxZoom });
  L.tileLayer(data.tiles.url, { maxZoom: data.maxZoom, attribution: data.tiles.attribution })
    .addTo(map);
  map.setView(r.estimate ? [r.estimate.lat, r.estimate.lon] : [20, 0], 2);
  function show(layers) {
    map.fitBounds(L.featureGroup(layers).getBounds(), { padding: [24, 24], maxZoom: 8 });
  }
  function clickable(node, layers) {
    node.style.cursor = "pointer";
    node.title = "Show on the map";
    node.addEventListener("click", function () { show(layers); });
  }
  const est = r.estimate ? L.circle([r.estimate.lat, r.estimate.lon], {
    radius: r.estimate.radius_km * 1000, color: "#2563eb", weight: 2, fillOpacity: 0.18
  }).bindTooltip("Estimate: \\u00b1" + r.estimate.radius_km + " km, share " +
                 pct(r.estimate.share)).addTo(map) : null;
  const alts = r.alternatives.map(function (a, i) {
    return L.circle([a.lat, a.lon], {
      radius: a.radius_km * 1000, color: "#d97706", weight: 2, dashArray: "6 6", fillOpacity: 0.08
    }).bindTooltip("Alternative " + (i + 1) + ": \\u00b1" + a.radius_km + " km, share " +
                   pct(a.share)).addTo(map);
  });

  el("h1", r.video.split("/").pop());
  el("p", r.frames_used + " of " + r.frames_sampled + " frames used, " +
     r.frames_skipped_people + " skipped as mostly people (" + r.backend + ")", "muted");
  el("h2", "Estimate");
  if (est) {
    const p = el("p");
    const s = document.createElement("span");
    s.className = "swatch"; s.style.background = "var(--est)";
    p.appendChild(s);
    p.appendChild(document.createTextNode(
      fmt(r.estimate) + " (share " + pct(r.estimate.share) + ")"));
    clickable(p, [est]);
  } else {
    el("p", "No estimate: no usable frames.");
  }
  if (alts.length) {
    el("h2", "Alternatives");
    const ul = el("ul");
    r.alternatives.forEach(function (a, i) {
      const li = document.createElement("li");
      li.textContent = fmt(a) + " (share " + pct(a.share) + ")";
      clickable(li, [alts[i]]);
      ul.appendChild(li);
    });
    clickable(el("p", "Show all", "muted"), (est ? [est] : []).concat(alts));
  }
  if (r.vlm) {
    el("h2", "VLM second opinion (" + r.vlm.model + ")");
    if (r.vlm.error && !r.vlm.frames_answered) {
      el("p", "Unavailable: " + r.vlm.error, "muted");
    } else {
      el("p", (r.vlm.country || "no country") + (r.vlm.city ? " / " + r.vlm.city : "") +
         " (" + r.vlm.frames_answered + " of " + r.vlm.frames_asked + " frames answered)");
      if (r.vlm.clues.length) {
        const ul = el("ul");
        r.vlm.clues.forEach(function (c) {
          const li = document.createElement("li"); li.textContent = c; ul.appendChild(li);
        });
      }
      el("p", r.vlm.note, "muted");
    }
  }
  el("p", r.notice, "muted");

  // Open on the estimate's area; the alternatives are a click away in the panel.
  if (est) {
    show([est]);
  } else if (alts.length) {
    show(alts);
  }
})();
</script>
</body>
</html>
"""


def _script_json(payload: Any) -> str:
    """JSON that is safe inside a <script> element (no "</script>" or HTML comment breakout)."""
    text = json.dumps(payload, ensure_ascii=False)
    return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def render_map_html(
    result: LocateResult,
    *,
    tiles_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: str = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    ),
    title: str | None = None,
) -> str:
    payload = {
        "result": result.model_dump(mode="json", exclude={"frames"}),
        "tiles": {"url": tiles_url, "attribution": attribution},
        "maxZoom": MAX_ZOOM,
    }
    page_title = title or f"Where was this filmed? {result.video.rsplit('/', 1)[-1]}"
    return (
        _TEMPLATE.replace("__TITLE__", html.escape(page_title))
        .replace("__LV__", LEAFLET_VERSION)
        .replace("__CSS_SRI__", LEAFLET_CSS_SRI)
        .replace("__JS_SRI__", LEAFLET_JS_SRI)
        .replace("__DATA__", _script_json(payload))
    )

#!/usr/bin/env python3
"""Dependency-free structural checks for the static browser demo."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"web check failed: {message}")


def main() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    local_refs = []
    for ref in re.findall(r'(?:href|src)="([^"]+)"', html):
        if ref.startswith(("http://", "https://", "#")):
            continue
        local_refs.append(ref)
        require((WEB / ref).is_file(), f"missing local asset {ref}")

    controls = (WEB / "controls.bin").read_bytes()
    require(controls[:4] == b"TAV1", "bad control-map magic")
    version, anchors, packets, latent = struct.unpack_from("<IIII", controls, 4)
    require((version, packets, latent) == (1, 256, 128), "unexpected control-map dimensions")
    expected_size = 20 + anchors * (4 + 4 * latent + 2 * 4 * packets * latent)
    require(len(controls) == expected_size, "truncated or extended control map")

    metadata = json.loads((WEB / "controls.json").read_text(encoding="utf-8"))
    require(metadata["format"] == "tinyavatar-web-controls-v1", "bad metadata format")
    require(len(metadata["anchors"]) == anchors, "metadata/control anchor mismatch")
    require(metadata["num_packets"] == packets, "metadata/control packet mismatch")

    poster = (WEB / "poster.png").read_bytes()
    require(poster[:8] == b"\x89PNG\r\n\x1a\n", "poster is not a PNG")
    width, height = struct.unpack_from(">II", poster, 16)
    require((width, height) == (96, 96), "poster dimensions do not match model")

    model = WEB / "tinyavatar.onnx"
    require(model.stat().st_size > 1_000_000, "ONNX model is missing or suspiciously small")

    # Anti-fire invariants. The browser port intentionally has a frozen
    # anchor Jacobian, so manifold dragging must be target-seeking and bounded.
    # Regression here recreates the old pointer-event-rate latent integrator.
    app = (WEB / "app.js").read_text(encoding="utf-8")
    require("const LATENT_TRUST_RADIUS = 3.0;" in app, "missing latent trust region")
    require(
        "baseOffset: copyArray(state.latentOffset)" in app,
        "drag does not freeze its starting latent offset",
    )
    require(
        "latentTarget(state.drag.pin, state.drag.baseOffset, targetX, targetY)" in app,
        "manifold drag is not target-seeking",
    )
    require(
        "state.latentOffset[latent] += delta[latent]" not in app,
        "open latent pointer integrator returned",
    )

    print(
        f"web demo ok: {anchors} faces, {packets} packets, "
        f"{len(local_refs)} local page assets, {model.stat().st_size / 1e6:.1f} MB model, "
        "target-seeking bounded latent drag"
    )


if __name__ == "__main__":
    main()

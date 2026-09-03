#!/usr/bin/env python3
"""Export the shipped TinyAvatar checkpoint as a browser-playable ragdoll.

The web demo needs two things that are awkward to derive on a phone:

* a decoder + Gabor renderer that accepts a packet mask and direct shifts;
* the local decoder position Jacobian at a few deterministic face samples.

The Jacobian is not a new model. It is the same derivative used by
``splat_ragdoll.py``, frozen at each demo anchor so the browser can solve the
two-row damped least-squares pin update without shipping PyTorch.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from pathlib import Path

import torch
import torch.nn as nn

import splat_trainer5 as ST


class WebHead(nn.Module):
    """Decoder/renderer head with two explicit, browser-controlled edits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.dec = model.dec
        self.ren = model.ren
        self.ren.use_checkpoint = False

    def forward(
        self,
        z_latent: torch.Tensor,
        packet_mask: torch.Tensor,
        packet_shift: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.dec(z_latent)
        px, py, sigma, theta, freq, coeff = self.ren.activate(raw.float())
        px = torch.clamp(px + packet_shift[..., 0], 0.001, 0.999)
        py = torch.clamp(py + packet_shift[..., 1], 0.001, 0.999)
        packet_coeff = coeff
        coeff = packet_coeff * packet_mask[..., None, None]

        out = None
        for start in range(0, self.ren.N, self.ren.chunk):
            take = slice(start, start + self.ren.chunk)
            chunk = self.ren._chunk(
                px[:, take],
                py[:, take],
                sigma[:, take],
                theta[:, take],
                freq[:, take],
                coeff[:, take],
            )
            out = chunk if out is None else out + chunk

        active = torch.cat(
            (
                px[..., None],
                py[..., None],
                sigma[..., None],
                theta[..., None],
                freq[..., None],
                packet_coeff.reshape(*packet_coeff.shape[:2], 6),
            ),
            dim=-1,
        )
        return torch.sigmoid(out), active


def latent_for_seed(seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(128, generator=generator) * scale


def position_jacobian(model: nn.Module, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def positions(latent: torch.Tensor) -> torch.Tensor:
        raw = model.dec(latent[None])
        px, py, *_ = model.ren.activate(raw.float())
        return torch.cat((px[0], py[0]))

    jac = torch.autograd.functional.jacobian(positions, z, vectorize=True)
    count = model.ren.N
    return jac[:count].contiguous(), jac[count:].contiguous()


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def write_png(path: Path, rgb: torch.Tensor) -> None:
    """Write uint8 HxWx3 without adding an image-library dependency."""
    image = rgb.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    height, width, _ = image.shape
    rows = b"".join(b"\x00" + bytes(image[y].reshape(-1).tolist()) for y in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(rows, level=9))
    payload += png_chunk(b"IEND", b"")
    path.write_bytes(payload)


@torch.no_grad()
def render_gallery(model: nn.Module, seeds: list[int], scale: float, path: Path) -> None:
    cols = min(6, len(seeds))
    rows = math.ceil(len(seeds) / cols)
    side = model.ren.H
    gutter = 6
    gallery = torch.full((rows * side + (rows - 1) * gutter,
                          cols * side + (cols - 1) * gutter, 3), 0.04)
    for index, seed in enumerate(seeds):
        z = latent_for_seed(seed, scale)
        image = model.ren(model.dec(z[None]))[0].permute(1, 2, 0)
        row, col = divmod(index, cols)
        y = row * (side + gutter)
        x = col * (side + gutter)
        gallery[y:y + side, x:x + side] = image
    write_png(path, gallery)


@torch.no_grad()
def render_poster(model: nn.Module, seed: int, scale: float, path: Path) -> None:
    z = latent_for_seed(seed, scale)
    image = model.ren(model.dec(z[None]))[0].permute(1, 2, 0)
    write_png(path, image)


def write_controls(
    path: Path,
    anchors: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]],
    count: int,
) -> None:
    """Little-endian binary: header, then seed/z/Jx/Jy for every anchor."""
    latent = 128
    with path.open("wb") as handle:
        handle.write(b"TAV1")
        handle.write(struct.pack("<IIII", 1, len(anchors), count, latent))
        for seed, z, jx, jy in anchors:
            handle.write(struct.pack("<i", seed))
            for tensor in (z, jx, jy):
                handle.write(tensor.detach().to(torch.float32).cpu().numpy().tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model2.pt")
    parser.add_argument("--out", default="web")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument(
        "--gallery-seeds",
        default="",
        help="optional comma-separated development gallery; omitted in release builds",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, checkpoint = ST.load_splatvae(args.checkpoint)
    model.eval()
    count = int(checkpoint["num_packets"])
    size = int(checkpoint["image_size"])

    gallery_seeds = [int(value) for value in args.gallery_seeds.split(",") if value.strip()]
    if gallery_seeds:
        render_gallery(model, gallery_seeds, args.scale, out / "anchor-gallery.png")

    chosen_seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not chosen_seeds:
        raise SystemExit("--seeds must contain at least one integer")
    render_poster(model, chosen_seeds[0], args.scale, out / "poster.png")

    head = WebHead(model).eval()
    torch.onnx.export(
        head,
        (
            torch.zeros(1, 128),
            torch.ones(1, count),
            torch.zeros(1, count, 2),
        ),
        out / "tinyavatar.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["z_latent", "packet_mask", "packet_shift"],
        output_names=["rendered_image", "packet_params"],
        dynamo=False,
    )

    anchors = []
    metadata = []
    for seed in chosen_seeds:
        z = latent_for_seed(seed, args.scale)
        jx, jy = position_jacobian(model, z)
        anchors.append((seed, z, jx, jy))
        metadata.append({"seed": seed})

    write_controls(out / "controls.bin", anchors, count)

    data = {
        "format": "tinyavatar-web-controls-v1",
        "checkpoint": Path(args.checkpoint).name,
        "image_size": size,
        "num_packets": count,
        "latent_size": 128,
        "grab_radius": 0.10,
        "native_damping": 0.08,
        "browser_damping": 0.005,
        "display_rule": "immutable-anchor-relative-state",
        "anchors": metadata,
    }
    (out / "controls.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"exported {len(anchors)} anchors, {count} packets, {size}px -> {out}")


if __name__ == "__main__":
    main()

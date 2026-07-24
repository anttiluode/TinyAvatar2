#!/usr/bin/env python3
"""
sharpness_ceiling.py -- is model5_constQ actually at its ceiling?

WHERE WE ARE.  Five hypotheses are now dead by measurement:

  FS1/FS2 [K]  face is not a starved low-contrast region
  FS3  1.00     no spatial misallocation
  FS5  [K]      no spectral cliff (capture 0.999/0.993/0.979/0.942/0.861)
  FS6  [K]      in/out ratio 1.00 / 1.00 / 1.00 / 1.00 / 0.93 -- the face is
                treated IDENTICALLY to the background at every frequency
  FS7  0.988    recon pairs are as different as input pairs.  The decoder is
                NOT collapsing distinct poses onto one face.  Conditional-mean
                blur is dead too

What survives: the reconstruction is UNIFORMLY, MILDLY soft.  Captured power
0.825 in the top octave = 58% of the amplitude retained there.  That is a real
attenuation and it applies everywhere equally.

THE REMAINING QUESTION
----------------------
Is that residual softness (a) all there is, meaning the model is at the
information ceiling for a 128px canvas, or (b) accompanied by extra spatial
error the radial spectrum cannot see (misplaced edges, phase error)?

FS9 answers it.  Build a SPECTRAL TWIN of each input: the input with each
radial band attenuated by exactly the amount the model was measured to lose.
  captured = 1 - P_res/P_in ; if recon = a*input per band then a = 1-sqrt(1-c)
  c = 0.825 -> a = 0.582     c = 0.957 -> a = 0.793     c = 0.991 -> a = 0.905
Then compare the twin against the model's actual reconstruction.

  MSE(twin, recon) << MSE(recon, input)  =>  the model's ONLY deficit is the
      band attenuation we already measured.  Nothing is misplaced.  It is at
      its ceiling for this canvas, and the fix is more pixels on the face.
  MSE(twin, recon) ~ MSE(recon, input)   =>  something is going wrong in SPACE
      that the radial spectrum averaged away.  Keep looking.

FS10 is the plain-sight check nobody has run: how many pixels IS the face?
A 50px-wide face has a ~1px eyelid line.  No basis, latent or loss fixes that.

GATES (pre-registered)
  FS9   MSE(twin, recon) < 0.30 * MSE(recon, input)
  FS10  diagnostic -- face width in px, and the linear gain a tight crop buys

USAGE
  python3.13 sharpness_ceiling.py --selftest
  python3.13 sharpness_ceiling.py --data faces1 --n 64 \
        --model runs/tiny2/model5_constQ.pt
"""

import argparse
import sys

import numpy as np

from face_budget import (_load_gray, list_images, motion_mask, close, erode,
                         radial_power, _load_model, _forward_recon, box_mask)


def band_gain_filter(img, edges, gains):
    """Multiply each radial band of img's spectrum by its gain."""
    H, W = img.shape
    m = img.mean()
    F = np.fft.fftshift(np.fft.fft2(img - m))
    fy = np.fft.fftshift(np.fft.fftfreq(H)) * H
    fx = np.fft.fftshift(np.fft.fftfreq(W)) * W
    R = np.hypot(*np.meshgrid(fx, fy, indexing="xy"))
    G = np.ones_like(R)
    for (lo, hi), g in zip(zip(edges[:-1], edges[1:]), gains):
        G[(R >= lo) & (R < hi)] = g
    G[R >= edges[-1]] = gains[-1]
    return np.fft.ifft2(np.fft.ifftshift(F * G)).real + m


def measure_capture(inputs, recons, edges):
    nb = min(inputs.shape[1:]) // 2
    Pin = np.zeros(nb)
    Pre = np.zeros(nb)
    for t in range(len(inputs)):
        Pin += radial_power(inputs[t])
        Pre += radial_power(recons[t] - inputs[t])
    caps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a, b = int(round(lo)), min(int(round(hi)), nb)
        if b <= a:
            caps.append(1.0)
            continue
        caps.append(1.0 - Pre[a:b].sum() / max(Pin[a:b].sum(), 1e-30))
    return caps


def fs9(inputs, recons, edges):
    caps = measure_capture(inputs, recons, edges)
    gains = [max(0.0, 1.0 - np.sqrt(max(0.0, 1.0 - c))) for c in caps]
    print("\n  FS9  spectral twin -- input attenuated by the model's own losses")
    print("       band            captured   implied amplitude gain")
    for (lo, hi), c, g in zip(zip(edges[:-1], edges[1:]), caps, gains):
        print(f"       {lo:5.1f} - {hi:5.1f}    {c:6.3f}          {g:6.3f}")
    twins = np.stack([band_gain_filter(x, edges, gains) for x in inputs])
    mse_model = float(((recons - inputs) ** 2).mean())
    mse_twin = float(((twins - recons) ** 2).mean())
    ratio = mse_twin / max(mse_model, 1e-30)
    print(f"\n       MSE(recon, input)  {mse_model:.6f}   <- what the model loses")
    print(f"       MSE(twin,  recon)  {mse_twin:.6f}   <- what is left over")
    print(f"    FS9 ratio {ratio:.3f} < 0.30   [{'V' if ratio < 0.30 else 'K'}]")
    print("        [V] => the softness IS the whole story.  Nothing is")
    print("        misplaced in space; the model is at its information ceiling")
    print("        for this canvas.  More pixels on the face is the only fix.")
    print("        [K] => there is spatial error the radial spectrum hides.")
    return ratio < 0.30, twins


def fs10(lum, mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        print("\n  FS10 skipped: empty mask")
        return
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    size = lum.shape[1]
    print(f"\n  FS10 how many pixels is the face?")
    print(f"       moving-region bounding box: {w} x {h} px on a {size}px canvas")
    print(f"       face occupies {w/size*100:.0f}% of the frame width")
    for name, frac in (("eye width ~1/5 face", 0.20),
                       ("iris ~1/12 face", 0.083),
                       ("eyelid line ~1/50 face", 0.02)):
        print(f"       {name:26s} {w*frac:5.1f} px")
    tight = 0.85 * size
    print(f"       a tight crop would make the face ~{tight:.0f}px wide"
          f"  =  {tight/w:.1f}x linear")
    print("       Detail below ~2px cannot exist in the data, so no basis,")
    print("       latent or loss can put it in the render.")


def run(args):
    files = list_images(args.data, args.n)
    size = args.size
    lum = np.stack([_load_gray(f, size) for f in files])
    if args.box:
        b = tuple(int(v) for v in args.box.split(","))
        mask = box_mask(b, size)
    else:
        mask, _ = motion_mask(lum, args.motion_pct, args.motion_close)
    print(f"loaded {len(files)} frames at {size}px; "
          f"moving region {mask.mean()*100:.1f}% of canvas")

    fs10(lum, mask)

    if not args.model:
        print("\n  no --model, stopping after FS10")
        return

    import torch
    from PIL import Image
    model = _load_model(args.model)
    model.train(False)

    def load_rgb(p):
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0

    rgb = np.stack([load_rgb(f) for f in files])
    x = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).float()
    with torch.no_grad():
        outs = [_forward_recon(model, x[i:i + args.batch])
                for i in range(0, len(x), args.batch)]
        r = torch.cat(outs).clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    lrec = 0.299 * r[:, :, :, 0] + 0.587 * r[:, :, :, 1] + 0.114 * r[:, :, :, 2]

    edges = [args.f_lo * (args.f_max / args.f_lo) ** (i / args.octaves)
             for i in range(args.octaves + 1)]
    ok, twins = fs9(lum, lrec, edges)

    # the strip that settles it by eye
    k = min(args.strips, len(files))
    idx = np.linspace(0, len(files) - 1, k).round().astype(int)
    rows = []
    for i in idx:
        rows.append(np.concatenate(
            [lum[i], lrec[i], np.clip(twins[i], 0, 1)], axis=1))
    strip = np.clip(np.concatenate(rows, axis=0), 0, 1)
    Image.fromarray((strip * 255).astype(np.uint8)).save("ceiling_strip.png")
    print("\n  wrote ceiling_strip.png  (input | model recon | spectral twin)")
    print("  If columns 2 and 3 are indistinguishable, the model is doing")
    print("  exactly and only what the measured band attenuation says.")


def run_selftest(args):
    from face_budget import gaussian_blur
    ok = True
    rng = np.random.default_rng(0)
    size = 64
    edges = [1.0 * 16.0 ** (i / 4) for i in range(5)]
    base = np.stack([gaussian_blur(rng.standard_normal((size, size)), 0.7) * 0.15
                     + 0.5 for _ in range(10)])

    print("SC1  band_gain_filter with all gains 1.0 is the identity")
    out = band_gain_filter(base[0], edges, [1.0] * 4)
    d = float(np.abs(out - base[0]).max())
    print(f"     max |diff| {d:.2e}  (<1e-9)  [{'V' if d < 1e-9 else 'K'}]")
    ok &= d < 1e-9

    print("\nSC2  POSITIVE -- a purely spectral loss (blur) must trip FS9")
    rec = np.stack([gaussian_blur(b, 1.6) for b in base])
    r2, _ = fs9(base, rec, edges)
    print(f"     -> must be V  [{'V' if r2 else 'K'}]")
    ok &= bool(r2)

    print("\nSC3  NEGATIVE -- a SPATIAL corruption must NOT trip FS9")
    rec = base.copy()
    m = np.zeros((size, size), dtype=bool)
    m[20:44, 20:44] = True
    for t in range(len(base)):
        # shift the patch by 2px: same spectrum family, wrong place
        rec[t][m] = np.roll(np.roll(base[t], 2, axis=0), 2, axis=1)[m]
    r3, _ = fs9(base, rec, edges)
    print(f"     -> must be K  [{'V' if not r3 else 'K'}]")
    ok &= (not r3)

    print(f"\nSELFTEST {'ALL [V]' if ok else '[K] -- do not trust the metrics'}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--data", default="faces1")
    p.add_argument("--model", default="")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--box", default="")
    p.add_argument("--motion_pct", type=float, default=85.0)
    p.add_argument("--motion_close", type=int, default=6)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--strips", type=int, default=6)
    p.add_argument("--octaves", type=int, default=5)
    p.add_argument("--f_lo", type=float, default=1.0)
    p.add_argument("--f_max", type=float, default=32.0)
    a = p.parse_args()
    if a.selftest:
        return run_selftest(a)
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
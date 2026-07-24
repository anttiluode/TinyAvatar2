#!/usr/bin/env python3
"""
face_budget.py -- TinyAvatar2 diagnostic: does the face get any of the loss?

THE QUESTION
------------
model5_constQ renders the room sharp (curtain folds, picture frame, table edge)
and the head as a dark featureless mass -- in a DIRECT reconstruction of a
training frame, pins 0, |z-z0| 0.00.  So the fine-octave packets are alive and
carrying amplitude.  They are not dead; they are somewhere else.

Hypothesis under test: the face is a small, LOW-CONTRAST minority of the image,
so it holds almost none of the available L2 signal, so the optimizer correctly
spends its budget on the room.

The number that decides this is NOT the residual error (an unmodelled face has
high residual by definition -- that is circular).  It is the AVAILABLE signal:
how much error could ever have been removed there.  That is what sets gradient
pressure.  --stats measures it and needs no model at all.

MODES
-----
  --selftest   numpy only.  Two-sided check of the metric code on synthetic
               data (planted low-contrast patch must trip the gate; planted
               high-contrast patch must not).  Run this first.

  --stats      numpy only, no torch, no checkpoint.  THE DECISIVE TEST.
               Gates FS1 (available-signal share vs area share) and
               FS2 (contrast deficit).

  --recon      needs torch + splat_trainer5 + the checkpoint.  Where the
               residual actually sits (FS3), input|recon|error strips so you
               can confirm preprocessing matches training, and a finite-
               difference SVD of the decoder Jacobian over the face box to
               count how many latent directions move the face (FS4).

REGISTERED GATES  (thresholds set BEFORE seeing any of his data -- they are
                   pre-registered guesses, not tuned)
  FS1  detail_share < 0.5 * area_share            [V] => face is a low-contrast
                                                        minority of the signal
  FS2  mean|grad| inside < 0.6 * outside          [V] => contrast deficit
  FS3  residual_share > 2.0 * detail_share        [V] => face starved beyond
                                                        even its small stake
  FS4  effective rank of dRender_face/dz          diagnostic, no threshold

FS1 and FS2 firing together means: re-prep with a tighter crop and light your
face.  FS1 NOT firing means my diagnosis is wrong and you should say so.

USAGE
-----
  python3.13 face_budget.py --selftest
  python3.13 face_budget.py --stats --data faces1 --n 300
  python3.13 face_budget.py --recon --data faces1 --n 64 \
        --model runs/tiny2/model5_constQ.pt

REGION DEFINITION
-----------------
Two regions are reported side by side, always:
  haar    median Haar face box over the dataset (needs the cascade xml; your
          Windows Store python was missing it before -- if it fails, --box
          x0,y0,x1,y1 in pixels, or just rely on the second region)
  motion  per-pixel temporal variance over the dataset, thresholded at the
          --motion_pct percentile.  Needs no cascade file, and is arguably the
          more honest mask: it IS the region the latent has to explain.

Everything is computed on luminance at the working resolution (--size, default
128 to match the model).  Detail energy is reported at three blur scales so the
conclusion cannot be a sigma artefact.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# image io  (PIL preferred, cv2 fallback, neither needed for --selftest)
# ----------------------------------------------------------------------------

def _load_gray(path, size):
    """Load an image as float64 luminance in [0,1], resized to size x size."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float64) / 255.0
    except ImportError:
        import cv2
        a = cv2.imread(path, cv2.IMREAD_COLOR)
        if a is None:
            raise IOError(path)
        a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
        a = a[:, :, ::-1].astype(np.float64) / 255.0
    # Rec.601 luma
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def list_images(folder, n):
    pats = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for p in pats:
        files.extend(glob.glob(os.path.join(folder, p)))
    files.sort()
    if not files:
        raise SystemExit(f"no images found in {folder!r}")
    if n and n < len(files):
        idx = np.linspace(0, len(files) - 1, n).round().astype(int)
        files = [files[i] for i in idx]
    return files


# ----------------------------------------------------------------------------
# metrics  (pure numpy -- these are what --selftest verifies)
# ----------------------------------------------------------------------------

def gaussian_blur(img, sigma):
    """Separable gaussian blur, reflect padding.  sigma in pixels."""
    if sigma <= 0:
        return img.copy()
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, ((0, 0), (r, r)), mode="reflect")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, pad)
    pad = np.pad(out, ((r, r), (0, 0)), mode="reflect")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, pad)
    return out


def detail_energy(img, sigma):
    """Per-pixel squared high-pass energy: the error a model must EARN.

    Low octaves give the blurred version away almost for free; what costs
    packets is img - blur(img).  Summing this over a region is the ceiling on
    how much L2 that region can ever contribute.
    """
    d = img - gaussian_blur(img, sigma)
    return d * d


def grad_mag(img):
    gy, gx = np.gradient(img)
    return np.hypot(gx, gy)


def _erode_axis(m, r, axis):
    out = m.copy()
    for d in range(1, r + 1):
        for s in (d, -d):
            sh = np.zeros_like(m)
            if axis == 0:
                if s > 0:
                    sh[s:, :] = m[:-s, :]
                else:
                    sh[:s, :] = m[-s:, :]
            else:
                if s > 0:
                    sh[:, s:] = m[:, :-s]
                else:
                    sh[:, :s] = m[:, -s:]
            out &= sh
    return out


def erode(mask, r):
    """Binary erosion with a square structuring element, separable."""
    if r <= 0:
        return mask.copy()
    return _erode_axis(_erode_axis(mask, r, 0), r, 1)


def dilate(mask, r):
    if r <= 0:
        return mask.copy()
    return ~erode(~mask, r)


def close(mask, r):
    """Fill the interior of a rim-like mask: dilate then erode."""
    return erode(dilate(mask, r), r)


def region_share(field, mask):
    """Fraction of a non-negative field that lies inside mask."""
    tot = float(field.sum())
    if tot <= 0:
        return float("nan")
    return float(field[mask].sum()) / tot


def summarise(img_stack, mask, sigmas=(1.0, 2.0, 4.0)):
    """img_stack: (T,H,W) luminance.  mask: (H,W) bool.  Returns a dict."""
    T, H, W = img_stack.shape
    area_share = float(mask.sum()) / float(H * W)

    det_shares = {}
    for s in sigmas:
        acc_in = 0.0
        acc_all = 0.0
        for t in range(T):
            e = detail_energy(img_stack[t], s)
            acc_in += float(e[mask].sum())
            acc_all += float(e.sum())
        det_shares[s] = acc_in / acc_all if acc_all > 0 else float("nan")

    g_in = 0.0
    g_out = 0.0
    for t in range(T):
        g = grad_mag(img_stack[t])
        g_in += float(g[mask].mean())
        g_out += float(g[~mask].mean())
    g_in /= T
    g_out /= T

    flat_in = img_stack[:, mask].ravel()
    flat_out = img_stack[:, ~mask].ravel()

    return {
        "area_share": area_share,
        "detail_share": det_shares,
        "grad_in": g_in,
        "grad_out": g_out,
        "grad_ratio": g_in / g_out if g_out > 0 else float("nan"),
        "p5_in": float(np.percentile(flat_in, 5)),
        "p95_in": float(np.percentile(flat_in, 95)),
        "p5_out": float(np.percentile(flat_out, 5)),
        "p95_out": float(np.percentile(flat_out, 95)),
        "mean_in": float(flat_in.mean()),
        "mean_out": float(flat_out.mean()),
    }


# ----------------------------------------------------------------------------
# region masks
# ----------------------------------------------------------------------------

def haar_box(files, size):
    """Median Haar face box over the dataset.  Returns (x0,y0,x1,y1) or None."""
    try:
        import cv2
    except ImportError:
        print("  [haar] cv2 not importable -- skipping haar region")
        return None
    xml = os.path.join(getattr(cv2, "data", None).haarcascades,
                       "haarcascade_frontalface_default.xml") \
        if getattr(cv2, "data", None) else ""
    if not xml or not os.path.exists(xml):
        print("  [haar] cascade xml not found (the Windows Store python issue)")
        print("         -> use --box x0,y0,x1,y1, or rely on the motion region")
        return None
    cas = cv2.CascadeClassifier(xml)
    boxes = []
    for f in files:
        g = (_load_gray(f, size) * 255).astype(np.uint8)
        det = cas.detectMultiScale(g, 1.1, 4)
        if len(det):
            x, y, w, h = max(det, key=lambda b: b[2] * b[3])
            boxes.append([x, y, x + w, y + h])
    if not boxes:
        print("  [haar] no faces detected in any frame")
        return None
    b = np.median(np.array(boxes, dtype=np.float64), axis=0).round().astype(int)
    print(f"  [haar] {len(boxes)}/{len(files)} frames detected, median box {tuple(b)}")
    return tuple(int(v) for v in b)


def box_mask(box, size):
    x0, y0, x1, y1 = box
    m = np.zeros((size, size), dtype=bool)
    m[max(0, y0):min(size, y1), max(0, x0):min(size, x1)] = True
    return m


def motion_mask(img_stack, pct, close_r=0):
    """Pixels whose temporal variance is above the pct-th percentile.

    Raw thresholded variance is a RIM detector -- a moving head puts its
    variance on the silhouette, not on the cheek.  close_r>0 morphologically
    closes it into a filled region so there is an interior left to erode.
    """
    var = img_stack.var(axis=0)
    thr = np.percentile(var, pct)
    m = var > thr
    if close_r > 0:
        m = close(m, close_r)
    return m, var


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------

def report(name, s):
    a = s["area_share"]
    print(f"\n  region: {name}")
    print(f"    area share of canvas          {a*100:6.2f} %")
    for sig, v in sorted(s["detail_share"].items()):
        conc = v / a if a > 0 else float("nan")
        print(f"    detail-energy share (s={sig:.0f}px) {v*100:6.2f} %"
              f"   concentration {conc:5.2f}x area")
    print(f"    mean |grad|  inside {s['grad_in']:.5f}   outside {s['grad_out']:.5f}"
          f"   ratio {s['grad_ratio']:.3f}")
    print(f"    luminance    inside  p5 {s['p5_in']:.3f}  p95 {s['p95_in']:.3f}"
          f"  mean {s['mean_in']:.3f}")
    print(f"                 outside p5 {s['p5_out']:.3f}  p95 {s['p95_out']:.3f}"
          f"  mean {s['mean_out']:.3f}")


def gates(s, sigma_gate=2.0):
    a = s["area_share"]
    d = s["detail_share"][sigma_gate]
    fs1 = d < 0.5 * a
    fs2 = s["grad_ratio"] < 0.6
    print(f"    FS1 detail_share {d*100:.2f}% < 0.5*area {0.5*a*100:.2f}%"
          f"   [{'V' if fs1 else 'K'}]")
    print(f"    FS2 grad ratio   {s['grad_ratio']:.3f} < 0.60"
          f"                 [{'V' if fs2 else 'K'}]")
    return fs1, fs2


def analyse(stack, mask, name, erode_r, sigmas=(1.0, 2.0, 4.0)):
    """Report the full region AND its eroded interior, gate on the interior.

    Why: a head silhouetted against a bright window is a huge high-contrast
    edge, and that edge sits inside any face box.  Measured on the selftest
    synthetic, 99.85% of a face box's detail energy was in the boundary rim
    and 0.15% in the interior.  A box-level number is therefore a measurement
    of the outline, not of the eyes, nose and lips -- which are exactly the
    things that are missing in the render.  Erode, then gate.
    """
    inner = erode(mask, erode_r)
    if inner.sum() < 50:
        print(f"\n  region {name}: too small to erode by {erode_r}px, "
              f"reporting full only")
        s = summarise(stack, mask, sigmas)
        report(name, s)
        return gates(s)

    s_full = summarise(stack, mask, sigmas)
    report(name + "  [full, includes silhouette]", s_full)
    s_in = summarise(stack, inner, sigmas)
    report(name + f"  [interior, eroded {erode_r}px]  <-- GATED", s_in)
    rim_frac = 1.0 - (s_in["detail_share"][2.0] / max(s_full["detail_share"][2.0],
                                                      1e-12))
    print(f"    silhouette carries {rim_frac*100:5.1f}% of this region's "
          f"detail energy")
    return gates(s_in)


# ----------------------------------------------------------------------------
# --stats
# ----------------------------------------------------------------------------

def run_stats(args):
    files = list_images(args.data, args.n)
    print(f"loading {len(files)} images at {args.size}px from {args.data}")
    stack = np.stack([_load_gray(f, args.size) for f in files])
    print(f"stack {stack.shape}  global mean {stack.mean():.3f}")

    regions = []
    if args.box:
        b = tuple(int(v) for v in args.box.split(","))
        regions.append(("box (manual)", box_mask(b, args.size)))
    else:
        b = haar_box(files, args.size)
        if b is not None:
            regions.append(("haar face box", box_mask(b, args.size)))

    mm, var = motion_mask(stack, args.motion_pct, args.motion_close)
    regions.append((f"motion (var > p{args.motion_pct:g})", mm))

    verdicts = {}
    for name, mask in regions:
        verdicts[name] = analyse(stack, mask, name, args.erode)

    if args.save_masks:
        try:
            from PIL import Image
            v = (var - var.min()) / max(1e-12, float(np.ptp(var)))
            Image.fromarray((v * 255).astype(np.uint8)).save("var_map.png")
            for name, mask in regions:
                fn = "mask_" + name.split()[0] + ".png"
                Image.fromarray((mask * 255).astype(np.uint8)).save(fn)
            print("\n  wrote var_map.png and mask_*.png -- LOOK AT THEM before "
                  "believing any of the above")
        except ImportError:
            print("\n  (PIL not available, skipped mask dump)")

    print("\n  reading: FS1+FS2 both [V] => the face is a small, low-contrast")
    print("  minority of the signal and the optimizer is behaving correctly.")
    print("  Fix is a tighter crop and a lamp, not the basis.")
    print("  FS1 [K] => diagnosis wrong, the face has its fair share, look")
    print("  elsewhere (latent rank, loss form).")
    return verdicts


# ----------------------------------------------------------------------------
# --recon   (needs torch)
# ----------------------------------------------------------------------------

def _import_trainer():
    import importlib
    for name in ("splat_trainer5", "splat_trainer4q", "splat_trainer3v2"):
        try:
            return importlib.import_module(name), name
        except ImportError:
            continue
    raise SystemExit("could not import splat_trainer5 / 4q / 3v2 -- run this "
                     "from the folder that has them")


def _load_model(path):
    import torch
    ST, nm = _import_trainer()
    print(f"  trainer module: {nm}")
    fn = getattr(ST, "load_splatvae", None)
    if fn is None:
        raise SystemExit(f"{nm} has no load_splatvae -- adapt _load_model()")
    last = None
    for arg in (path, torch.load(path, map_location="cpu")):
        try:
            m = fn(arg)
            break
        except Exception as e:          # noqa: BLE001 - deliberately tolerant
            last = e
            m = None
    if m is None:
        raise SystemExit(f"load_splatvae rejected both path and dict: {last}")
    if isinstance(m, (tuple, list)):
        m = m[0]
    m.eval()
    return m


def _forward_recon(model, x):
    """x: (B,3,H,W) in [0,1]. Returns recon (B,3,H,W)."""
    # 1. Check if model has a native forward
    if hasattr(model, "forward") and callable(getattr(model, "forward")):
        try:
            out = model(x)
            while isinstance(out, (tuple, list)):
                out = out[0]
            return out
        except NotImplementedError:
            pass

    # 2. SplatVAEQ / Trainer route: enc -> dec -> ren
    if hasattr(model, "enc") and hasattr(model, "dec") and hasattr(model, "ren"):
        mu, _ = model.enc(x)
        raw = model.dec(mu)
        px, py, sg, th, fr, cf = model.ren.activate(raw.float())
        # Use render_from_params or model.ren(raw)
        out = model.ren(raw)
        while isinstance(out, (tuple, list)):
            out = out[0]
            
        # Ensure sigmoid if output is raw logits
        if out.min() < 0.0 or out.max() > 1.0:
            out = torch.sigmoid(out)
        return out

    raise AttributeError("Could not determine reconstruction pathway for model.")


def _find_decoder(model):
    """Probe for a z -> image callable.  Returns fn or None."""
    for name in ("decode", "dec", "decoder", "render", "generate"):
        fn = getattr(model, name, None)
        if callable(fn):
            return name, fn
    return None, None


def run_recon(args):
    import torch

    files = list_images(args.data, args.n)
    size = args.size
    model = _load_model(args.model)

    # rebuild colour stack for the model, luminance stack for the masks
    def load_rgb(p):
        try:
            from PIL import Image
            im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32) / 255.0
        except ImportError:
            import cv2
            a = cv2.imread(p, cv2.IMREAD_COLOR)
            a = cv2.resize(a, (size, size))
            return (a[:, :, ::-1].astype(np.float32) / 255.0)

    rgb = np.stack([load_rgb(f) for f in files])              # (T,H,W,3)
    lum = np.stack([_load_gray(f, size) for f in files])      # (T,H,W)

    if args.box:
        b = tuple(int(v) for v in args.box.split(","))
        mask = box_mask(b, size)
        mname = "box (manual)"
    else:
        b = haar_box(files, size)
        if b is not None:
            mask, mname = box_mask(b, size), "haar face box"
        else:
            mask, _ = motion_mask(lum, args.motion_pct, args.motion_close)
            mname = f"motion (var > p{args.motion_pct:g})"
    print(f"  region for FS3/FS4: {mname}  ({mask.sum()/mask.size*100:.1f}% of canvas)")

    x = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).float()

    # BatchNorm caveat from his own ledger: eval-mode running stats depressed
    # PSNR ~14x in MSE at 3000 steps.  Report both modes, never just one.
    modes = ["eval", "batch"] if args.bn == "both" else [args.bn]
    resid_share = {}
    recons = {}
    for mode in modes:
        model.train(mode == "batch")
        with torch.no_grad():
            outs = []
            for i in range(0, len(x), args.batch):
                outs.append(_forward_recon(model, x[i:i + args.batch]))
            r = torch.cat(outs).clamp(0, 1).numpy().transpose(0, 2, 3, 1)
        recons[mode] = r
        err = ((r - rgb) ** 2).mean(axis=3)                    # (T,H,W)
        tot = float(err.sum())
        inside = float(err[:, mask].sum())
        resid_share[mode] = inside / tot
        mse_in = float(err[:, mask].mean())
        mse_out = float(err[:, ~mask].mean())
        print(f"\n  BN mode = {mode}")
        print(f"    overall MSE            {err.mean():.6f}"
              f"   (PSNR {10*math.log10(1.0/max(err.mean(),1e-12)):.2f} dB)")
        print(f"    MSE inside region      {mse_in:.6f}")
        print(f"    MSE outside region     {mse_out:.6f}")
        print(f"    residual share inside  {resid_share[mode]*100:6.2f} %")
    model.train(False)

    # FS3 needs the detail share from the same region
    inner = erode(mask, args.erode)
    if inner.sum() < 50:
        inner = mask
    s = summarise(lum, inner, sigmas=(2.0,))
    d = s["detail_share"][2.0]
    ref = resid_share.get("eval", list(resid_share.values())[0])
    print(f"\n    detail share (s=2px)   {d*100:6.2f} %")
    print(f"    FS3 residual {ref*100:.2f}% > 2.0 * detail {2*d*100:.2f}%"
          f"   [{'V' if ref > 2.0*d else 'K'}]")

    # strips: input | recon | error, so preprocessing can be eyeballed
    try:
        from PIL import Image
        k = min(args.strips, len(files))
        idx = np.linspace(0, len(files) - 1, k).round().astype(int)
        mode0 = modes[0]
        rows = []
        for i in idx:
            e = np.abs(recons[mode0][i] - rgb[i]).mean(axis=2)
            e = np.repeat((e / max(e.max(), 1e-9))[:, :, None], 3, axis=2)
            rows.append(np.concatenate([rgb[i], recons[mode0][i], e], axis=1))
        strip = np.concatenate(rows, axis=0)
        Image.fromarray((strip * 255).astype(np.uint8)).save("recon_strip.png")
        print(f"\n  wrote recon_strip.png  (input | recon | |error|, BN={mode0})")
        print("  CHECK: is the recon FRAMED the same as the input?  If this")
        print("  loader resizes but training face-cropped, the encode is")
        print("  off-distribution and FS3 is not interpretable.")
    except ImportError:
        print("\n  (PIL not available, skipped recon_strip.png)")

    # FS4 -- how many latent directions move the face?
    dname, dec = _find_decoder(model)
    if dec is None:
        print("\n  FS4 skipped: no decode/dec/decoder/render attribute found on"
              " the model.")
        print("  Point _find_decoder() at whatever your z -> image entry is,"
              " or read the")
        print("  singular spectrum pin_driver.py already prints at startup.")
        return
    print(f"\n  FS4: finite-difference Jacobian via model.{dname}()")
    try:
        with torch.no_grad():
            zdim = args.zdim
            z0 = torch.zeros(1, zdim)
            base = dec(z0)
            while isinstance(base, (tuple, list)):
                base = base[0]
            base = base.squeeze(0).numpy()
            mflat = mask.ravel()
            cols = []
            eps = args.eps
            for j in range(zdim):
                zz = z0.clone()
                zz[0, j] += eps
                out = dec(zz)
                while isinstance(out, (tuple, list)):
                    out = out[0]
                d_ = (out.squeeze(0).numpy() - base) / eps
                d_ = d_.mean(axis=0).ravel()[mflat]     # face pixels only
                cols.append(d_)
            J = np.stack(cols, axis=1)                  # (npix, zdim)
        sv = np.linalg.svd(J, compute_uv=False)
        e = sv ** 2
        e = e / e.sum()
        eff = float(np.exp(-(e * np.log(e + 1e-300)).sum()))   # participation
        c = np.cumsum(e)
        n90 = int(np.searchsorted(c, 0.90) + 1)
        print(f"    singular values (top 12): "
              + " ".join(f"{v:.3f}" for v in sv[:12]))
        print(f"    dims to reach 90% of face-motion energy: {n90} / {zdim}")
        print(f"    participation ratio (effective rank):    {eff:.1f}")
        print("    reading: a cliff after ~5 dims => the decoder listens to")
        print("    almost none of z over the face, and beta is real.  A broad")
        print("    spectrum => beta is a distraction, it is the loss budget.")
    except Exception as ex:            # noqa: BLE001
        print(f"    FS4 failed ({type(ex).__name__}: {ex}) -- adapt the decoder"
              " call, or read pin_driver's startup print instead.")


# ----------------------------------------------------------------------------
# --selftest   (two-sided: the gate must fire on a planted low-contrast face
#               and must NOT fire on a planted high-contrast one)
# ----------------------------------------------------------------------------

def _synth(size, face_amp, bg_amp, seed=0):
    rng = np.random.default_rng(seed)
    T, H = 24, size
    box = (int(0.34 * H), int(0.20 * H), int(0.66 * H), int(0.62 * H))
    stack = np.zeros((T, H, H))
    # the room is STATIC across frames -- that is the whole situation
    room = np.clip(gaussian_blur(0.5 + bg_amp * rng.standard_normal((H, H)), 0.8),
                   0, 1)
    for t in range(T):
        img = room.copy()
        # low/high contrast "face" patch, dark, with its own texture, moving
        fx = box[0] + int(round(2 * math.sin(2 * math.pi * t / T)))
        f = 0.5 - 0.28 + face_amp * rng.standard_normal((box[3] - box[1],
                                                         box[2] - box[0]))
        f = gaussian_blur(f, 0.8)
        img[box[1]:box[3], fx:fx + (box[2] - box[0])] = np.clip(f, 0, 1)
        stack[t] = img
    return stack, box


def run_selftest(args):
    size = 128
    ok = True

    print("ST0  gaussian_blur preserves DC")
    a = np.random.default_rng(1).random((64, 64))
    b = gaussian_blur(a, 3.0)
    d = abs(a.mean() - b.mean())
    print(f"     |mean(a)-mean(blur(a))| = {d:.2e}  (<1e-3)  "
          f"[{'V' if d < 1e-3 else 'K'}]")
    ok &= d < 1e-3

    print("\nST1  blur kills high-pass energy, keeps low-pass")
    e_sharp = detail_energy(a, 2.0).sum()
    e_smooth = detail_energy(gaussian_blur(a, 6.0), 2.0).sum()
    r = e_smooth / e_sharp
    print(f"     detail(blurred)/detail(sharp) = {r:.4f}  (<0.1)  "
          f"[{'V' if r < 0.1 else 'K'}]")
    ok &= r < 0.1

    print("\nST2  POSITIVE control -- planted LOW-contrast face must trip FS1/FS2")
    st, box = _synth(size, face_amp=0.008, bg_amp=0.16, seed=2)
    m = box_mask(box, size)
    f1, f2 = analyse(st, m, "planted low-contrast face", 8, sigmas=(2.0,))
    print(f"     -> FS1 {'V' if f1 else 'K'}  FS2 {'V' if f2 else 'K'}"
          f"   (both must be V)  [{'V' if (f1 and f2) else 'K'}]")
    ok &= (f1 and f2)

    print("\nST3  NEGATIVE control -- planted HIGH-contrast face must NOT trip FS1")
    st, box = _synth(size, face_amp=0.30, bg_amp=0.04, seed=3)
    m = box_mask(box, size)
    f1, f2 = analyse(st, m, "planted high-contrast face", 8, sigmas=(2.0,))
    print(f"     -> FS1 must be K: {'K (good)' if not f1 else 'V (BAD - gate is'
          ' not a real test)'}  [{'V' if not f1 else 'K'}]")
    ok &= (not f1)

    print("\nST4  motion mask finds the moving patch")
    st, box = _synth(size, face_amp=0.10, bg_amp=0.16, seed=4)
    mm, var = motion_mask(st, 85.0)
    inside = box_mask(box, size)
    # motion mask should overlap the patch far more than chance
    hit = mm[inside].mean() / max(mm.mean(), 1e-9)
    print(f"     motion-mask density inside patch / overall = {hit:.2f}  (>1.5)"
          f"  [{'V' if hit > 1.5 else 'K'}]")
    ok &= hit > 1.5

    print(f"\nSELFTEST {'ALL [V]' if ok else '[K] -- do not trust the metrics'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--recon", action="store_true")
    p.add_argument("--data", default="faces1")
    p.add_argument("--model", default="runs/tiny2/model5_constQ.pt")
    p.add_argument("--n", type=int, default=300, help="frames to sample")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--box", default="", help="x0,y0,x1,y1 in pixels, overrides haar")
    p.add_argument("--motion_pct", type=float, default=85.0)
    p.add_argument("--motion_close", type=int, default=6,
                   help="morphological closing radius for the motion mask; raw\n                        thresholded variance is a rim, not a region")
    p.add_argument("--erode", type=int, default=8,
                   help="px to erode the region by, to strip the silhouette "
                        "rim off the interior (the gated quantity)")
    p.add_argument("--save_masks", action="store_true", default=True)
    p.add_argument("--bn", choices=["eval", "batch", "both"], default="both")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--strips", type=int, default=6)
    p.add_argument("--zdim", type=int, default=128)
    p.add_argument("--eps", type=float, default=0.05)
    args = p.parse_args()

    if args.selftest:
        return run_selftest(args)
    if args.stats:
        run_stats(args)
        return 0
    if args.recon:
        run_recon(args)
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
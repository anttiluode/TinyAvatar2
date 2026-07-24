#!/usr/bin/env python3
"""
face_dof.py -- how many degrees of freedom does the face actually have?

WHERE WE ARE
------------
Three hypotheses have now been killed by measurement:

  FS1/FS2 [K]  the face is NOT a low-contrast starved region.  Interior
               detail concentration 1.97x area, gradient ratio 1.619, mean
               luminance 0.488 vs 0.403 outside.  It is brighter and busier
               than the room.
  FS3   ratio  1.00.  Residual share 40.88% vs detail share 40.68% on the
               SAME mask.  Zero spatial misallocation.
  FS5   [K]    capture per octave 0.999 / 0.993 / 0.979 / 0.942 / 0.861.
               No spectral cliff.  The model is not band-limited.

And yet: 28.3 dB PSNR, shirt fabric rendered with visible pattern, face
rendered as a smooth mass.

WHY ALL THREE MISSED IT
-----------------------
FS3 is spatial, marginalised over frequency.  FS5 is spectral, marginalised
over space.  A failure that lives *only* in the interaction -- high frequencies
missing specifically over the face, while the background's high frequencies are
fine -- leaves both marginals looking healthy.  That is the joint distribution
neither test looks at.  FS6 below looks at it.

THE OTHER LEAD
--------------
FS4 reported effective rank 1.4 over the face, 2 dims of 128 to reach 90%.
That number is NOT yet trustworthy: v2 evaluated the Jacobian at z0 = 0, the
origin of the prior.  Your own RG4 ramp measured |z| ~ 9.92-10.57 for real
encodings, and sqrt(128) = 11.3.  So real codes live on a shell of radius ~10
and the Jacobian was probed ~10 units away from anywhere the model has ever
been.  FS4b re-measures it at actual encoded latents.

Do not retrain on the strength of a Jacobian taken at a point the model never
visits.  That is a 7-hour run on an unverified premise.

THE TEST THAT SETTLES BETA
--------------------------
Low model rank has two very different causes:
  (a) the model is throwing information away  (beta, encoder, decoder capacity)
  (b) the DATA only has that many DOF          ("me moving a little bit and
      opening my mouth" is maybe 2-3 modes)
If (b), rank 2 is correct behaviour and lowering beta buys nothing.
FS8 measures the data's own rank over the face region and FS4b measures the
model's.  The comparison is the answer.

GATES  (thresholds pre-registered, before seeing his numbers)
  FS6   capture(top octave, inside) < 0.6 * capture(top octave, outside)
        [V] => high frequencies are missing specifically over the face
  FS7   median ||recon_i - recon_j|| / ||x_i - x_j|| over the face < 0.5
        [V] => reconstructions are collapsing toward each other
  FS4b  effective rank of dRender_face/dz at REAL latents  (diagnostic)
  FS8   data effective rank over the face region           (diagnostic)
        FS8 >> FS4b  => the model discards DOF the data has  -> beta is real
        FS8 ~= FS4b  => the data is the ceiling             -> capture more

USAGE
-----
  python3.13 face_dof.py --selftest
  python3.13 face_dof.py --data faces1 --n 200            # FS8 only, no model
  python3.13 face_dof.py --data faces1 --n 64 --model runs/tiny2/model5_constQ.pt
"""

import argparse
import math
import sys

import numpy as np

try:
    from face_budget import (_load_gray, list_images, erode, close,
                             motion_mask, gaussian_blur, box_mask,
                             radial_power, _load_model, _forward_recon,
                             _find_decoder)
except ImportError as e:                       # noqa: F841
    print("face_dof.py needs face_budget.py in the same folder")
    raise


# ----------------------------------------------------------------------------
# rank helpers
# ----------------------------------------------------------------------------

def spectrum_stats(sv):
    """Effective rank of a singular-value spectrum."""
    e = np.asarray(sv, dtype=np.float64) ** 2
    tot = e.sum()
    if tot <= 0:
        return 0.0, 0
    p = e / tot
    pr = float(np.exp(-(p * np.log(p + 1e-300)).sum()))     # participation
    n90 = int(np.searchsorted(np.cumsum(p), 0.90) + 1)
    return pr, n90


def data_rank(stack, mask, max_modes=64):
    """FS8: effective rank of the DATA over a region.  stack (T,H,W)."""
    X = stack[:, mask].astype(np.float64)          # (T, npix)
    X = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(X, compute_uv=False)
    pr, n90 = spectrum_stats(sv[:max_modes])
    return sv[:max_modes], pr, n90


# ----------------------------------------------------------------------------
# FS6 : capture per octave, INSIDE vs OUTSIDE the face
# ----------------------------------------------------------------------------

def soft_window(mask, taper=3.0):
    """Taper a binary mask so the FFT does not see its own step edge.

    Without this, the mask boundary injects broadband power into every band
    and both regions look identical at high frequency.
    """
    w = gaussian_blur(mask.astype(np.float64), taper)
    return w / max(w.max(), 1e-12)


def capture_regional(inputs, recons, mask, edges, taper=3.0):
    """Fraction of power captured per octave, separately in and out."""
    win = {"inside": soft_window(mask, taper),
           "outside": soft_window(~mask, taper)}
    out = {}
    for name, w in win.items():
        nb = min(inputs.shape[1:]) // 2
        Pin = np.zeros(nb)
        Pre = np.zeros(nb)
        for t in range(len(inputs)):
            Pin += radial_power(inputs[t] * w)
            Pre += radial_power((recons[t] - inputs[t]) * w)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            a, b = int(round(lo)), min(int(round(hi)), nb)
            if b <= a:
                continue
            pi_, pr_ = Pin[a:b].sum(), Pre[a:b].sum()
            rows.append((lo, hi, pi_ / max(Pin.sum(), 1e-30),
                         1.0 - pr_ / max(pi_, 1e-30)))
        out[name] = rows
    return out


def report_fs6(reg):
    ins, outs = reg["inside"], reg["outside"]
    print("\n  FS6  capture per octave, INSIDE vs OUTSIDE the face region")
    print("      band (cyc/img)    inside            outside        in/out")
    for (lo, hi, si, ci), (_, _, so, co) in zip(ins, outs):
        r = ci / co if co > 0 else float("nan")
        bi = "#" * int(max(0.0, min(1.0, ci)) * 20)
        print(f"      {lo:5.1f} - {hi:5.1f}   {ci:6.3f} {bi:<20s}"
              f" {co:6.3f}   {r:5.2f}")
    ci, co = ins[-1][3], outs[-1][3]
    ok = ci < 0.6 * co
    print(f"    FS6 top-octave inside {ci:.3f} < 0.6 * outside {0.6*co:.3f}"
          f"   [{'V' if ok else 'K'}]")
    print("        [V] => the high frequencies are missing SPECIFICALLY over")
    print("        the face.  Neither FS3 (spatial) nor FS5 (spectral) can see")
    print("        this, because each marginalises out the axis it lives on.")
    return ok


# ----------------------------------------------------------------------------
# FS7 : are reconstructions collapsing toward each other?
# ----------------------------------------------------------------------------

def recon_diversity(inputs, recons, mask, n_pairs=400, seed=0):
    rng = np.random.default_rng(seed)
    T = len(inputs)
    ratios = []
    din_all, drec_all = [], []
    for _ in range(n_pairs):
        i, j = rng.integers(0, T, 2)
        if i == j:
            continue
        din = float(np.sqrt(((inputs[i][mask] - inputs[j][mask]) ** 2).mean()))
        drc = float(np.sqrt(((recons[i][mask] - recons[j][mask]) ** 2).mean()))
        if din > 1e-6:
            ratios.append(drc / din)
            din_all.append(din)
            drec_all.append(drc)
    return (float(np.median(ratios)), float(np.median(din_all)),
            float(np.median(drec_all)))


# ----------------------------------------------------------------------------
# FS4b : Jacobian at REAL latents
# ----------------------------------------------------------------------------

def jacobian_rank(model, dec, z, mask, eps=0.05):
    import torch
    with torch.no_grad():
        base = dec(z)
        while isinstance(base, (tuple, list)):
            base = base[0]
        base = base.squeeze(0).numpy().mean(axis=0).ravel()[mask.ravel()]
        cols = []
        for k in range(z.shape[1]):
            zz = z.clone()
            zz[0, k] += eps
            o = dec(zz)
            while isinstance(o, (tuple, list)):
                o = o[0]
            o = o.squeeze(0).numpy().mean(axis=0).ravel()[mask.ravel()]
            cols.append((o - base) / eps)
        J = np.stack(cols, axis=1)
    return np.linalg.svd(J, compute_uv=False)


# ----------------------------------------------------------------------------

def run(args):
    files = list_images(args.data, args.n)
    size = args.size
    print(f"loading {len(files)} images at {size}px")
    lum = np.stack([_load_gray(f, size) for f in files])

    if args.box:
        b = tuple(int(v) for v in args.box.split(","))
        mask, mname = box_mask(b, size), "box (manual)"
    else:
        mask, _ = motion_mask(lum, args.motion_pct, args.motion_close)
        mname = f"motion (var > p{args.motion_pct:g}, closed)"
    inner = erode(mask, args.erode)
    if inner.sum() < 50:
        inner = mask
    print(f"  region: {mname}  full {mask.mean()*100:.1f}%  "
          f"interior {inner.mean()*100:.1f}% of canvas")

    # ---- FS8 : the data's own rank (no model needed) -----------------------
    sv, pr, n90 = data_rank(lum, inner)
    print(f"\n  FS8  DATA rank over the face interior ({len(files)} frames)")
    print(f"       singular values (top 12): "
          + " ".join(f"{v:.1f}" for v in sv[:12]))
    print(f"       dims to 90% of variance: {n90}")
    print(f"       effective rank:          {pr:.1f}")
    svb, prb, n90b = data_rank(lum, ~mask)
    print(f"       (background, for scale:  eff rank {prb:.1f}, "
          f"{n90b} dims to 90%)")

    if not args.model:
        print("\n  no --model given, stopping after FS8")
        print("  FS8 is the ceiling: the model cannot carry more DOF than the")
        print("  data has.  Compare it against FS4b before touching beta.")
        return

    # ---- model-side --------------------------------------------------------
    import torch
    model = _load_model(args.model)

    def load_rgb(p):
        from PIL import Image
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0

    rgb = np.stack([load_rgb(f) for f in files])
    x = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).float()
    model.train(False)
    with torch.no_grad():
        outs = [ _forward_recon(model, x[i:i + args.batch])
                 for i in range(0, len(x), args.batch) ]
        r = torch.cat(outs).clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    lrec = 0.299 * r[:, :, :, 0] + 0.587 * r[:, :, :, 1] + 0.114 * r[:, :, :, 2]

    # ---- FS6 ---------------------------------------------------------------
    edges = [args.f_lo * (args.f_max / args.f_lo) ** (i / args.octaves)
             for i in range(args.octaves + 1)]
    report_fs6(capture_regional(lum, lrec, mask, edges, args.taper))

    # ---- FS7 ---------------------------------------------------------------
    ratio, din, drc = recon_diversity(lum, lrec, inner)
    print(f"\n  FS7  reconstruction diversity over the face interior")
    print(f"       median RMS between INPUT pairs  {din:.4f}")
    print(f"       median RMS between RECON pairs  {drc:.4f}")
    print(f"    FS7 ratio {ratio:.3f} < 0.50   [{'V' if ratio < 0.5 else 'K'}]")
    print("        [V] => the decoder is squeezing distinct poses onto nearly")
    print("        the same face.  That is the conditional mean, and it is a")
    print("        latent-capacity result, not a basis result.")

    # ---- FS4b --------------------------------------------------------------
    dname, dec = _find_decoder(model)
    if dec is None:
        print("\n  FS4b skipped: no z -> image path found")
        return
    enc = getattr(model, "enc", None)
    if enc is None:
        print("\n  FS4b skipped: model has no .enc to get real latents from")
        return
    print(f"\n  FS4b Jacobian at REAL latents (via {dname}), "
          f"{args.jac_samples} samples")
    with torch.no_grad():
        idx = np.linspace(0, len(x) - 1, args.jac_samples).round().astype(int)
        zs = []
        for i in idx:
            z = enc(x[i:i + 1])
            while isinstance(z, (tuple, list)):
                z = z[0]
            zs.append(z)
    prs, n90s = [], []
    for k, z in enumerate(zs):
        sv = jacobian_rank(model, dec, z, inner, args.eps)
        p, n = spectrum_stats(sv)
        prs.append(p)
        n90s.append(n)
        print(f"       sample {k}: |z| {float(z.norm()):6.2f}   sv "
              + " ".join(f"{v:.2f}" for v in sv[:6])
              + f"   eff rank {p:.1f}   {n} dims to 90%")
    z0 = torch.zeros_like(zs[0])
    sv0 = jacobian_rank(model, dec, z0, inner, args.eps)
    p0, n0 = spectrum_stats(sv0)
    print(f"       ORIGIN z=0 (v2's probe point, |z| 0.00): "
          f"eff rank {p0:.1f}, {n0} dims to 90%")
    print(f"\n       model eff rank at real latents: {np.median(prs):.1f}"
          f"   vs DATA eff rank {pr:.1f}")
    if pr > 2.0 * np.median(prs):
        print("       => the DATA has DOF the model is discarding.  beta /")
        print("          encoder capacity is a real suspect.  Retrain.")
    else:
        print("       => the model is carrying about as many DOF as the data")
        print("          has.  Lowering beta buys nothing; capture footage")
        print("          with more expression range instead.")


# ----------------------------------------------------------------------------

def run_selftest(args):
    ok = True
    size = 64
    rng = np.random.default_rng(0)

    print("SD1  data_rank recovers a planted rank-3 dataset")
    T = 40
    basis = rng.standard_normal((3, size, size))
    coef = rng.standard_normal((T, 3))
    stack = np.einsum("tk,kij->tij", coef, basis) * 0.05 + 0.5
    m = np.ones((size, size), dtype=bool)
    sv, pr, n90 = data_rank(stack, m)
    print(f"     eff rank {pr:.2f} (want ~3), dims to 90% = {n90} (want 3)"
          f"  [{'V' if 2.0 < pr < 4.0 and n90 == 3 else 'K'}]")
    ok &= (2.0 < pr < 4.0 and n90 == 3)

    print("\nSD2  recon_diversity: identity ~1.0, constant ~0.0")
    r_id = recon_diversity(stack, stack.copy(), m)[0]
    const = np.repeat(stack.mean(axis=0)[None], T, axis=0)
    r_c = recon_diversity(stack, const, m)[0]
    print(f"     identity ratio {r_id:.3f} (want ~1)   "
          f"constant ratio {r_c:.3f} (want ~0)"
          f"  [{'V' if abs(r_id-1) < 0.01 and r_c < 0.01 else 'K'}]")
    ok &= (abs(r_id - 1) < 0.01 and r_c < 0.01)

    print("\nSD3  POSITIVE -- blur applied ONLY inside the mask must trip FS6")
    base = np.stack([gaussian_blur(rng.standard_normal((size, size)), 0.8)
                     for _ in range(8)])
    mm = np.zeros((size, size), dtype=bool)
    mm[16:48, 16:48] = True
    rec = base.copy()
    for t in range(len(base)):
        b = gaussian_blur(base[t], 2.5)
        rec[t] = np.where(mm, b, base[t])
    edges = [1.0 * 16.0 ** (i / 4) for i in range(5)]
    r3 = report_fs6(capture_regional(base, rec, mm, edges, 3.0))
    print(f"     -> must be V  [{'V' if r3 else 'K'}]")
    ok &= bool(r3)

    print("\nSD4  NEGATIVE -- blur applied EVERYWHERE must NOT trip FS6")
    rec2 = np.stack([gaussian_blur(b, 2.5) for b in base])
    r4 = report_fs6(capture_regional(base, rec2, mm, edges, 3.0))
    print(f"     -> must be K  [{'V' if not r4 else 'K'}]")
    ok &= (not r4)

    print("\nSD5  soft_window tapers (no step edge for the FFT to see)")
    w = soft_window(mm, 3.0)
    g = np.abs(np.gradient(w)[0]).max()
    gh = np.abs(np.gradient(mm.astype(float))[0]).max()
    print(f"     max |grad| tapered {g:.3f} vs hard {gh:.3f}"
          f"  [{'V' if g < 0.5 * gh else 'K'}]")
    ok &= g < 0.5 * gh

    print(f"\nSELFTEST {'ALL [V]' if ok else '[K] -- do not trust the metrics'}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--data", default="faces1")
    p.add_argument("--model", default="")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--box", default="")
    p.add_argument("--motion_pct", type=float, default=85.0)
    p.add_argument("--motion_close", type=int, default=6)
    p.add_argument("--erode", type=int, default=8)
    p.add_argument("--taper", type=float, default=3.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--jac_samples", type=int, default=4)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--q", type=float, default=0.6)
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
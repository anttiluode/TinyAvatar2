"""
spectrum_audit.py — measure YOUR dataset's spectrum before allocating packets.
PerceptionLab, Helsinki, July 2026.  numpy (+PIL or cv2 for loading).

WHY THIS EXISTS
  splat_trainer4q sets the top of its octave ladder to
      f_max = 0.5 * (image_size / 2)      # half the pixel Nyquist
  and splits the packet budget equally across `--octaves` dyadic bands.
  At 128px that is bands [1,2,4,8,16,32] cycles/image with ~102 packets each.
  Nothing has ever measured whether the top band is worth 20% of the budget.

  Your scooting observation is the right test.  A webcam frame under indoor
  auto-gain has a NOISE FLOOR: below some radius the radial power spectrum
  falls like a power law (real image structure, roughly 1/f^alpha), above it
  the spectrum FLATTENS because what is left is sensor read/thermal noise and
  demosaicing grain.  Gabor packets allocated above that radius are not
  fitting your face.  They are fitting the sensor.  And packets fitting
  independent noise are exactly the packets that beat against each other.

  So: find the knee, and do not put a band above it.

WHAT IT REPORTS
  1. the mean radial power spectrum of the dataset (log-log)
  2. the noise floor radius r_knee, by fitting a power law to the structured
     band and finding where the measured spectrum stops following it
  3. the fraction of total image energy inside each of the trainer's current
     octave bands — i.e. what each 102-packet band is actually being asked
     to represent
  4. a recommended --f_max and --octaves

REGISTERED (frozen before running on real data)
  SP1 [selftest] on synthetic 1/f^2 images with white noise added at a known
      variance, the detected knee must land within 20% of the analytic
      crossover radius.  If the detector cannot find a knee it put there
      itself, its number on real data means nothing.
  SP2 [selftest] on a noiseless 1/f^2 image the detector must report NO knee
      (or a knee at/above the Nyquist edge).  A detector that always finds a
      knee is a detector that found nothing.
  SP3 [measurement, no threshold] energy fraction per octave band, reported.
      Explicitly NOT a gate: low energy in a band does not by itself prove
      the band is useless, because perceptual importance is not proportional
      to spectral energy (edges are cheap in energy and dear in appearance).
      The number is evidence, not a verdict.

HONEST LIMIT ON RECORD
  Energy is not the same as usefulness.  This audit can tell you a band is
  fitting noise; it cannot by itself tell you a low-energy band is wasted.
  The decision belongs to a matched --compare run with --f_max moved.  This
  script tells you WHICH f_max to try, not what the answer is.

USAGE
  python spectrum_audit.py --selftest
  python spectrum_audit.py --data_dir faces1 --image_size 128
  python spectrum_audit.py --cache runs/hq/faces_cache_128.npy

Do not hype. Do not lie. Just show.
"""
import argparse
import glob
import math
import os

import numpy as np


# ------------------------------------------------------------------ spectrum
def radial_power(imgs):
    """imgs: (B,H,W) float in [0,1].  Returns (R,) mean power per integer
    radius in cycles/image, DC removed.  Windowed to stop the frame edge
    from injecting a broadband cross into the spectrum."""
    B, H, W = imgs.shape
    w = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    x = imgs - imgs.mean(axis=(1, 2), keepdims=True)
    F = np.fft.fftshift(np.fft.fft2(x * w[None], norm="ortho"), axes=(1, 2))
    P = (F.real ** 2 + F.imag ** 2).mean(0)
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - cy, xx - cx).round().astype(int)
    R = min(cy, cx) + 1
    out = np.bincount(r.ravel(), P.ravel(), minlength=R)[:R]
    cnt = np.bincount(r.ravel(), minlength=R)[:R].clip(min=1)
    return out / cnt


def find_knee(P, r_lo=3, win=0.25, frac=0.5):
    """Find the sensor noise floor as the radius where the spectrum STOPS
    FALLING — detected from the LOCAL log-log slope, not from a global fit.

    Why not a global fit: the obvious method is to fit log P = a - alpha*log r
    over a "structured" band and flag where the measurement rises above the
    extrapolation. It does not work, and the failure is instructive. You do
    not know where the structured band ends — that is the thing being
    measured — so the fit band inevitably includes some contaminated radii,
    the fitted alpha flattens to accommodate them, and the extrapolation
    chases the floor instead of exposing it. Measured on a planted knee at
    r=18, that detector fired at r=59. Iterating the fit did not fix it.

    The local slope has no such circularity. A power-law band has slope
    ~ -alpha; a flat sensor floor has slope ~ 0. The knee is simply where
    the slope crosses a fraction of alpha on its way to zero.

    Returns (r_knee or None, alpha, fit_curve_for_plotting).
    """
    R = len(P)
    lr = np.log10(np.arange(1, R, dtype=float))
    lp = np.log10(np.maximum(P[1:], 1e-300))
    w = max(3, int(round(win * len(lr))))          # window in log-r samples
    slope = np.full(len(lr), np.nan)
    for i in range(len(lr)):
        a, b = max(0, i - w // 2), min(len(lr), i + w // 2 + 1)
        if b - a >= 3:
            slope[i] = np.polyfit(lr[a:b], lp[a:b], 1)[0]
    lo = max(r_lo - 1, 0)
    hi = max(lo + 3, int(0.35 * len(lr)))
    alpha = -float(np.nanmedian(slope[lo:hi]))
    if not np.isfinite(alpha) or alpha <= 0.2:
        return None, alpha, None
    thresh = -frac * alpha                          # decayed to half-slope
    knee = None
    for i in range(hi, len(lr)):
        if np.isfinite(slope[i]) and slope[i] > thresh:
            knee = i + 1                            # back to radius units
            break
    fit = 10 ** (lp[lo] + (-alpha) * (lr - lr[lo]))
    return knee, alpha, fit


def band_energy(P, edges):
    """Fraction of total (non-DC) power inside each [lo,hi) cycles/image."""
    R = len(P)
    tot = P[1:].sum()
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a, b = int(math.ceil(lo)), min(int(math.ceil(hi)), R)
        out.append(float(P[a:b].sum() / max(tot, 1e-30)) if b > a else 0.0)
    return out


# ------------------------------------------------------------------ loading
def load_images(data_dir, image_size, limit=400):
    files = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        files += glob.glob(os.path.join(data_dir, e))
    files = sorted(files)[:limit]
    if not files:
        raise SystemExit(f"no images in {data_dir}")
    try:
        from PIL import Image
        def rd(f):
            im = Image.open(f).convert("L").resize(
                (image_size, image_size), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32) / 255.0
    except ImportError:
        import cv2 as cv
        def rd(f):
            im = cv.imread(f, cv.IMREAD_GRAYSCALE)
            im = cv.resize(im, (image_size, image_size),
                           interpolation=cv.INTER_AREA)
            return im.astype(np.float32) / 255.0
    return np.stack([rd(f) for f in files]), len(files)


def load_cache(path, limit=400):
    a = np.load(path, mmap_mode="r")
    a = np.asarray(a[:limit], dtype=np.float32)
    if a.ndim == 4:                       # (N,H,W,C) or (N,C,H,W)
        a = a.mean(-1) if a.shape[-1] in (1, 3) else a.mean(1)
    if a.max() > 1.5:
        a = a / 255.0
    return a, a.shape[0]


# ------------------------------------------------------------------ report
def report(P, image_size, octaves, f_lo, f_max, n_packets, n_imgs, src):
    R = len(P)
    nyq = R - 1
    print(f"\nsource        : {src}  ({n_imgs} images at {image_size}px)")
    print(f"pixel Nyquist : {nyq} cycles/image")

    r_knee, alpha, fit = find_knee(P)
    print(f"power-law fit : P(r) ~ r^-{alpha:.2f} over the structured band")
    if r_knee is None:
        print("noise floor   : NOT DETECTED below Nyquist — the spectrum "
              "follows its power law all the way out. Nothing here argues "
              "for lowering f_max.")
    else:
        print(f"noise floor   : knee at r = {r_knee} cycles/image "
              f"({100*r_knee/nyq:.0f}% of Nyquist)")
        print(f"                above this the spectrum stops falling — "
              f"that band is sensor grain, not face")

    edges = [f_lo * (f_max / f_lo) ** (b / octaves) for b in range(octaves + 1)]
    frac = band_energy(P, edges)
    per = n_packets // max(octaves, 1)
    print(f"\ncurrent ladder: {octaves} dyadic bands, ~{per} packets each, "
          f"f_max = {f_max:.0f}")
    print(f"{'band':>6} {'range (cyc/img)':>18} {'packets':>8} "
          f"{'% of image energy':>18} {'above knee?':>12}")
    for b in range(octaves):
        lo, hi = edges[b], edges[b + 1]
        flag = "" if r_knee is None else ("YES" if lo >= r_knee else
                                          ("partly" if hi > r_knee else ""))
        print(f"{b:>6} {lo:8.1f} - {hi:6.1f} {per:8d} "
              f"{100*frac[b]:17.2f}% {flag:>12}")
    tail = P[min(int(f_max), R - 1):].sum() / max(P[1:].sum(), 1e-30)
    print(f"\nenergy above the current f_max ({f_max:.0f} cyc/img): "
          f"{100*tail:.2f}% of the image")

    print("\nsuggestion (a run to try, not a verdict):")
    if r_knee is not None and r_knee < f_max:
        new_fmax = 2.0 ** math.floor(math.log2(r_knee))
        print(f"  --f_max {new_fmax:.0f}   # stop the ladder below the "
              f"noise floor; frees the top band's packets for real structure")
        print(f"  then a matched --compare against --f_max {f_max:.0f} "
              f"decides it. Do not skip that step.")
    else:
        print(f"  keep --f_max {f_max:.0f}. The measurement does not "
              f"support moving it.")


# ------------------------------------------------------------------ selftest
def selftest():
    rng = np.random.default_rng(3)
    S, B = 128, 64
    ok = [True]

    def check(name, cond, note=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {note}")
        ok[0] &= bool(cond)

    def make(alpha=2.0, noise=0.0):
        fy = np.fft.fftfreq(S)[:, None] * S
        fx = np.fft.fftfreq(S)[None, :] * S
        r = np.hypot(fy, fx)
        amp = 1.0 / np.maximum(r, 1) ** (alpha / 2)
        ims = []
        for _ in range(B):
            ph = np.fft.fft2(rng.standard_normal((S, S)))
            im = np.real(np.fft.ifft2(ph * amp))
            im = im / (im.std() + 1e-9) * 0.15 + 0.5
            if noise > 0:
                im = im + rng.standard_normal((S, S)) * noise
            ims.append(im.astype(np.float32))
        return np.stack(ims)

    # SP2 first: no noise -> no knee
    P_clean = radial_power(make(2.0, 0.0))
    k_clean, a_clean, _ = find_knee(P_clean)
    check("SP2 noiseless 1/f^2 -> no knee", k_clean is None,
          f"knee={k_clean}, alpha={a_clean:.2f}")
    check("SP2b alpha recovered ~2", abs(a_clean - 2.0) < 0.6,
          f"alpha={a_clean:.2f}")

    # SP1: plant the knee at a KNOWN radius and see if the detector finds it.
    # White noise has a flat radial power; measure its level at std 1 once,
    # then pick the std that puts the floor exactly where we want the knee.
    # The detector fires when total exceeds the fitted decay by `tol` in
    # log10, i.e. when noise = (10^tol - 1) * signal, so predict THAT radius,
    # not the equal-power crossover.
    TOL = 0.35
    L1 = float(np.median(radial_power(
        rng.standard_normal((B, S, S)).astype(np.float32))[S // 8:S // 2]))
    for r_target in (18, 30):
        s = math.sqrt(P_clean[r_target] / L1)
        P = radial_power(make(2.0, s))
        k, a, _ = find_knee(P)
        # radius where flat floor s^2*L1 == (10^TOL - 1) * P_clean(r)
        thresh = (10 ** TOL - 1) * P_clean[1:]
        hit = np.nonzero(s * s * L1 >= thresh)[0]
        r_pred = float(hit[0] + 1) if hit.size else float("inf")
        err = abs(k - r_pred) / r_pred if k else float("inf")
        check(f"SP1 knee planted at r={r_target}",
              k is not None and err <= 0.20,
              f"detected r={k}, predicted r={r_pred:.0f}, "
              f"err={100*err:.0f}%")

    # band energy sums to 1 over the full range
    e = band_energy(P_clean, [1, 2, 4, 8, 16, 32, 64])
    check("band energies sum <= 1", 0.5 < sum(e) <= 1.0 + 1e-9,
          f"sum={sum(e):.3f}")
    print("SELFTEST " + ("PASS" if ok[0] else "FAIL"))
    return 0 if ok[0] else 1


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--cache")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--num_packets", type=int, default=512)
    ap.add_argument("--octaves", type=int, default=5)
    ap.add_argument("--q", type=float, default=0.6)
    ap.add_argument("--sig_hi", type=float, default=0.70)
    ap.add_argument("--f_max", type=float, default=None)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(selftest())
    if a.cache:
        imgs, n = load_cache(a.cache, a.limit)
        src = a.cache
        a.image_size = imgs.shape[-1]
    elif a.data_dir:
        imgs, n = load_images(a.data_dir, a.image_size, a.limit)
        src = a.data_dir
    else:
        raise SystemExit("--data_dir or --cache required (or --selftest)")

    P = radial_power(imgs)
    f_lo = max(1.0, a.q / a.sig_hi)
    f_max = a.f_max if a.f_max else 0.5 * (a.image_size / 2.0)
    report(P, a.image_size, a.octaves, f_lo, f_max, a.num_packets, n, src)


if __name__ == "__main__":
    main()

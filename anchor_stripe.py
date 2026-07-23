"""
anchor_stripe.py — does the octave banding stripe the image?  numpy, executed.

THE OBSERVATION
  GaborRendererQ anchors packet k to a raster-scan position on a regular
  side x side lattice (side = ceil(sqrt(N))), and assigns its octave with
      band = floor(k * octaves / n_car)
  Both are functions of the SAME index k. Contiguous k is a contiguous run
  of raster-scan cells, which is a horizontal strip of the lattice. So band
  and image region are coupled: at 512 packets / 5 octaves / a 23x23 grid,
  band 0 (f 1-2, the head-outline gestalt) is anchored to x in [0.08, 0.23]
  and band 4 (f 16-32) to x in [0.73, 0.92].

  The decoder CAN move packets — px = sigmoid(anchor_logit + raw) and raw is
  unbounded — but crossing the frame needs raw ~ +-4.6 against weight decay
  pulling raw to 0. The lattice is a prior, and this prior says "no coarse
  structure exists on the right half of the face."

REGISTERED BEFORE THE RUN
  AS1 [diagnostic] with band = floor(k*O/n), each band's anchor x-range
      spans < 40% of the frame.  With band = k % O, every band spans > 80%.
  AS2 [FRAME POWER] least-squares PSNR at matched N, matched per-octave
      counts, matched sigma/freq — striped anchors vs interleaved anchors.
      Direction predicted: interleaved wins, because striped cannot place a
      low-frequency envelope over the right side of the image at all.
      Gate: interleaved - striped >= +1.0 dB.
  AS3 [control] shuffling the band assignment RANDOMLY (rather than
      interleaving) should score the same as interleaved. If it does not,
      the effect is not about spatial coverage and AS2's story is wrong.
  AS4 [does it survive learned offsets?] repeat AS2 letting every centre
      move by a bounded offset, the way the decoder can. If the gap closes
      completely the bug is cosmetic; if it persists it is real.

HONEST LIMIT
  Synthetic multiscale target, fixed geometry, least squares. This measures
  the PRIOR's reachable set, not a trained model. A trained decoder may
  recover some of the gap by learning large offsets; AS4 estimates how much.

Do not hype. Do not lie. Just show.
"""
import math
import numpy as np

S, Q, F0, OCT = 96, 0.6, 1.0, 5
N = 512
rng_global = np.random.default_rng(5)
yy, xx = (np.mgrid[0:S, 0:S].astype(float) + 0.5) / S


def anchors(n):
    side = int(math.ceil(math.sqrt(n)))
    ax = np.linspace(0.08, 0.92, side)
    g = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)[:n]
    return g


def bands(n, mode, rng):
    k = np.arange(n)
    if mode == "striped":                 # what ships
        return np.minimum((k * OCT // n), OCT - 1)
    if mode == "interleave":              # the one-line fix
        return k % OCT
    if mode == "shuffle":                 # control
        b = np.minimum((k * OCT // n), OCT - 1)
        return rng.permutation(b)
    raise ValueError(mode)


def build(n, mode, rng, jitter=0.0):
    a = anchors(n)
    b = bands(n, mode, rng)
    f = F0 * 2.0 ** b
    sig = Q / f
    th = rng.uniform(0, np.pi, n)
    c = a.copy()
    if jitter > 0:
        c = c + rng.uniform(-jitter, jitter, c.shape)
        c = np.clip(c, 0.02, 0.98)
    return [(c[i, 0], c[i, 1], f[i], th[i], sig[i]) for i in range(n)], b, a


def to_R(pk):
    rows = []
    for cx, cy, f, th, sg in pk:
        dx, dy = xx - cx, yy - cy
        u = np.cos(th) * dx + np.sin(th) * dy
        env = np.exp(-(dx * dx + dy * dy) / (2 * sg * sg))
        rows += [(env * np.cos(2 * np.pi * f * u)).ravel(),
                 (env * np.sin(2 * np.pi * f * u)).ravel()]
    return np.array(rows)


def fit(pk, tgt):
    R = to_R(pk)
    c, *_ = np.linalg.lstsq(R.T, tgt.ravel(), rcond=None)
    mse = float(np.mean(((c @ R).reshape(S, S) - tgt) ** 2))
    return 10 * np.log10((tgt.max() - tgt.min()) ** 2 / mse)


def synthetic_face(rng):
    im = np.zeros((S, S))
    im += .55 * np.exp(-(((xx - .5) / .30) ** 2 + ((yy - .52) / .38) ** 2) ** 3)
    for sx in (-1, 1):
        d2 = (xx - (.5 + sx * .13)) ** 2 + (yy - .42) ** 2
        im -= .45 * np.exp(-d2 / (2 * .035 ** 2))
        im -= .35 * np.exp(-d2 / (2 * .013 ** 2))
        im += .18 * np.cos(2 * np.pi * 30 * np.sqrt(d2 + 1e-9)) * \
            np.exp(-d2 / (2 * .022 ** 2))
    im -= .30 * np.exp(-((xx - .5) ** 2 / (2 * .09 ** 2)
                         + (yy - .70) ** 2 / (2 * .012 ** 2)))
    im += .10 * np.exp(-((xx - .5) ** 2 / (2 * .03 ** 2)
                         + (yy - .57) ** 2 / (2 * .05 ** 2)))
    k = np.fft.fftfreq(S)[:, None] ** 2 + np.fft.fftfreq(S)[None, :] ** 2
    t = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((S, S)))
                             / (k + 1e-3) ** .55))
    return (im + .12 * t / (t.std() + 1e-9)) - im.mean()


# ---------------------------------------------------------------- AS1
print(f"anchors: {int(math.ceil(math.sqrt(N)))}x"
      f"{int(math.ceil(math.sqrt(N)))} lattice, N={N}, octaves={OCT}\n")
print(f"{'mode':>11} {'band':>5} {'freq':>9} {'n':>4} {'anchor x span':>16} "
      f"{'% of frame':>11}")
spans = {}
for mode in ("striped", "interleave"):
    _, b, a = build(N, mode, np.random.default_rng(0))
    sp = []
    for bb in range(OCT):
        m = b == bb
        lo, hi = a[m, 0].min(), a[m, 0].max()
        sp.append((hi - lo) / 0.84)
        print(f"{mode:>11} {bb:5d} {2**bb:4d}-{2**(bb+1):<4d} {int(m.sum()):4d} "
              f"{lo:6.3f} - {hi:.3f} {100*sp[-1]:10.0f}%")
    spans[mode] = sp
    print()
as1 = max(spans["striped"]) < 0.40 and min(spans["interleave"]) > 0.80
print(f"AS1 {'[V]' if as1 else '[K]'} striped max span "
      f"{100*max(spans['striped']):.0f}% (<40), interleave min span "
      f"{100*min(spans['interleave']):.0f}% (>80)\n")

# ------------------------------------------------------------ AS2/AS3/AS4
res = {m: {j: [] for j in (0.0, 0.05, 0.15)} for m in
       ("striped", "interleave", "shuffle")}
for seed in range(4):
    rng = np.random.default_rng(400 + seed)
    tgt = synthetic_face(rng)
    for mode in res:
        for jit in (0.0, 0.05, 0.15):
            pk, _, _ = build(N, mode, np.random.default_rng(900 + seed), jit)
            res[mode][jit].append(fit(pk, tgt))

print(f"least-squares PSNR, matched N={N}, matched per-octave counts, "
      f"4 seeds")
print(f"{'anchors':>11} {'no jitter':>16} {'jitter 0.05':>16} "
      f"{'jitter 0.15':>16}")
for mode in ("striped", "interleave", "shuffle"):
    row = "".join(f"{np.mean(res[mode][j]):11.2f} +-{np.std(res[mode][j]):4.2f}"
                  for j in (0.0, 0.05, 0.15))
    print(f"{mode:>11} {row}")

d0 = np.mean(res["interleave"][0.0]) - np.mean(res["striped"][0.0])
d1 = np.mean(res["interleave"][0.15]) - np.mean(res["striped"][0.15])
ds = abs(np.mean(res["interleave"][0.0]) - np.mean(res["shuffle"][0.0]))
print(f"\nAS2 {'[V]' if d0 >= 1.0 else '[K]'} interleave - striped = "
      f"{d0:+.2f} dB with fixed anchors (gate >= +1.0)")
print(f"AS3 {'[V]' if ds <= 0.5 else '[K]'} shuffle control matches "
      f"interleave to {ds:.2f} dB (gate <= 0.5) — effect is spatial "
      f"coverage, not ordering")
print(f"AS4 gap under learned-offset jitter 0.15: {d1:+.2f} dB "
      f"({'persists' if d1 >= 0.5 else 'closes'})")

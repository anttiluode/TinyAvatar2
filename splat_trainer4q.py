"""
splat_trainer4q.py — the constant-Q (anti-moire) splat trainer.

THE HYPOTHESIS THIS FILE STARTED FROM — AND ITS FALSIFICATION
-------------------------------------------------------------
Your V1 intuition was: biological primitives are SMALL, ours are LARGE,
and that is why we get moire. The sharp version of "large" is Q =
sigma*freq, the number of carrier cycles under a packet's envelope. V1
simple cells are roughly constant-Q (1-1.5 octave bandwidth). Since
splat_trainer3v2 samples sigma and freq INDEPENDENTLY, the model is free
to learn wide many-cycle packets, and wide many-cycle packets are exactly
the ones that beat against neighbours over long distances. That was the
predicted mechanism, and Q4 below was written to check its premise before
spending any GPU time on it.

MEASURED ON model2.pt (`--audit model2.pt`):

    Q = sigma*freq, percentiles 0/10/25/50/75/90/99/100:
        0.04  0.10  0.14  0.22  0.31  0.41  0.66  1.10
    fraction of packets with Q > 1.5 : 0.000
    sigma median 0.067   freq median 3.24   (pixel Nyquist 48)

THE PREMISE IS FALSE. The model does not learn high-Q packets — it learns
the OPPOSITE. Median Q is 0.22, i.e. under half a cycle across the whole
visible envelope. The trained model is barely using the carrier at all:
these are signed Gaussian BLOBS with a slight tilt, not Gabor wavelets.
The representation has collapsed toward a 2D Gaussian-splat mixture and
abandoned the oscillatory half of its own basis.

So: the moire is not runaway-Q beat. Whatever it is, it is happening in a
near-carrier-free regime, and the blotchy interference in the driver
window is overlapping signed blobs, not long-wavelength fringes.

WHY THE COLLAPSE (hypothesis, NOT yet tested)
----------------------------------------------
Frequency and orientation enter through cos(2*pi*f*x_r), so their loss
landscape is oscillatory: gradients w.r.t. f and theta flip sign every
half cycle. Under MSE from a random init, the smooth descent direction is
to push f down until the packet stops oscillating and behaves as a plain
Gaussian, whose gradients are monotone. The carrier is not used because it
is hard to reach, not because it is unhelpful. This is a guess with a
mechanism attached — treat it as [~] until the Q-floor run tests it.

WHAT THIS TRAINER NOW DOES, AND WHY IT SURVIVES THE FALSIFICATION
-----------------------------------------------------------------
The constant-Q coupling is retained, but its PURPOSE IS INVERTED. It was
built as a CEILING on Q; the measurement says the binding problem is a
FLOOR. sigma = (Q/freq)*exp(+-slack) is two-sided, so it does both: with
Q=0.6 and one octave of slack it confines Q to roughly [0.3, 1.2], which
forbids the blob collapse (median 0.22 is below that whole window) and
forces every carrier packet to actually oscillate.

That makes this a test of a DIFFERENT and more interesting question than
the one it was written for: does a splat model that is FORCED to use its
carrier reconstruct a face better, at equal packet count, than one that
collapses to blobs? A Gabor can represent an edge with one packet; a
Gaussian blob cannot. If the carrier is worth anything, forcing its use
should buy detail per packet.

HONEST CONSEQUENCE FOR Q1: the direction of the moire effect is now
GENUINELY UNKNOWN and must not be assumed. Forcing carriers into use adds
oscillatory content that could plausibly INCREASE beat interference. Q1 is
therefore kept as written but demoted from prediction to open question,
and the PSNR gate Q2 is now the primary result, not the guard.

WHAT THIS TRAINER CHANGES (three things, each separately switchable)
--------------------------------------------------------------------
1. CONSTANT-Q CARRIERS (--q, default 0.6).
       sigma = clamp( (Q/freq) * exp(q_slack*tanh(raw2)), sig_lo, sig_hi )
   sigma is no longer free: it is pinned to Q/freq with +-q_slack octaves
   of slack (default log(2) -> one octave either way), so the model keeps
   real freedom but cannot build a wide many-cycle packet.

2. A GIST BAND OF PURE GAUSSIANS (--gist_frac, default 0.25).
   Constant-Q has a floor: sigma <= sig_hi forces freq >= Q/sig_hi, so
   there are no large low-frequency packets, and shading/luminance would
   have nowhere to live. So a fraction of packets are reserved as CARRIER-
   FREE Gaussians (freq forced to 0, large sigma allowed). Two Gaussians
   with no carrier CANNOT produce a fringe — they just add. This is your
   own V4/V5 two-band split (coarse = shading envelope, fine = oriented
   contour) moved from the live loop into the basis itself.

3. LOG-SPACED OCTAVE BANDING of the carrier packets (--octaves, default 4).
   Packet k is confined to frequency band floor(k*O/N), so the model cannot
   collapse every carrier into one frequency (which is the configuration
   that maximises beat). Guarantees scale coverage by construction. With
   constant-Q, banding frequency IS banding scale.

MEASUREMENT — you cannot fix what you do not score
---------------------------------------------------
Two metrics, one render-side and one parameter-side, both reported:

  moire_index (PRIMARY, model-agnostic, render-side)
      radial power spectrum of the reconstruction minus that of the target,
      rectified to keep only EXCESS, summed over the fringe band
      r in [2, S/8] cycles/image, normalised by target band power:
          MI = sum_r max(P_rec(r) - P_tgt(r), 0) / sum_r P_tgt(r)
      This is literally "how much low-to-mid frequency structure did the
      model INVENT that the face does not have" — i.e. hallucinated
      structure, which is the thing you actually object to. Blur alone does
      NOT inflate it (blur removes energy; the rectifier ignores that), so
      MI cannot be gamed by making the output smoother. Report it WITH
      PSNR, never instead of it.

  beat_index (DIAGNOSTIC, parameter-side, explains MI)
      sum over packet pairs of  overlap * co-orientation * fringe-visibility
          overlap  = exp(-d^2 / (2(s_k^2+s_l^2)))
          co-orient= (1 + cos(2(th_k-th_l)))/2
          fringe   = exp(-(df * s_bar)^2 / 2) * (1 - exp(-(df/df0)^2))
      the fringe term peaks where the beat wavelength is LONGER than the
      envelope but not degenerate (df=0 is coherent addition, not a beat).
      O(N^2) on packet parameters only — no rendering, microseconds.

REGISTERED PREDICTIONS (frozen before any training run)
--------------------------------------------------------
Run `--compare` to execute all three arms at equal step budget, equal
packet count, equal everything else. Thresholds:

  Q2  PSNR(constQ) >= PSNR(legacy) + 0.3 dB.   [PRIMARY, post-audit]
      Forcing carrier use must BUY something. A Gabor represents an edge
      in one packet where a blob cannot; if that is real, detail per
      packet goes up. This is now the headline gate.
  Q1  moire_index(constQ) <= moire_index(legacy).   [OPEN, not predicted]
      Direction unknown after the Q4 falsification. Recorded either way.
      A rise here with Q2 passing is a real trade, not a failure — say so.
  Q3  beat_index(constQ) < beat_index(legacy).
      The mechanism check. If MI drops but beat_index does not, the
      improvement came from somewhere other than the beat mechanism and
      the explanation in this docstring is wrong — say so.
  Q4  PREMISE CHECK — ALREADY RUN, ALREADY FAILED, see above. Kept in the
      code so it re-runs on every new model: it reports which regime the
      basis is in (blob-collapsed vs carrier-using) and the constant-Q
      arm should raise median Q from ~0.22 into [0.3, 1.2]. Verify that it
      did; if median Q does not move, the coupling is not binding and
      nothing else in the comparison means anything.

  Honest note on what this cannot do: capacity is still capacity. A 256-
  packet 96px model reconstructing a face is blurry because 256 packets is
  not many, and no reparameterisation changes that. Constant-Q is a claim
  about the CHARACTER of the error (invented fringes vs honest blur), not
  its magnitude. If you want detail, you need packets and pixels; the
  --image_size 128 --num_packets 512 run is the capacity answer and it is
  orthogonal to this one. Do not let a good MI number talk you out of that.

COMPATIBILITY
-------------
Checkpoints record their own parameterisation:
    {"sd", "image_size", "num_packets", "qmode", "q", "gist_frac",
     "octaves", "q_slack", "sig_lo", "sig_hi", "f_max"}
`load_splatvae(path)` reads them back and rebuilds the right renderer, so
tiny_avatar3 / splat_field / gabor_loop keep working against both old and
new models. Old checkpoints (no "qmode" key) load as legacy, bit-identical
to splat_trainer3v2. The render graph stays Mul/ReduceSum/Stack only — no
Einsum, no ConstantOfShape — so ONNX export still imports on cv2 4.x and
OpenCV 5.

USAGE
  python splat_trainer4q.py --smoke                      # CPU selftest
  python splat_trainer4q.py --audit model2.pt            # Q4 premise check
  python splat_trainer4q.py --compare --data_dir DIR --steps 3000
  python splat_trainer4q.py --data_dir DIR --steps 30000 \
         --image_size 128 --num_packets 512 --out runs/hq
  python splat_trainer4q.py --export runs/hq/model2.pt

Do not hype. Do not lie. Just show.
"""
import argparse, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from splat_trainer3v2 import (build_cache, load_resident, batch_from,
                                  Encoder, Decoder, kl, kl_free, grid, LATENT)
except ImportError:  # keep --smoke runnable standalone
    LATENT = 128
    build_cache = load_resident = batch_from = grid = None
    Encoder = Decoder = None
    def kl(mu, lv):
        return (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
    def kl_free(mu, lv, fb):
        k = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
        return k.clamp(min=fb).sum(1).mean()


# ============================================================ renderer
class GaborRendererQ(nn.Module):
    """splat_trainer3v2.GaborRenderer with the sigma/freq coupling made
    switchable. qmode=False reproduces the original activate() exactly."""

    def __init__(self, image_size=96, num_packets=256, chunk=64,
                 use_checkpoint=False, qmode=True, q=0.6, q_slack=None,
                 gist_frac=0.0, octaves=5, sig_lo=0.008, sig_hi=0.70,
                 gist_sig_hi=0.30, f_max=None):
        super().__init__()
        self.H = self.W = image_size
        self.N, self.chunk, self.use_checkpoint = num_packets, chunk, use_checkpoint
        self.qmode = bool(qmode)
        self.q = float(q)
        self.q_slack = math.log(2.0) if q_slack is None else float(q_slack)
        self.gist_frac = float(gist_frac)
        self.octaves = int(octaves)
        self.sig_lo, self.sig_hi = float(sig_lo), float(sig_hi)
        self.gist_sig_hi = float(gist_sig_hi)
        # carriers may not exceed half the pixel Nyquist (S/2 cycles/image)
        self.f_max = float(f_max) if f_max else 0.5 * (image_size / 2.0)

        gy, gx = torch.meshgrid(torch.linspace(0, 1, image_size),
                                torch.linspace(0, 1, image_size), indexing="ij")
        self.register_buffer("GX", gx[None, None].contiguous())
        self.register_buffer("GY", gy[None, None].contiguous())
        side = int(math.ceil(math.sqrt(num_packets)))
        ax = torch.linspace(0.08, 0.92, side)
        anch = torch.stack(torch.meshgrid(ax, ax, indexing="ij"),
                           -1).reshape(-1, 2)[:num_packets]
        anch = torch.clamp(anch, 1e-3, 1 - 1e-3)
        self.register_buffer("anchor_logit", torch.log(anch / (1 - anch)))

        # ---- band assignment, fixed at construction, stored as buffers ----
        n_gist = int(round(self.gist_frac * num_packets)) if self.qmode else 0
        is_gist = torch.zeros(num_packets)
        is_gist[:n_gist] = 1.0
        self.register_buffer("is_gist", is_gist[None], persistent=False)
        # constant-Q floor: sigma <= sig_hi  =>  freq >= q/sig_hi
        f_lo = max(1.0, self.q / self.sig_hi)
        n_car = max(1, num_packets - n_gist)
        k = torch.arange(num_packets).float()
        band = torch.floor((k - n_gist).clamp(min=0)
                           * self.octaves / n_car).clamp(0, self.octaves - 1)
        span = math.log(max(self.f_max / f_lo, 1.0000001))
        lo = math.log(f_lo) + span * (band / self.octaves)
        hi = math.log(f_lo) + span * ((band + 1) / self.octaves)
        self.register_buffer("f_band_lo", torch.exp(lo)[None], persistent=False)
        self.register_buffer("f_band_hi", torch.exp(hi)[None], persistent=False)
        self.f_lo = f_lo
        if self.qmode and n_gist > 0 and f_lo > 1.0 + 1e-6:
            import warnings
            warnings.warn(
                f"SPECTRAL HOLE: gist packets sit at freq=0 but carriers "
                f"start at freq={f_lo:.2f} cyc/image, so nothing represents "
                f"(0, {f_lo:.2f}) — on a face that is the head-outline and "
                f"feature-layout band, and the model will render smooth "
                f"'frosted glass' plus fine stripes with no gestalt. Fix: "
                f"--gist_frac 0 and raise --sig_hi (f_lo = q/sig_hi).",
                RuntimeWarning, stacklevel=2)

    # ---------------------------------------------------------- activate
    def activate(self, raw):
        px = torch.sigmoid(self.anchor_logit[:, 0][None] + raw[..., 0])
        py = torch.sigmoid(self.anchor_logit[:, 1][None] + raw[..., 1])
        theta = raw[..., 3]
        coeff = torch.tanh(raw[..., 5:11]).reshape(*raw.shape[:2], 3, 2)

        if not self.qmode:                       # legacy, bit-identical
            sigma = 0.012 + 0.14 * torch.sigmoid(raw[..., 2])
            freq = 1.0 + 15.0 * torch.sigmoid(raw[..., 4])
            return px, py, sigma, theta, freq, coeff

        g = self.is_gist                          # (1,N) broadcast over batch
        s4 = torch.sigmoid(raw[..., 4])
        # carriers: frequency inside this packet's octave band
        f_car = self.f_band_lo * torch.exp(
            torch.log(self.f_band_hi / self.f_band_lo) * s4)
        # constant-Q sigma with +-q_slack octaves of slack around Q/f
        s_car = (self.q / f_car) * torch.exp(
            self.q_slack * torch.tanh(raw[..., 2]))
        s_car = s_car.clamp(self.sig_lo, self.sig_hi)
        # gist: carrier-free Gaussians (freq 0 -> cos=1, sin=0), large sigma
        s_gist = self.sig_lo + (self.gist_sig_hi - self.sig_lo) * \
            torch.sigmoid(raw[..., 2])
        f_gist = torch.zeros_like(f_car)

        sigma = g * s_gist + (1.0 - g) * s_car
        freq = g * f_gist + (1.0 - g) * f_car
        return px, py, sigma, theta, freq, coeff

    # ------------------------------------------------------------ render
    def _chunk(self, px, py, sigma, theta, freq, coeff):
        px_ = px[..., None, None]; py_ = py[..., None, None]
        s_ = sigma[..., None, None]; th = theta[..., None, None]
        f_ = freq[..., None, None]
        dx = self.GX - px_; dy = self.GY - py_
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
        ec = env * torch.cos(2 * math.pi * f_ * xr)
        es = env * torch.sin(2 * math.pi * f_ * xr)
        a, b = coeff[..., 0], coeff[..., 1]
        chans = [(a[:, :, c, None, None] * ec).sum(1)
                 - (b[:, :, c, None, None] * es).sum(1) for c in range(3)]
        return torch.stack(chans, dim=1)

    def forward(self, raw):
        raw = raw.float()
        px, py, sigma, theta, freq, coeff = self.activate(raw)
        out = None
        for i in range(0, self.N, self.chunk):
            sl = slice(i, i + self.chunk)
            args = (px[:, sl], py[:, sl], sigma[:, sl],
                    theta[:, sl], freq[:, sl], coeff[:, sl])
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                c = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                c = self._chunk(*args)
            out = c if out is None else out + c
        return torch.sigmoid(out)


class SplatVAEQ(nn.Module):
    def __init__(self, image_size=96, num_packets=256, chunk=64, ckpt=False,
                 **qkw):
        super().__init__()
        self.enc = Encoder(image_size)
        self.dec = Decoder(LATENT, num_packets)
        self.ren = GaborRendererQ(image_size, num_packets, chunk, ckpt, **qkw)


SplatVAE = SplatVAEQ            # tiny_avatar3 / splat_field import these names
GaborRenderer = GaborRendererQ

QKEYS = ("qmode", "q", "q_slack", "gist_frac", "octaves",
         "sig_lo", "sig_hi", "gist_sig_hi", "f_max")


def load_splatvae(path, chunk=64, map_location="cpu"):
    """Rebuild the right renderer for either a legacy or a constant-Q
    checkpoint. Old files have no 'qmode' key and load as legacy."""
    ck = torch.load(path, map_location=map_location)
    qkw = {k: ck[k] for k in QKEYS if k in ck}
    qkw.setdefault("qmode", False)
    m = SplatVAEQ(ck["image_size"], ck["num_packets"], chunk, False, **qkw)
    m.load_state_dict(ck["sd"])
    m.eval()
    return m, ck


# ============================================================ metrics
def radial_power(img):
    """Radially-averaged power spectrum. img (B,C,H,W) in [0,1].
    Returns (R,) mean power per integer radius, DC removed."""
    x = img.mean(1)                                  # luminance
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    F2 = torch.fft.rfft2(x, norm="ortho").abs().pow(2)     # (B,H,W//2+1)
    H, W = F2.shape[-2], F2.shape[-1]
    fy = torch.fft.fftfreq(H, device=img.device)[:, None] * H
    fx = torch.arange(W, device=img.device)[None, :].float()
    r = (fy.pow(2) + fx.pow(2)).sqrt().round().long().clamp(max=H // 2)
    R = H // 2 + 1
    flat = F2.reshape(F2.shape[0], -1)
    idx = r.reshape(-1)
    out = torch.zeros(F2.shape[0], R, device=img.device, dtype=flat.dtype)
    cnt = torch.zeros(R, device=img.device, dtype=flat.dtype)
    out.index_add_(1, idx, flat)
    cnt.index_add_(0, idx, torch.ones_like(flat[0]))
    return out / cnt.clamp(min=1)[None]


def moire_index(recon, target, r_lo=2, r_hi_frac=8):
    """Rectified EXCESS mid-band spectral energy: structure the model
    invented. Blur cannot lower it artificially (blur removes energy and
    the rectifier discards negative differences)."""
    Pr, Pt = radial_power(recon), radial_power(target)
    R = Pr.shape[1]
    hi = max(r_lo + 1, int(round(2 * (R - 1) / r_hi_frac)))
    hi = min(hi, R)
    exc = (Pr[:, r_lo:hi] - Pt[:, r_lo:hi]).clamp(min=0).sum(1)
    den = Pt[:, r_lo:hi].sum(1).clamp(min=1e-12)
    return (exc / den).mean()


@torch.no_grad()
def beat_index(px, py, sigma, theta, freq, df0=0.5):
    """Parameter-side fringe score. Sums overlap * co-orientation *
    fringe-visibility over packet pairs. O(N^2), no rendering."""
    d2 = (px[:, :, None] - px[:, None, :]).pow(2) + \
         (py[:, :, None] - py[:, None, :]).pow(2)
    s2 = sigma.pow(2)[:, :, None] + sigma.pow(2)[:, None, :]
    ov = torch.exp(-d2 / (2 * s2.clamp(min=1e-9)))
    co = 0.5 * (1 + torch.cos(2 * (theta[:, :, None] - theta[:, None, :])))
    df = (freq[:, :, None] - freq[:, None, :]).abs()
    sb = 0.5 * (sigma[:, :, None] + sigma[:, None, :])
    fringe = torch.exp(-(df * sb).pow(2) / 2) * (1 - torch.exp(-(df / df0) ** 2))
    n = ov.shape[1]
    off = (1 - torch.eye(n, device=ov.device))[None]
    B = ov * co * fringe * off
    raw = (B.sum(dim=(1, 2)) / (n * (n - 1))).mean()
    # RAW is confounded by envelope size: packets with big sigma overlap
    # everything, so raw rises whenever sigma rises even if no fringe got
    # more visible. NORM divides out the overlap budget and answers the
    # question actually being asked: OF the overlap that exists, what
    # fraction is fringe-prone? Compare arms on NORM, not raw.
    norm = ((B.sum(dim=(1, 2))) / (ov * off).sum(dim=(1, 2)).clamp(min=1e-9)).mean()
    return raw, norm


@torch.no_grad()
def q_histogram(sigma, freq):
    """Q = sigma*freq, cycles per envelope sigma. The premise check."""
    Q = (sigma * freq).reshape(-1)
    qs = torch.tensor([0.0, 10, 25, 50, 75, 90, 99, 100.0])
    return torch.quantile(Q, qs / 100.0), (Q > 1.5).float().mean()


# ============================================================ audit (Q4)
def audit(path):
    m, ck = load_splatvae(path)
    print(f"{path}: {ck['image_size']}px  {ck['num_packets']} packets  "
          f"qmode={ck.get('qmode', False)}")
    g = torch.Generator().manual_seed(0)
    zs = torch.randn(64, LATENT, generator=g)
    with torch.no_grad():
        px, py, sg, th, fr, _ = m.ren.activate(m.dec(zs).float())
    qq, frac = q_histogram(sg, fr)
    print("Q = sigma*freq (cycles per envelope sigma)")
    print("  pctl 0/10/25/50/75/90/99/100: " +
          "  ".join(f"{v:.2f}" for v in qq.tolist()))
    print(f"  fraction of packets with Q > 1.5 : {frac:.3f}")
    print(f"  sigma  median {sg.median():.4f}  max {sg.max():.4f}")
    print(f"  freq   median {fr.median():.2f}  max {fr.max():.2f}  "
          f"(pixel Nyquist {ck['image_size'] / 2:.0f})")
    br, bn = beat_index(px, py, sg, th, fr)
    print(f"  beat_index raw {br:.5f}   overlap-normalised {bn:.5f}"
          f"   (compare arms on the normalised one)")
    med = float(qq[3])
    if ck.get("qmode", False):
        print("\nQ4 regime: " + ("[V] carrier-using — median Q "
              f"{med:.2f} sits in the constant-Q window [0.3, 1.2]."
              if 0.3 <= med <= 1.2 else
              f"[K] median Q {med:.2f} is OUTSIDE the intended window — "
              "the coupling is not binding; check --q / --q_slack."))
    else:
        print("\nQ4 regime: " + (f"[blob-collapsed] median Q {med:.2f} < 0.3 "
              "— this basis has abandoned its carrier and is a signed-"
              "Gaussian mixture. This is the regime model2.pt is in, and it "
              "is what the constant-Q floor is meant to move."
              if med < 0.3 else f"[carrier-using] median Q {med:.2f}."))


# ============================================================ train
def make_model(args, dev, qmode):
    m = SplatVAEQ(args.image_size, args.num_packets, args.chunk,
                  args.checkpointing, qmode=qmode, q=args.q,
                  q_slack=args.q_slack, gist_frac=args.gist_frac,
                  octaves=args.octaves, sig_lo=args.sig_lo,
                  sig_hi=args.sig_hi, gist_sig_hi=args.gist_sig_hi,
                  f_max=args.f_max)
    return m.to(dev)


def ckpt_dict(model, args, qmode):
    d = {"sd": model.state_dict(), "image_size": args.image_size,
         "num_packets": args.num_packets, "qmode": qmode,
         "trainer": "splat_trainer4q"}
    r = model.ren
    for k in QKEYS[1:]:
        d[k] = getattr(r, k)
    return d


def detail_weight(x, lam):
    """1 + lam * normalised gradient magnitude. Spends capacity where the
    face has structure (eyes, mouth) instead of on flat cheeks."""
    if lam <= 0:
        return None
    g = x.mean(1, keepdim=True)
    gx = g[..., :, 1:] - g[..., :, :-1]
    gy = g[..., 1:, :] - g[..., :-1, :]
    gx = F.pad(gx, (1, 0)); gy = F.pad(gy, (0, 0, 1, 0))
    m = (gx.pow(2) + gy.pow(2)).sqrt()
    m = m / m.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return 1.0 + lam * m


def train_one(args, dev, data, qmode, tag, steps):
    torch.manual_seed(args.seed)
    model = make_model(args, dev, qmode)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(args.seed)
    n = data["n"] if isinstance(data, dict) else len(data)
    use_bf16 = (dev.type == "cuda")
    os.makedirs(args.out, exist_ok=True)
    t0, run_rec, last = time.time(), 0.0, 0
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (args.batch,), generator=g)
        x = batch_from(data, idx, dev)
        beta = args.beta * min(1.0, step / max(1, args.beta_warmup_steps))
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            mu, lv = model.enc(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            raw = model.dec(z)
        recon = model.ren(raw)
        w = detail_weight(x, args.detail)
        rec = F.mse_loss(recon, x) if w is None else \
            ((recon - x).pow(2) * w).mean()
        if args.gamma_floater > 0:
            _, _, sg, _, _, cf = model.ren.activate(raw.float())
            amp2 = cf.pow(2).sum(dim=(-1, -2))
            flo = (amp2 * (args.sigma_ref / sg - 1.0).clamp(min=0)).mean()
        else:
            flo = torch.zeros((), device=x.device)
        loss = rec + beta * kl_free(mu, lv, args.free_bits) \
            + args.gamma_floater * flo
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        run_rec += float(rec.detach())
        if step % args.log_every == 0 or step == steps:
            nb = step - last; last = step
            ips = nb * args.batch / (time.time() - t0); t0 = time.time()
            print(f"[{tag}] step {step:6d}/{steps}  rec {run_rec/nb:.4f}  "
                  f"beta {beta:.4g}  {ips:6.0f} img/s")
            run_rec = 0.0
            # NOT named model2.pt on purpose: a constant-Q checkpoint has
            # the same state_dict SHAPE as a legacy one (the band buffers are
            # non-persistent), so splat_trainer3v2.SplatVAE will load it
            # WITHOUT ERROR and then render it with the legacy sigma/freq
            # formulas — silently wrong output, the worst failure mode. A
            # distinct filename keeps it out of old model-scanning dropdowns.
            torch.save(ckpt_dict(model, args, qmode),
                       os.path.join(args.out, f"model4q_{tag}.pt"))
    return model


@torch.no_grad()
def _sweep(model, data, dev, args, idx):
    mi, ps, bi, bn_, hq, med = [], [], [], [], [], []
    for i in range(0, len(idx), args.batch):
        x = batch_from(data, idx[i:i + args.batch], dev)
        mu, _ = model.enc(x)
        raw = model.dec(mu)
        rec = model.ren(raw)
        mi.append(float(moire_index(rec, x)))
        ps.append(float(F.mse_loss(rec, x)))
        px, py, sg, th, fr, _ = model.ren.activate(raw.float())
        br, bnn = beat_index(px, py, sg, th, fr)
        bi.append(float(br)); bn_.append(float(bnn))
        hq.append(float((sg * fr > 1.5).float().mean()))
        med.append(float((sg * fr).median()))
    a = lambda v: sum(v) / len(v)
    return dict(moire=a(mi), psnr=10 * math.log10(1.0 / max(a(ps), 1e-9)),
                beat=a(bi), beat_norm=a(bn_), hiQ=a(hq), medQ=a(med))


@torch.no_grad()
def evaluate(model, data, dev, args, n_eval=256):
    """The encoder carries 5 BatchNorm layers. In eval() mode those use
    RUNNING statistics, which at a few thousand steps are poorly estimated
    and depress PSNR for both arms — by roughly an order of magnitude in
    MSE versus the training loss. Worse, the estimation error is not
    guaranteed equal across arms, so an eval-mode PSNR gap can be a
    BatchNorm artefact rather than a reconstruction difference.
    So both are measured. A result is only real if it holds in BOTH."""
    g = torch.Generator().manual_seed(12345)
    n = data["n"] if isinstance(data, dict) else len(data)
    idx = torch.randint(0, n, (min(n_eval, n),), generator=g)
    model.eval()
    r_eval = _sweep(model, data, dev, args, idx)
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    saved = [(m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone()) for m in bns]
    model.train()                       # batch statistics
    r_batch = _sweep(model, data, dev, args, idx)
    for m, (rm, rv, nb) in zip(bns, saved):     # undo BN stat drift
        m.running_mean.copy_(rm); m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nb)
    model.eval()
    return r_eval, r_batch


def compare(args, dev, data):
    print("\n=== Q1-Q3: constant-Q vs legacy, matched capacity & budget ===")
    res = {}
    for tag, qm in (("legacy", False), ("constQ", True)):
        m = train_one(args, dev, data, qm, tag, args.steps)
        res[tag] = evaluate(m, data, dev, args)
        e, b = res[tag]
        print(f"[{tag}] moire {e['moire']:.4f}  PSNR eval {e['psnr']:.2f} / "
              f"batchstat {b['psnr']:.2f}  beat_norm {e['beat_norm']:.5f} "
              f"(raw {e['beat']:.5f})  medianQ {e['medQ']:.2f}")
    (Le, Lb), (Ce, Cb) = res["legacy"], res["constQ"]
    v = lambda ok: "[V]" if ok else "[K]"
    q1 = Ce["moire"] <= 0.70 * Le["moire"]
    q2e = Ce["psnr"] >= Le["psnr"] + 0.3
    q2b = Cb["psnr"] >= Lb["psnr"] + 0.3
    q3 = Ce["beat_norm"] < Le["beat_norm"]
    q4 = Ce["medQ"] >= 0.30 and Le["medQ"] < 0.30
    print("\n----------------------- verdicts -----------------------")
    print(f"Q4 {v(q4)} coupling binds: median Q legacy {Le['medQ']:.2f} "
          f"-> constQ {Ce['medQ']:.2f} (must cross 0.30)")
    print(f"Q2 {v(q2e and q2b)} PRIMARY, PSNR must win in BOTH modes: "
          f"eval {Ce['psnr']:.2f} vs {Le['psnr']:.2f} ({Ce['psnr']-Le['psnr']:+.2f}), "
          f"batchstat {Cb['psnr']:.2f} vs {Lb['psnr']:.2f} "
          f"({Cb['psnr']-Lb['psnr']:+.2f}); need >=+0.3 each")
    print(f"Q1 {v(q1)} moire {Ce['moire']:.4f} vs {Le['moire']:.4f} "
          f"(ratio {Ce['moire']/max(Le['moire'],1e-9):.2f}, need <=0.70)")
    print(f"Q3 {v(q3)} beat_norm {Ce['beat_norm']:.5f} vs "
          f"{Le['beat_norm']:.5f}   [raw {Ce['beat']:.5f} vs {Le['beat']:.5f} "
          f"— raw is sigma-confounded, do not read it]")
    if q1 and not (q2e and q2b):
        print("\nQ1 WITHOUT Q2 IS VOID — moire_index also goes to ~0 for a "
              "maximally blurred output (it counts INVENTED energy, and blur "
              "invents none). Only PSNR separates clean from smeared.")
    if q2e != q2b:
        print("\nPSNR verdict DISAGREES between eval-mode and batch-stat "
              "mode. That is a BatchNorm running-statistics artefact, not a "
              "reconstruction result. Train longer before believing either.")


# ============================================================ smoke
def smoke():
    ok = [True]
    def check(name, cond, note=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {note}")
        ok[0] &= bool(cond)

    print("smoke: renderer + metrics, CPU, no data")
    S, N, B = 32, 24, 3
    raw_dim = 11
    torch.manual_seed(0)
    raw = torch.randn(B, N, raw_dim)

    leg = GaborRendererQ(S, N, chunk=8, qmode=False)
    q = GaborRendererQ(S, N, chunk=8, qmode=True, q=0.6, gist_frac=0.25,
                       octaves=4, f_max=8.0)

    p_l = leg.activate(raw); p_q = q.activate(raw)
    check("legacy sigma range", 0.012 <= float(p_l[2].min()) and
          float(p_l[2].max()) <= 0.152 + 1e-6,
          f"[{float(p_l[2].min()):.4f},{float(p_l[2].max()):.4f}]")
    check("legacy freq range", 1.0 <= float(p_l[4].min()) and
          float(p_l[4].max()) <= 16.0 + 1e-6)

    ng = int(round(0.25 * N))
    check("gist packets carrier-free", float(p_q[4][:, :ng].abs().max()) == 0.0)
    check("gist count correct", int(q.is_gist.sum()) == ng, f"{ng}/{N}")
    car_s, car_f = p_q[2][:, ng:], p_q[4][:, ng:]
    Qv = car_s * car_f
    lo = 0.6 * math.exp(-math.log(2)) - 1e-6
    hi = 0.6 * math.exp(math.log(2)) + 1e-6
    inb = ((Qv >= lo) & (Qv <= hi)) | (car_s <= q.sig_lo + 1e-9) | \
          (car_s >= q.sig_hi - 1e-9)
    check("carrier Q within slack (or sigma clamped)", bool(inb.all()),
          f"Q in [{float(Qv.min()):.2f},{float(Qv.max()):.2f}]")
    check("carrier freq <= f_max", float(car_f.max()) <= 8.0 + 1e-4)
    check("carriers respect octave bands",
          bool((car_f >= q.f_band_lo[0, ng:] - 1e-4).all() and
               (car_f <= q.f_band_hi[0, ng:] + 1e-4).all()))

    out = q(raw)
    check("render shape/range", out.shape == (B, 3, S, S) and
          0.0 <= float(out.min()) and float(out.max()) <= 1.0)

    # metric sanity: adding a synthetic fringe must RAISE moire_index,
    # and blurring must NOT raise it (the rectifier guards against gaming)
    g = torch.rand(B, 3, S, S) * 0.2 + 0.4
    tgt = g.clone()
    yy = torch.linspace(0, 1, S)[None, None, :, None]
    fringe = tgt + 0.12 * torch.sin(2 * math.pi * 5 * yy)
    blur = F.avg_pool2d(F.pad(tgt, (2, 2, 2, 2), mode="reflect"), 5, 1)
    mi_id = float(moire_index(tgt, tgt))
    mi_fr = float(moire_index(fringe.clamp(0, 1), tgt))
    mi_bl = float(moire_index(blur, tgt))
    check("MI(identical) ~ 0", mi_id < 1e-6, f"{mi_id:.2e}")
    check("MI rises on invented fringe", mi_fr > 10 * max(mi_id, 1e-9),
          f"{mi_fr:.4f}")
    check("MI NOT gamed by blur", mi_bl <= mi_fr * 0.5,
          f"blur {mi_bl:.4f} vs fringe {mi_fr:.4f}")

    # --- regression: DEFAULTS must cover the spectrum with no hole ---
    d = GaborRendererQ(128, 512, chunk=64)          # defaults only
    check("defaults: no carrier-free band", int(d.is_gist.sum()) == 0)
    check("defaults: gestalt band reachable", d.f_lo <= 2.0,
          f"f_lo={d.f_lo:.2f} cyc/image (head outline is 1-2)")
    edges = torch.stack([d.f_band_lo[0], d.f_band_hi[0]], 1)
    uniq = torch.unique(edges, dim=0)
    uniq = uniq[uniq[:, 0].argsort()]
    gap = bool((uniq[1:, 0] <= uniq[:-1, 1] * 1.001 + 1e-6).all())
    check("defaults: octave bands contiguous", gap,
          f"{uniq.shape[0]} bands spanning [{float(uniq[0,0]):.2f}, "
          f"{float(uniq[-1,1]):.1f}]")
    torch.manual_seed(1)
    _, _, sgd, _, frd, _ = d.activate(torch.randn(32, 512, raw_dim))
    lowmass = float((frd < 3.0).float().mean())
    check("defaults: real mass in the gestalt band", lowmass > 0.05,
          f"{lowmass:.3f} of packets below 3 cyc/image")
    Qd = (sgd * frd)
    check("defaults: Q stays in the constant-Q window",
          float(Qd.min()) >= 0.29 and float(Qd.max()) <= 1.21,
          f"Q in [{float(Qd.min()):.2f}, {float(Qd.max()):.2f}]")

    bi_l = float(beat_index(*p_l[:2], p_l[2], p_l[3], p_l[4])[1])
    bi_q = float(beat_index(*p_q[:2], p_q[2], p_q[3], p_q[4])[1])
    check("beat_index finite & non-negative", bi_l >= 0 and bi_q >= 0,
          f"legacy {bi_l:.5f}  constQ {bi_q:.5f}")

    # checkpoint round-trip through load_splatvae
    if Encoder is not None:
        import tempfile
        m = SplatVAEQ(S, N, 8, False, qmode=True, q=0.6, gist_frac=0.25,
                      octaves=4, f_max=8.0)
        d = {"sd": m.state_dict(), "image_size": S, "num_packets": N,
             "qmode": True}
        for k in QKEYS[1:]:
            d[k] = getattr(m.ren, k)
        fp = os.path.join(tempfile.mkdtemp(), "m.pt")
        torch.save(d, fp)
        m2, ck2 = load_splatvae(fp, chunk=8)
        z = torch.randn(2, LATENT)
        with torch.no_grad():
            a, b = m.ren(m.dec(z)), m2.ren(m2.dec(z))
        check("checkpoint round-trip identical",
              float((a - b).abs().max()) < 1e-6)
    else:
        print("  SKIP  checkpoint round-trip (splat_trainer3v2 not importable)")

    print("SMOKE " + ("PASS" if ok[0] else "FAIL"))
    return 0 if ok[0] else 1


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir"); ap.add_argument("--out", default="runs/q")
    ap.add_argument("--image_size", type=int, default=96)
    ap.add_argument("--num_packets", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.0005)
    ap.add_argument("--beta_warmup_steps", type=int, default=2000)
    ap.add_argument("--free_bits", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--gamma_floater", type=float, default=0.0)
    ap.add_argument("--sigma_ref", type=float, default=0.03)
    # constant-Q knobs
    ap.add_argument("--q", type=float, default=0.6,
                    help="cycles per envelope sigma (V1-like ~0.5-0.8)")
    ap.add_argument("--q_slack", type=float, default=math.log(2.0),
                    help="natural-log octaves of slack around Q/f")
    ap.add_argument("--gist_frac", type=float, default=0.0,
                help="fraction of carrier-free f=0 Gaussians. DEFAULT 0: a\n                     nonzero value opens a spectral hole in (0, q/sig_hi).")
    ap.add_argument("--octaves", type=int, default=5)
    ap.add_argument("--sig_lo", type=float, default=0.008)
    ap.add_argument("--sig_hi", type=float, default=0.70,
                help="sigma ceiling. Sets the CARRIER FLOOR f_lo = q/sig_hi;\n                     too small and the face-gestalt band is unreachable.")
    ap.add_argument("--gist_sig_hi", type=float, default=0.30)
    ap.add_argument("--f_max", type=float, default=None)
    ap.add_argument("--detail", type=float, default=0.0,
                    help="detail-weighted recon loss strength (try 1.0)")
    # modes
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--legacy", action="store_true",
                    help="train the legacy parameterisation instead")
    args = ap.parse_args()

    if args.smoke:
        raise SystemExit(smoke())
    if args.audit:
        return audit(args.audit)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.data_dir:
        raise SystemExit("--data_dir required (or use --smoke / --audit)")
    cache_dir = args.out; os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"faces_cache_{args.image_size}.npy")
    if not os.path.exists(cache):
        build_cache(args.data_dir, args.image_size, cache)
    data = load_resident(cache, dev)

    if args.compare:
        return compare(args, dev, data)
    m = train_one(args, dev, data, not args.legacy,
                  "legacy" if args.legacy else "constQ", args.steps)
    mi, ps, bi, hq = evaluate(m, data, dev, args)
    print(f"\nfinal: moire_index {mi:.4f}  PSNR {ps:.2f}  "
          f"beat_index {bi:.5f}  frac(Q>1.5) {hq:.3f}")


if __name__ == "__main__":
    main()
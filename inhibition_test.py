"""
inhibition_test.py — what does V1-style divisive lateral inhibition actually
do to a splat renderer?  torch, CPU, executed.

THE PROPOSAL (from the V1 comparison)
  Build W_kl from spatial overlap x frequency match x co-orientation, then
      A~_k = A_k / (1 + gamma * sum_l W_kl A_l)
  applied to the packet coefficients before rendering, phase preserved.

THE STRUCTURAL OBJECTION, WHICH IS WHAT THIS TESTS
  In V1, normalization earns its keep because the drive arrives from the
  world with an enormous dynamic range and neurons have bounded firing
  rates. In this renderer the coefficients are ALREADY the free, tanh-
  bounded output of a learned decoder. So render-time normalization does not
  add a capability the decoder lacked — it REMOVES one: the map from raw
  coefficients to images stops being linear and stops being onto. Anything
  normalization would produce, the decoder could already have emitted
  directly, if that helped the loss.
  Prediction: render-time inhibition is a CAPACITY COST, growing with gamma.

REGISTERED BEFORE THE RUN
  IN1 [capacity] best achievable PSNR with inhibition <= without, at every
      gamma, on fixed geometry. Reported as a curve, not a pass/fail —
      the question is how much it costs, not whether.
  IN2 [does it even fire?] mean gain must fall visibly below 1 at the tested
      gammas. A gamma that costs nothing because it does nothing is not a
      null result, it is an untested one.
  IN3 [moire] does inhibition lower moire_index on a constant-Q basis? The
      honest prior is no measurable room: the shipped constant-Q arm already
      measures 0.0003 against legacy's 0.0132.
  IN4 [the frequency-window bug] the proposed code uses
          freq_ov = exp(-(f_k - f_l)^2 / 2)
      with f in cycles/image. On a dyadic ladder that is NOT scale
      invariant: neighbouring octaves at the bottom (1 vs 2, df=1) inhibit
      strongly while neighbouring octaves at the top (16 vs 32, df=16) do
      not interact at all. Measured here against the log-frequency version
      df = log2(f_k/f_l), which is the constant-Q-consistent form.

Do not hype. Do not lie. Just show.
"""
import math
import torch

torch.manual_seed(0)
S, N, Q = 96, 256, 0.6
OCT, F0 = 5, 1.0
dev = torch.device("cpu")
gy, gx = torch.meshgrid(torch.linspace(0, 1, S), torch.linspace(0, 1, S),
                        indexing="ij")


# ---------------------------------------------------------------- basis
g = torch.Generator().manual_seed(1)
band = torch.arange(N) % OCT
freq = F0 * 2.0 ** band.float() * (1 + 0.5 * torch.rand(N, generator=g))
sigma = Q / freq
px = torch.rand(N, generator=g) * 0.84 + 0.08
py = torch.rand(N, generator=g) * 0.84 + 0.08
theta = torch.rand(N, generator=g) * math.pi

dx = gx[None] - px[:, None, None]
dy = gy[None] - py[:, None, None]
u = dx * torch.cos(theta)[:, None, None] + dy * torch.sin(theta)[:, None, None]
env = torch.exp(-(dx * dx + dy * dy) / (2 * sigma[:, None, None] ** 2))
BASIS = torch.stack([env * torch.cos(2 * math.pi * freq[:, None, None] * u),
                     env * torch.sin(2 * math.pi * freq[:, None, None] * u)], 1)
BASIS = BASIS.reshape(2 * N, S * S)                      # (2N, P)


# ---------------------------------------------------------------- target
def synthetic_face():
    im = torch.zeros(S, S)
    im += .55 * torch.exp(-(((gx - .5) / .30) ** 2 + ((gy - .52) / .38) ** 2) ** 3)
    for sx in (-1, 1):
        d2 = (gx - (.5 + sx * .13)) ** 2 + (gy - .42) ** 2
        im -= .45 * torch.exp(-d2 / (2 * .035 ** 2))
        im -= .35 * torch.exp(-d2 / (2 * .013 ** 2))
        im += .18 * torch.cos(2 * math.pi * 30 * torch.sqrt(d2 + 1e-9)) * \
            torch.exp(-d2 / (2 * .022 ** 2))
    im -= .30 * torch.exp(-((gx - .5) ** 2 / (2 * .09 ** 2)
                            + (gy - .70) ** 2 / (2 * .012 ** 2)))
    n = torch.randn(S, S, generator=g)
    k = torch.fft.fftfreq(S)[:, None] ** 2 + torch.fft.fftfreq(S)[None, :] ** 2
    t = torch.fft.ifft2(torch.fft.fft2(n) / (k + 1e-3) ** .55).real
    return im + .12 * t / t.std() - im.mean()


TGT = synthetic_face()
RNG_ = float(TGT.max() - TGT.min())


# ---------------------------------------------------------------- inhibition
def weights(log_freq):
    d2 = (px[:, None] - px[None]) ** 2 + (py[:, None] - py[None]) ** 2
    s2 = (sigma[:, None] ** 2 + sigma[None] ** 2).clamp(min=1e-9)
    spat = torch.exp(-d2 / (2 * s2))
    if log_freq:                       # constant-Q-consistent
        df = torch.log2(freq[:, None] / freq[None])
        fo = torch.exp(-df ** 2 / (2 * 1.0 ** 2))
    else:                              # the proposed code, as written
        fo = torch.exp(-(freq[:, None] - freq[None]) ** 2 / 2.0)
    orient = torch.cos(theta[:, None] - theta[None]) ** 2
    W = spat * fo * orient
    W.fill_diagonal_(0.0)
    return W


def render(raw, W, gamma):
    """raw: (N,2) pre-inhibition phasor. Returns image and mean gain."""
    A = torch.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2 + 1e-6)
    if gamma > 0:
        gain = 1.0 / (1.0 + gamma * (W @ A))
    else:
        gain = torch.ones_like(A)
    c = raw * (gain / A)[:, None] * A[:, None]      # scale magnitude, keep phase
    return (c.reshape(-1) @ BASIS).reshape(S, S), gain.mean()


def psnr(img):
    return 10 * math.log10(RNG_ ** 2 / float(((img - TGT) ** 2).mean()))


def radial_moire(img):
    """rectified excess mid-band energy vs the target — the shipped metric."""
    def rp(x):
        x = x - x.mean()
        F2 = torch.fft.rfft2(x, norm="ortho").abs() ** 2
        H, Wd = F2.shape
        fy = torch.fft.fftfreq(H)[:, None] * H
        fx = torch.arange(Wd)[None, :].float()
        r = (fy ** 2 + fx ** 2).sqrt().round().long().clamp(max=H // 2)
        R = H // 2 + 1
        o = torch.zeros(R).index_add_(0, r.reshape(-1), F2.reshape(-1))
        c = torch.zeros(R).index_add_(0, r.reshape(-1),
                                      torch.ones(H * Wd))
        return o / c.clamp(min=1)
    Pr, Pt = rp(img), rp(TGT)
    lo, hi = 2, 2 * (len(Pr) - 1) // 8
    return float((Pr[lo:hi] - Pt[lo:hi]).clamp(min=0).sum()
                 / Pt[lo:hi].sum().clamp(min=1e-12))


def best_fit(W, gamma, steps=3000):
    raw = (0.01 * torch.randn(N, 2, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        img, _ = render(raw, W, gamma)
        loss = ((img - TGT) ** 2).mean()
        loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        img, mg = render(raw, W, gamma)
    return psnr(img), float(mg), radial_moire(img)


# ---------------------------------------------------------------- run
lin = (BASIS.T @ torch.linalg.lstsq(BASIS.T, TGT.reshape(-1)).solution
       ).reshape(S, S)
print(f"basis: N={N} constant-Q packets, {S}x{S}, {OCT} octaves")
print(f"closed-form least squares (no inhibition): PSNR {psnr(lin):.2f} dB, "
      f"moire {radial_moire(lin):.4f}")
print(f"\n{'gamma':>7} {'freq window':>12} {'PSNR':>8} {'vs free':>9} "
      f"{'mean gain':>10} {'moire':>8}")
base = None
for lf in (True, False):
    W = weights(lf)
    for gamma in (0.0, 0.05, 0.2, 1.0, 5.0):
        p, mg, mo = best_fit(W, gamma)
        if base is None:
            base = p
        tag = "log2(f)" if lf else "linear f"
        print(f"{gamma:7.2f} {tag:>12} {p:8.2f} {p-base:+9.2f} "
              f"{mg:10.4f} {mo:8.4f}")
        if gamma == 0.0 and not lf:
            pass

# IN4: how differently do the two frequency windows couple the octaves?
Wl, Wf = weights(True), weights(False)
print("\nIN4  mean inhibition weight between ADJACENT octave bands:")
print(f"{'bands':>10} {'log2(f) window':>16} {'linear f window':>17}")
for b in range(OCT - 1):
    m = (band[:, None] == b) & (band[None] == b + 1)
    print(f"{2**b:4d}v{2**(b+1):<4d} {float(Wl[m].mean()):16.5f} "
          f"{float(Wf[m].mean()):17.5f}")

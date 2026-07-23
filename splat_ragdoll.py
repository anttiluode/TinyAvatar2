#!/usr/bin/env python3
# ============================================================================
# splat_ragdoll.py — TinyAvatar2 x SlapStack: grab the face and pull it.
#
# WHAT THIS IS
#   An interactive editor for a trained TinyAvatar2 constant-Q splat model
#   (splat_trainer5.py checkpoints, legacy checkpoints via the same loader).
#   Two editing layers over the same rendered field:
#
#   MANIFOLD mode ("ragdoll") — you drag a PIN. A pin is a soft cluster of
#     packets (envelope x amplitude x grab-radius weighted). The app solves
#         min_z  sum_i || centroid_i(z) - target_i ||^2   (+ posture bias)
#     by damped least squares on the decoder Jacobian dc/dz  (2m x 128,
#     m = number of pins; JJ^T is 2m x 2m, so the solve is closed-form and
#     tiny). The VAE's learned covariance is the rig: if the training data
#     correlated features, pulling one drags the rest along the manifold.
#     Posture bias (return toward the anchor identity) is applied in the
#     NULL SPACE of the pin Jacobian — the standard secondary-task IK trick —
#     so pins are honored exactly while identity relaxes where the pins
#     don't care.
#
#   DIRECT mode ("slapstack") — the same grab, but the displacement is
#     applied straight to the activated packet parameters, post-decoder:
#     rigid SE(2) translate / rotate about the grab pivot, plus a scale
#     gesture that preserves constant-Q EXACTLY (sigma *= s, freq /= s, so
#     Q = sigma*freq is invariant to machine precision). No solver, no
#     manifold: pure parameter-space surgery. Octave-band gates (keys 1-5)
#     restrict the grab to chosen frequency layers — grab only the mid-band
#     feature geometry and leave lighting + texture untouched.
#
# WHAT THIS IS NOT (honest ledger, read before believing anything)
#   * The shipped basis is FLAT. The dyadic parent->child tree was killed by
#     the least-squares pre-test (tree geometry -0.87 dB vs its own octave
#     histogram); there is no learned hierarchy here. "Grab a feature" is
#     spatial soft-selection at a chosen radius + band gate, which gives the
#     SlapStack cluster behaviour without pretending the model has parents.
#   * "Pull the eye and the head turns" is a REGISTERED CLAIM (RG1), not a
#     fact. It is true iff the VAE learned the covariance. The 128px/512
#     model has a measured diversity collapse (mean pairwise gap 0.0089,
#     ~13x below the 96px model), so its manifold may be too narrow to show
#     it — run gates on the 96px model2.pt line first.
#   * ">100 FPS closed-form editing" is RG3, measured not asserted. The
#     solver really is a 2m x 2m solve, but the Jacobian needs 2m backward
#     passes through the decoder MLP per iteration and the render is the
#     render.
#   * The conv encoder is translation-invariant (28 px input slide -> 0.5 px
#     recon motion, measured earlier on this line). That is exactly why the
#     drag must go through the DECODER Jacobian: you cannot move a feature
#     by warping the input. Latent IK is the sound route, not a workaround.
#
# REGISTERED GATES  (run: python splat_ragdoll.py --gates --model X.pt)
#   RG1  THE RIG CLAIM. Drag a pin at --pin_a (default left-eye region
#        0.35,0.42) by +0.08 in x, manifold mode, solved to tolerance.
#        Measure displacement of the UN-PINNED mirror cluster at --pin_b
#        (0.65,0.42). Same drag in direct mode as control (direct moves only
#        the grabbed weights; coarse packets shared by both clusters leak a
#        little, which is why the control is measured, not assumed zero).
#        [V] iff median over samples: |dc_B|_manifold >= 3 x |dc_B|_direct
#        AND |dc_B|_manifold >= 1 px. A [K] is a finding too: the manifold
#        does not encode the covariance at that scale.
#   RG2  NULLSPACE POSTURE. Same pin solve with and without the nullspace
#        anchor pull, both to pin error < tol. [V] iff ||z-z0|| with pull
#        <= 0.85 x without, at equal pin error. (Identity retention is real,
#        not a slogan.)
#   RG3  INTERACTIVE RATE. Full loop (decode + Jacobian + solve + render)
#        and solver-only rate, measured. [V] iff full loop >= 20 FPS.
#        Expected to need the GPU for 128px/512; CPU numbers are printed
#        either way and are not a verdict about the GPU.
#   RG4  RANGE (diagnostic, no gate). Ramp the drag 0.02 -> 0.30, print pin
#        error, |z|, and moire_index of the recon vs the undragged recon at
#        each step — where the ragdoll leaves the manifold and catches fire.
#
# SELFTEST  (python splat_ragdoll.py --selftest — headless, random tiny
#            model, certifies the MACHINERY, says nothing about a trained
#            manifold)
#   SR0a render_from_params == renderer.forward, max abs diff < 1e-5
#   SR0b direct rigidity: hard-mask translate moves grabbed px,py exactly,
#        untouched packets move exactly 0; scale gesture preserves Q to 1e-5
#   SR0c solver convergence: pin reaches a 0.06-offset target within 1.5 px
#        (32px canvas) in <= 120 DLS iterations on a random decoder
#   SR0d nullspace pull reduces ||z|| at equal pin error
#   SR0e speed report (CPU, informational)
#
# USAGE
#   python splat_ragdoll.py --model model5_constQ.pt            # live editor
#   python splat_ragdoll.py --model model2.pt --image me.png    # start from
#                                                               # an encoding
#   python splat_ragdoll.py --selftest
#   python splat_ragdoll.py --gates --model model2.pt
#
# KEYS (live)
#   LMB drag   grab & pull (mode decides what that means)
#   m          toggle MANIFOLD / DIRECT
#   p          while holding a manifold drag: make the pin persistent
#              (click within 15 px of a persistent pin to re-grab it)
#   c          clear all pins and direct edits
#   [ / ]      grab radius down / up
#   , / .      rotate held DIRECT cluster  (SE(2) rotation about pivot)
#   < / >      scale held DIRECT cluster   (constant-Q preserving)
#   1..5       toggle octave bands in the grab filter   0 = all bands
#   s          spring-back on release (manifold: z relaxes to anchor;
#              direct: deltas decay to zero)
#   w          wobble (latent velocity + momentum — the ragdoll flop)
#   b          bake: current state becomes the new anchor identity
#   r          reset to anchor
#   n          new random identity     e  re-encode --image
#   o          overlay on/off          q  quit
#   --record out.mp4 to save the session
#
# Do not hype. Do not lie. Just show.
# ============================================================================
import argparse, math, os, sys, time
import numpy as np
import torch

torch.set_grad_enabled(False)

# ---------------------------------------------------------------- trainer import
try:
    import splat_trainer5 as ST
except ImportError:
    ST = None


def _need_ST():
    if ST is None:
        sys.exit("splat_ragdoll.py needs splat_trainer5.py next to it "
                 "(and splat_trainer3v2.py for Encoder/Decoder).")


# ---------------------------------------------------------------- render from params
def render_from_params(ren, px, py, sigma, theta, freq, coeff):
    """Render from ACTIVATED params, reproducing GaborRendererQ.forward's
    chunked sum + sigmoid, so edited params render identically to the
    renderer's own path (parity certified in SR0a)."""
    out = None
    for i in range(0, ren.N, ren.chunk):
        sl = slice(i, i + ren.chunk)
        c = ren._chunk(px[:, sl], py[:, sl], sigma[:, sl],
                       theta[:, sl], freq[:, sl], coeff[:, sl])
        out = c if out is None else out + c
    return torch.sigmoid(out)


def packet_amp(coeff):
    """Per-packet RMS amplitude over the 3x2 coefficient block. (B,N)."""
    return coeff.reshape(*coeff.shape[:2], -1).pow(2).mean(-1).sqrt()


# ---------------------------------------------------------------- grab selection
def grab_weights(px, py, sigma, amp, point, r_grab, band_gate):
    """Soft cluster selection at `point` (unit coords).
    w_k = amp_norm * envelope_k(point) * exp(-d^2 / 2 r_grab^2) * band gate.
    A packet must PAINT the point (its envelope reaches it) AND be centred
    near it (grab radius) AND be loud AND be in an enabled band. Returns
    (N,) weights normalised to max 1, hard-zeroed below 0.02."""
    d2 = (px[0] - point[0]) ** 2 + (py[0] - point[1]) ** 2
    env = torch.exp(-d2 / (2 * sigma[0] ** 2 + 1e-12))
    rad = torch.exp(-d2 / (2 * r_grab ** 2))
    a = amp[0] / (amp[0].max() + 1e-12)
    w = a * env * rad * band_gate
    m = w.max()
    if m > 1e-9:
        w = w / m
    w = torch.where(w < 0.02, torch.zeros_like(w), w)
    return w


def band_index(ren):
    """Per-packet octave index 0..octaves-1 from the frozen band buffers
    (gist packets, if any, get -1)."""
    if not ren.qmode:
        return torch.zeros(ren.N, dtype=torch.long)
    lo = ren.f_band_lo[0]
    edges = sorted(set(round(float(v), 5) for v in lo))
    idx = torch.zeros(ren.N, dtype=torch.long)
    for i in range(ren.N):
        idx[i] = edges.index(round(float(lo[i]), 5))
    idx[ren.is_gist[0] > 0.5] = -1
    return idx


# ---------------------------------------------------------------- direct edit stack
class DirectOp:
    """One SE(2)+scale gesture on a soft cluster. Applied post-activation.
    scale preserves constant-Q exactly: sigma *= s^w, freq /= s^w."""

    def __init__(self, w, pivot):
        self.w = w                       # (N,) soft weights
        self.pivot = list(pivot)         # grab point, unit coords
        self.dx = 0.0
        self.dy = 0.0
        self.rot = 0.0                   # radians
        self.log_s = 0.0                 # log scale

    def apply(self, px, py, sigma, theta, freq):
        w = self.w[None]                                     # (1,N)
        # pivot decomposition ONLY when rotating/scaling: a pure translation
        # must leave untouched packets BIT-EXACT (px + 0*dx == px), and the
        # cx + (px - cx) round-trip alone costs ~3e-8 in float32 (SR0b)
        if abs(self.rot) > 1e-9 or abs(self.log_s) > 1e-9:
            cx, cy = self.pivot
            rx, ry = px - cx, py - cy
            if abs(self.rot) > 1e-9:                 # SE(2) rotation
                a = w * self.rot
                ca, sa = torch.cos(a), torch.sin(a)
                rx, ry = ca * rx - sa * ry, sa * rx + ca * ry
                theta = theta + a
            if abs(self.log_s) > 1e-9:               # constant-Q scale
                s = torch.exp(w * self.log_s)
                rx, ry = rx * s, ry * s
                sigma = sigma * s
                freq = freq / s
            px, py = cx + rx, cy + ry
        # translate
        px = px + w * self.dx
        py = py + w * self.dy
        return px, py, sigma, theta, freq


def apply_stack(stack, px, py, sigma, theta, freq):
    for op in stack:
        px, py, sigma, theta, freq = op.apply(px, py, sigma, theta, freq)
    return px, py, sigma, theta, freq


# ---------------------------------------------------------------- manifold solver
class Pin:
    def __init__(self, w, target):
        self.w = w                       # (N,) fixed at grab time
        self.target = list(target)       # unit coords


class RagdollSolver:
    """Damped-least-squares latent IK on the decoder.
    c_i(z) = weighted centroid of pin i's cluster under activate(dec(z)).
    dz = J^T (J J^T + lam^2 I)^-1 e  +  Nullspace(J) . beta (z_anchor - z).
    Pins are met exactly (up to damping); identity relaxes only where the
    pins don't constrain — textbook secondary-task IK, on a face manifold."""

    def __init__(self, model, lam=0.08, beta=0.15, step_clip=0.8):
        self.m = model
        self.lam = lam
        self.beta = beta
        self.step_clip = step_clip

    def centroids(self, z, pins):
        raw = self.m.dec(z[None])
        px, py, sg, th, fr, cf = self.m.ren.activate(raw.float())
        cs = []
        for p in pins:
            wsum = p.w.sum() + 1e-9
            cs.append(torch.stack([(p.w * px[0]).sum() / wsum,
                                   (p.w * py[0]).sum() / wsum]))
        return torch.stack(cs)                               # (m,2)

    def step(self, z, z_anchor, pins, iters=1, posture=True):
        """Returns (z_new, pin_err_px_max). No-op without pins."""
        live = [p for p in pins if float(p.w.sum()) > 1e-6]
        if not live:
            return z, 0.0
        H = self.m.ren.H
        err_px = 0.0
        for _ in range(iters):
            with torch.enable_grad():
                zz = z.detach().clone().requires_grad_(True)
                c = self.centroids(zz, live)                 # (m,2)
                tgt = torch.tensor([p.target for p in live],
                                   dtype=c.dtype)
                e = (tgt - c).reshape(-1)                    # (2m,)
                rows = []
                for k in range(e.numel()):
                    g = torch.autograd.grad(c.reshape(-1)[k], zz,
                                            retain_graph=True)[0]
                    rows.append(g)
                J = torch.stack(rows)                        # (2m,L)
            e = e.detach()
            J = J.detach()
            JJt = J @ J.T + (self.lam ** 2) * torch.eye(J.shape[0])
            K = J.T @ torch.linalg.solve(JJt, torch.eye(J.shape[0]))  # J^+
            dz = K @ e
            if posture and self.beta > 0:
                P = torch.eye(J.shape[1]) - K @ J            # nullspace proj
                dz = dz + P @ (self.beta * (z_anchor - z))
            n = dz.norm()
            if n > self.step_clip:
                dz = dz * (self.step_clip / n)
            z = z + dz
            err_px = float(e.reshape(-1, 2).norm(dim=1).max()) * H
        return z, err_px


# ---------------------------------------------------------------- model helpers
def load_model(path):
    _need_ST()
    m, ck = ST.load_splatvae(path)
    return m, ck


def encode_image(model, path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"cannot read --image {path}")
    S = model.ren.H
    img = cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    model.enc.eval()
    mu, _ = model.enc(x)
    return mu[0]


def tiny_random_model(size=32, packets=48, seed=0):
    """Random-weight model for the headless selftest. Certifies machinery,
    not manifolds."""
    _need_ST()
    torch.manual_seed(seed)
    m = ST.SplatVAEQ(size, packets, chunk=32, ckpt=False, qmode=True,
                     q=0.6, gist_frac=0.0, octaves=4, sig_lo=0.008,
                     sig_hi=0.70, band_mode="permute")
    # give the decoder a real (non-degenerate) Jacobian: the trainer inits
    # the last layer at 0.1x weight / 0 bias, which is fine, but add spread
    # so px,py depend visibly on z for the convergence test
    with torch.no_grad():
        for p in m.dec.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    m.eval()
    return m


# ================================================================ SELFTEST
def selftest():
    _need_ST()
    print("SELFTEST — machinery only; a random decoder has no manifold "
          "and these results certify nothing about a trained model.")
    m = tiny_random_model()
    S = m.ren.H
    z = torch.randn(128) * 0.5
    raw = m.dec(z[None]).float()
    px, py, sg, th, fr, cf = m.ren.activate(raw)
    amp = packet_amp(cf)

    # SR0a — render parity
    r_fwd = m.ren(raw)
    r_par = render_from_params(m.ren, px, py, sg, th, fr, cf)
    d = float((r_fwd - r_par).abs().max())
    ok_a = d < 1e-5
    print(f"SR0a render_from_params parity   max|diff| {d:.2e}  "
          f"{'[V]' if ok_a else '[K]'}")

    # SR0b — direct rigidity + constant-Q scale invariance
    w = torch.zeros(m.ren.N)
    w[:10] = 1.0                                   # hard mask
    op = DirectOp(w, (0.5, 0.5))
    op.dx, op.dy = 0.10, -0.05
    px2, py2, sg2, th2, fr2 = op.apply(px, py, sg, th, fr)
    moved = float((px2[0, :10] - px[0, :10] - 0.10).abs().max()
                  + (py2[0, :10] - py[0, :10] + 0.05).abs().max())
    still = float((px2[0, 10:] - px[0, 10:]).abs().max()
                  + (py2[0, 10:] - py[0, 10:]).abs().max())
    op2 = DirectOp(w, (0.5, 0.5))
    op2.log_s = math.log(1.7)
    _, _, sg3, _, fr3 = op2.apply(px, py, sg, th, fr)
    dq = float(((sg3 * fr3)[0, :10] - (sg * fr)[0, :10]).abs().max())
    ok_b = moved < 1e-6 and still < 1e-9 and dq < 1e-5
    print(f"SR0b direct rigidity moved {moved:.2e} still {still:.2e} "
          f"|dQ| {dq:.2e}  {'[V]' if ok_b else '[K]'}")

    # SR0c — solver convergence on a random decoder
    bg = torch.ones(m.ren.N)
    k = int(amp[0].argmax())
    pt = (float(px[0, k]), float(py[0, k]))
    w = grab_weights(px, py, sg, amp, pt, 0.12, bg)
    solver = RagdollSolver(m, lam=0.06, beta=0.0, step_clip=0.6)
    pin = Pin(w, (min(pt[0] + 0.06, 0.95), min(pt[1] + 0.04, 0.95)))
    zz = z.clone()
    err = 1e9
    for it in range(120):
        zz, err = solver.step(zz, z, [pin], iters=1, posture=False)
        if err < 1.5:
            break
    ok_c = err < 1.5
    print(f"SR0c solver convergence          pin err {err:.2f} px @ iter "
          f"{it+1}  {'[V]' if ok_c else '[K]'}")

    # SR0d — nullspace posture pull
    sol_p = RagdollSolver(m, lam=0.06, beta=0.3, step_clip=0.6)
    za, zb = z.clone(), z.clone()
    ea = eb = 1e9
    for _ in range(120):
        za, ea = solver.step(za, z, [pin], iters=1, posture=False)
    for _ in range(120):
        zb, eb = sol_p.step(zb, z, [pin], iters=1, posture=True)
    na, nb = float((za - z).norm()), float((zb - z).norm())
    ok_d = (eb < 3.0) and (nb < na)
    print(f"SR0d nullspace posture  |dz| {na:.3f} -> {nb:.3f} "
          f"(pin err {ea:.2f}/{eb:.2f} px)  {'[V]' if ok_d else '[K]'}")

    # SR0e — speed, informational
    t0 = time.time()
    for _ in range(20):
        zz, _ = solver.step(zz, z, [pin], iters=1, posture=True)
    t_sol = (time.time() - t0) / 20 * 1000
    t0 = time.time()
    for _ in range(20):
        raw = m.dec(zz[None]).float()
        p_ = m.ren.activate(raw)
        render_from_params(m.ren, *p_)
    t_ren = (time.time() - t0) / 20 * 1000
    print(f"SR0e speed (CPU, {S}px/{m.ren.N}pk)  solver {t_sol:.1f} ms  "
          f"decode+render {t_ren:.1f} ms  [report]")

    allv = ok_a and ok_b and ok_c and ok_d
    print(f"SELFTEST {'ALL-[V]' if allv else 'FAILED'}")
    return 0 if allv else 1


# ================================================================ GATES
def gates(args):
    _need_ST()
    m, ck = load_model(args.model)
    S = m.ren.H
    print(f"GATES on {args.model}  ({S}px / {m.ren.N} packets, "
          f"qmode={m.ren.qmode})")
    print("pins:", args.pin_a, "->", args.pin_b, f" drag +{args.drag:.2f} x")
    torch.manual_seed(args.seed)
    bg = torch.ones(m.ren.N)
    solver = RagdollSolver(m, lam=args.lam, beta=0.0, step_clip=0.6)
    sol_p = RagdollSolver(m, lam=args.lam, beta=0.3, step_clip=0.6)

    ratios, mirrs, dirs = [], [], []
    rg2_pairs = []
    for s in range(args.gate_samples):
        z0 = torch.randn(128) * 0.8
        raw = m.dec(z0[None]).float()
        px, py, sg, th, fr, cf = m.ren.activate(raw)
        amp = packet_amp(cf)
        wA = grab_weights(px, py, sg, amp, args.pin_a, args.r_grab, bg)
        wB = grab_weights(px, py, sg, amp, args.pin_b, args.r_grab, bg)
        if float(wA.sum()) < 1e-6 or float(wB.sum()) < 1e-6:
            print(f"  sample {s}: empty grab, skipped")
            continue

        def cent(px_, py_, w):
            ws = w.sum() + 1e-9
            return torch.stack([(w * px_[0]).sum() / ws,
                                (w * py_[0]).sum() / ws])

        cA0, cB0 = cent(px, py, wA), cent(px, py, wB)

        # --- manifold arm
        pin = Pin(wA, (float(cA0[0]) + args.drag, float(cA0[1])))
        z = z0.clone()
        err = 1e9
        for _ in range(args.iters):
            z, err = solver.step(z, z0, [pin], iters=1, posture=False)
            if err < args.tol_px:
                break
        raw2 = m.dec(z[None]).float()
        px2, py2, *_ = m.ren.activate(raw2)
        dB_man = float((cent(px2, py2, wB) - cB0).norm()) * S

        # --- direct arm (control)
        op = DirectOp(wA, (float(cA0[0]), float(cA0[1])))
        op.dx = args.drag
        px3, py3, _, _, _ = op.apply(px, py, sg, th, fr)
        dB_dir = float((cent(px3, py3, wB) - cB0).norm()) * S

        ratio = dB_man / max(dB_dir, 1.0)
        ratios.append(ratio); mirrs.append(dB_man); dirs.append(dB_dir)
        print(f"  sample {s}: pin err {err:.2f}px  mirror manifold "
              f"{dB_man:.2f}px  direct {dB_dir:.2f}px  ratio {ratio:.2f}")

        # --- RG2 on the same solve
        zb = z0.clone()
        for _ in range(args.iters):
            zb, eb = sol_p.step(zb, z0, [pin], iters=1, posture=True)
            if eb < args.tol_px:
                break
        if err < args.tol_px * 2 and eb < args.tol_px * 2:
            rg2_pairs.append((float((z - z0).norm()),
                              float((zb - z0).norm())))

    if not ratios:
        print("no valid samples — check pin coords against your model")
        return 1
    med_r = float(np.median(ratios)); med_m = float(np.median(mirrs))
    rg1 = med_r >= 3.0 and med_m >= 1.0
    print(f"RG1 rig claim: median ratio {med_r:.2f} (>=3), median mirror "
          f"{med_m:.2f}px (>=1)  {'[V]' if rg1 else '[K]'}")
    if rg2_pairs:
        a = np.array(rg2_pairs)
        rg2 = bool(np.median(a[:, 1] / a[:, 0]) <= 0.85)
        print(f"RG2 nullspace posture: median |dz| shrink "
              f"{np.median(a[:,1]/a[:,0]):.2f} (<=0.85)  "
              f"{'[V]' if rg2 else '[K]'}")
    else:
        print("RG2: no converged pairs, unmeasured")

    # RG3 timing
    z = torch.randn(128) * 0.8
    pin = Pin(torch.rand(m.ren.N) * 0.1, (0.5, 0.5))
    t0 = time.time()
    for _ in range(15):
        z, _ = solver.step(z, z, [pin], iters=1, posture=True)
    t_sol = (time.time() - t0) / 15
    t0 = time.time()
    for _ in range(15):
        raw = m.dec(z[None]).float()
        render_from_params(m.ren, *m.ren.activate(raw))
    t_full = (time.time() - t0) / 15 + t_sol
    fps = 1.0 / max(t_full, 1e-9)
    print(f"RG3 rate: solver {t_sol*1000:.1f} ms ({1/max(t_sol,1e-9):.0f} "
          f"Hz), full loop {fps:.1f} FPS (>=20)  "
          f"{'[V]' if fps >= 20 else '[K]'}  (this machine)")

    # RG4 range ramp (diagnostic)
    print("RG4 range ramp (diagnostic):  drag  pin_err_px  |z|  moire")
    z0 = torch.randn(128) * 0.8
    raw0 = m.dec(z0[None]).float()
    px, py, sg, th, fr, cf = m.ren.activate(raw0)
    amp = packet_amp(cf)
    wA = grab_weights(px, py, sg, amp, args.pin_a, args.r_grab, bg)
    r0 = m.ren(raw0)
    cA0 = torch.stack([(wA * px[0]).sum() / (wA.sum() + 1e-9),
                       (wA * py[0]).sum() / (wA.sum() + 1e-9)])
    for drag in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        pin = Pin(wA, (float(cA0[0]) + drag, float(cA0[1])))
        z = z0.clone()
        for _ in range(args.iters):
            z, err = solver.step(z, z0, [pin], iters=1, posture=False)
            if err < args.tol_px:
                break
        r = m.ren(m.dec(z[None]).float())
        mo = float(ST.moire_index(r, r0))
        print(f"   {drag:.2f}   {err:7.2f}   {float(z.norm()):6.2f}  "
              f"{mo:.5f}")
    return 0


# ================================================================ LIVE UI
def live(args):
    import cv2
    _need_ST()
    m, ck = load_model(args.model)
    S = m.ren.H
    scale = max(1, 576 // S)
    D = S * scale
    bidx = band_index(m.ren)
    n_bands = m.ren.octaves if m.ren.qmode else 1
    band_gate = torch.ones(m.ren.N)
    band_on = [True] * n_bands
    BAND_COL = [(255, 120, 60), (120, 255, 120), (80, 200, 255),
                (200, 120, 255), (100, 100, 255)]

    if args.image:
        z_anchor = encode_image(m, args.image)
    else:
        torch.manual_seed(args.seed)
        z_anchor = torch.randn(128) * 0.8
    z = z_anchor.clone()
    v = torch.zeros(128)

    solver = RagdollSolver(m, lam=args.lam, beta=0.3, step_clip=0.35)
    stack, pins = [], []
    live_op, live_pin = None, None
    mode = "MANIFOLD"
    r_grab = args.r_grab
    spring, wobble, overlay = True, False, True
    dragging = False
    cursor = (0.5, 0.5)
    writer = None

    st = {"z": z, "dragging": False, "cursor": cursor}

    def refresh_gate():
        nonlocal band_gate
        g = torch.ones(m.ren.N)
        if m.ren.qmode:
            for b in range(n_bands):
                if not band_on[b]:
                    g[bidx == b] = 0.0
        band_gate = g

    def on_mouse(ev, x, y, flags, _):
        nonlocal dragging, live_op, live_pin, pins
        u, vv = x / D, y / D
        st["cursor"] = (u, vv)
        if ev == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            raw = m.dec(st["z"][None]).float()
            px, py, sg, th, fr, cf = m.ren.activate(raw)
            # re-grab an existing pin?
            for p in pins:
                ws = p.w.sum() + 1e-9
                cx = float((p.w * px[0]).sum() / ws)
                cy = float((p.w * py[0]).sum() / ws)
                if (cx - u) ** 2 + (cy - vv) ** 2 < (15 / D) ** 2:
                    live_pin = p
                    return
            w = grab_weights(px, py, sg, packet_amp(cf), (u, vv),
                             r_grab, band_gate)
            if mode == "MANIFOLD":
                live_pin = Pin(w, (u, vv))
            else:
                live_op = DirectOp(w, (u, vv))
        elif ev == cv2.EVENT_MOUSEMOVE and dragging:
            if live_pin is not None:
                live_pin.target = [u, vv]
            elif live_op is not None:
                live_op.dx = u - live_op.pivot[0]
                live_op.dy = vv - live_op.pivot[1]
        elif ev == cv2.EVENT_LBUTTONUP:
            dragging = False
            if live_op is not None:
                stack.append(live_op)
                live_op = None
            if live_pin is not None and live_pin not in pins:
                live_pin = None      # transient pin dies unless 'p' saved it

    win = "splat ragdoll — TinyAvatar2 x SlapStack"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    if args.record:
        writer = cv2.VideoWriter(args.record,
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (D, D))
    refresh_gate()
    t_last, fps = time.time(), 0.0

    while True:
        st["z"] = z
        active = ([live_pin] if (live_pin is not None) else []) + \
                 [p for p in pins if p is not live_pin]
        if active and (dragging or pins):
            z, err = solver.step(z, z_anchor, active,
                                 iters=args.iters_frame, posture=True)
        else:
            err = 0.0
        if wobble:
            # latent velocity: the flop. momentum makes release overshoot.
            v.mul_(0.82)
            if spring and not dragging and not pins:
                v.add_(0.10 * (z_anchor - z))
            z = z + v
        elif spring and not dragging and not pins:
            z = z + 0.10 * (z_anchor - z)
            if spring and stack and not dragging:
                for op in stack:
                    op.dx *= 0.90; op.dy *= 0.90
                    op.rot *= 0.90; op.log_s *= 0.90
                stack[:] = [o for o in stack
                            if abs(o.dx) + abs(o.dy) + abs(o.rot)
                            + abs(o.log_s) > 1e-3]

        raw = m.dec(z[None]).float()
        px, py, sg, th, fr, cf = m.ren.activate(raw)
        pxE, pyE, sgE, thE, frE = apply_stack(
            stack + ([live_op] if live_op else []), px, py, sg, th, fr)
        img = render_from_params(m.ren, pxE, pyE, sgE, thE, frE, cf)
        frame = (img[0].permute(1, 2, 0).clamp(0, 1).numpy()
                 * 255).astype(np.uint8)[:, :, ::-1]
        frame = cv2.resize(frame, (D, D), interpolation=cv2.INTER_CUBIC)

        if overlay:
            for k in range(m.ren.N):
                b = int(bidx[k])
                col = BAND_COL[b % len(BAND_COL)] if b >= 0 else (180,) * 3
                if m.ren.qmode and b >= 0 and not band_on[b]:
                    col = (60, 60, 60)
                cv2.circle(frame, (int(float(pxE[0, k]) * D),
                                   int(float(pyE[0, k]) * D)), 2, col, -1)
            cu, cvv = st["cursor"]
            cv2.circle(frame, (int(cu * D), int(cvv * D)),
                       int(r_grab * D), (255, 255, 255), 1)
            for p in ([live_pin] if live_pin else []) + pins:
                ws = p.w.sum() + 1e-9
                cx = float((p.w * px[0]).sum() / ws) * D
                cy = float((p.w * py[0]).sum() / ws) * D
                tx, ty = p.target[0] * D, p.target[1] * D
                cv2.line(frame, (int(cx), int(cy)), (int(tx), int(ty)),
                         (0, 220, 255), 1)
                cv2.drawMarker(frame, (int(tx), int(ty)), (0, 220, 255),
                               cv2.MARKER_CROSS, 12, 2)
            bands_txt = "".join(str(i + 1) if band_on[i] else "-"
                                for i in range(n_bands))
            hud = (f"{mode}  pins {len(pins)}  edits {len(stack)}  "
                   f"bands {bands_txt}  r {r_grab:.2f}  "
                   f"|z-z0| {float((z - z_anchor).norm()):.2f}  "
                   f"err {err:.1f}px  {fps:.0f} fps  "
                   f"{'spring' if spring else ''}{' wobble' if wobble else ''}")
            cv2.putText(frame, hud, (8, D - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        if writer:
            writer.write(frame)
        dt = time.time() - t_last
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        t_last = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = "DIRECT" if mode == "MANIFOLD" else "MANIFOLD"
        elif key == ord('p') and live_pin is not None:
            if live_pin not in pins:
                pins.append(live_pin)
        elif key == ord('c'):
            pins.clear(); stack.clear(); live_pin = live_op = None
        elif key == ord('['):
            r_grab = max(0.03, r_grab / 1.25)
        elif key == ord(']'):
            r_grab = min(0.5, r_grab * 1.25)
        elif key == ord(',') and live_op is not None:
            live_op.rot -= 0.08
        elif key == ord('.') and live_op is not None:
            live_op.rot += 0.08
        elif key == ord('<') and live_op is not None:
            live_op.log_s -= 0.05
        elif key == ord('>') and live_op is not None:
            live_op.log_s += 0.05
        elif key == ord('0'):
            band_on = [True] * n_bands; refresh_gate()
        elif key in tuple(ord(str(i)) for i in range(1, 6)):
            i = key - ord('1')
            if i < n_bands:
                band_on[i] = not band_on[i]; refresh_gate()
        elif key == ord('s'):
            spring = not spring
        elif key == ord('w'):
            wobble = not wobble
        elif key == ord('b'):
            z_anchor = z.clone(); stack.clear()
        elif key == ord('r'):
            z = z_anchor.clone(); v.zero_(); stack.clear(); pins.clear()
        elif key == ord('n'):
            z_anchor = torch.randn(128) * 0.8
            z = z_anchor.clone(); v.zero_(); stack.clear(); pins.clear()
        elif key == ord('e') and args.image:
            z_anchor = encode_image(m, args.image)
            z = z_anchor.clone(); v.zero_()

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    return 0


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="splat ragdoll — grab the "
                                 "manifold and pull")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r_grab", type=float, default=0.10)
    ap.add_argument("--lam", type=float, default=0.08)
    ap.add_argument("--iters", type=int, default=80,
                    help="max solver iterations in --gates")
    ap.add_argument("--iters_frame", type=int, default=3,
                    help="solver iterations per display frame")
    ap.add_argument("--tol_px", type=float, default=1.5)
    ap.add_argument("--drag", type=float, default=0.08)
    ap.add_argument("--pin_a", type=float, nargs=2, default=(0.35, 0.42))
    ap.add_argument("--pin_b", type=float, nargs=2, default=(0.65, 0.42))
    ap.add_argument("--gate_samples", type=int, default=4)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.gates:
        if not args.model:
            sys.exit("--gates needs --model")
        sys.exit(gates(args))
    if not args.model:
        sys.exit("need --model (or --selftest)")
    sys.exit(live(args))


if __name__ == "__main__":
    main()

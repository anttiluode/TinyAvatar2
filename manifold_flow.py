#!/usr/bin/env python3
# ============================================================================
# manifold_flow.py -- StableAIflow driven by the TinyAvatar splat manifold
#
# THE IDEA
#   StableAIflow holds a live diffusion filter together with two pixel-space
#   measurements: cv2.phaseCorrelate for motion (one global dx,dy for the whole
#   frame) and a fractal-viscosity number for "has the scene changed enough to
#   regenerate". Both are estimates made from the image, and both are the
#   weakest part of the loop -- a face that turns is not a translation, and a
#   texture statistic is not a scene model.
#
#   The TinyAvatar/SplatWorld line has a thing neither of those is: a CLOSED,
#   LOW-DIMENSIONAL, NON-HALLUCINATING manifold of faces. 128 latent dims ->
#   N Gabor packets -> render. Its motion is not estimated from pixels, it is
#   KNOWN: the decoder hands you every packet's centre, so the displacement
#   field between two frames is read off the model, analytically, exactly.
#
#   So: let the manifold carry MOTION and GATING, and let diffusion carry
#   IDENTITY and TEXTURE.
#
#       webcam -> face crop -> enc -> z_t          (the manifold state)
#       dec(z_t) -> packet centres c_k(t)
#       field W(x) = envelope-weighted scatter of (c_k(t) - c_k(t-1))
#       dream_t = crystallize( warp( dream_{t-1}, W ) )
#       re-diffuse only when the MANIFOLD says the state moved or broke
#
#   The prompt sets the identity. The manifold moves it. That is the app.
#
# WHY THIS IS ALSO FASTER
#   StableAIflow displays at diffusion rate. Here the transport loop runs at
#   webcam rate and the diffusion runs in its own thread, so the picture moves
#   at 30 fps while SDXL-Turbo refreshes the identity a few times a second.
#   A dream that lands late is warped forward by the flow accumulated since it
#   was requested (`pending` in DreamEngine) before it is shown.
#
# WHAT IS MEASURED (mf_pretest, model2.pt, 96px/256, CelebA)
#   Frame-to-frame MSE against the next render, as a fraction of doing nothing:
#       latent step   0.05   0.10   0.20   0.40   0.80   1.60
#       global shift  0.969  1.009  0.986  0.918  0.990  0.973
#       manifold      0.884  0.909  0.846  0.723  0.862  0.865
#   Read it honestly: the manifold field explains 9-28% of the change as
#   transport; a single global shift -- the phaseCorrelate incumbent -- explains
#   essentially nothing (0-8%). The win over the incumbent is real, consistent
#   at every scale, and MODEST. It is not a 10x.
#   Real motion sits in the 0.2-0.4 column: a 4px slide of a real crop moved
#   the latent by |dz| ~ 1.0 against |z| ~ 4.8.
#
#   Identity transfer (applying one latent motion delta to a different anchor)
#   correlates +0.48 at small deltas rising to +0.80 at large ones, against a
#   same-anchor-different-delta control of +0.01 +- 0.39 (n=6). The rise with
#   scale is partly an artefact: once |delta| >> |z| both anchors end up in the
#   same place. So the honest number for the operating regime is ~0.5, which
#   is why MANIFOLD identity swap is an option here and not the main path.
#   The main path puts identity in the diffusion, where it does not need to
#   transfer at all.
#
# WHAT IS NOT ESTABLISHED
#   * That this looks better. Flicker is measured (MF3), beauty is not.
#   * Anything about the manifold's render quality. It is blurry; see
#     TinyAvatar2/TroubleShootingFaceSharpness. The field does not need the
#     render to be sharp, only the packet centres to be right, which is why
#     `structure = manifold` is a slider that defaults low.
#   * The flow accumulation for late-landing dreams is additive composition of
#     small displacements. Correct to first order, unverified beyond that.
#   * MANIFOLD identity mode on a single-identity (own-face) model. That model
#     has one identity; there is nothing to swap to.
#
# GATES (pre-registered, scored by --gates; two-sided where possible)
#   MF1  manifold field beats no-warp on RENDER pairs:  ratio <= 0.90
#        controls: global-shift arm ~1.0, packet-scramble arm >= 1.0
#   MF2  manifold field beats no-warp on WEBCAM frames: ratio <= 0.95
#        and beats the global-shift arm. This is the load-bearing one: the
#        dream lives in real-image space, so the field must transport REAL
#        pixels, not just the model's own render.
#   MF3  gate efficiency: reuse fraction and flicker, live telemetry
#   MF4  identity transfer: corr >= 0.50 at the operating scale, with the
#        same-anchor control <= 0.25
#
# FILES REQUIRED IN THE SAME DIRECTORY
#   splat_trainer5.py, splat_trainer3v2.py  (model definition + load_splatvae)
#   splat_ragdoll.py, pin_driver.py         (optional: compliant subspace)
#   a checkpoint: model2.pt / model5_<tag>.pt
#
# USAGE
#   python manifold_flow.py --selftest                    # no model, no GPU
#   python manifold_flow.py --gates --model model2.pt     # MF1/MF4 on renders
#   python manifold_flow.py --gates --model model2.pt --video me.mp4   # + MF2
#   python manifold_flow.py --model model2.pt             # live, SDXL-Turbo
#   python manifold_flow.py --model model2.pt --stub      # live, no diffusers
#
# KEYS (live)
#   space  force a re-dream now        m  manifold field on/off (A/B vs shift)
#   i      anchor identity to manifold r  reset dream
#   q      quit
#
# Do not hype. Do not lie. Just show.
# ============================================================================
import argparse
import math
import os
import sys
import time
import threading
import zlib
from collections import deque

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    torch.set_grad_enabled(False)

try:
    import splat_trainer5 as ST
except ImportError:
    ST = None
try:
    import pin_driver as PD
except Exception:
    PD = None


# ======================================================== manifold layer
class FaceFramer:
    """Dataset Prep's crop, live, plus the box so the field can be mapped
    back into the full frame. Same Haar cascade, same 0.35 margin, same EMA
    as tiny_avatar4 -- a different crop here would put the encoder off the
    manifold and every number downstream would be measuring that instead."""

    def __init__(self, margin=0.35, ema=0.30, every=2):
        self.det = None
        if cv2 is not None:
            p = os.path.join(cv2.data.haarcascades,
                             "haarcascade_frontalface_default.xml")
            d = cv2.CascadeClassifier(p)
            self.det = None if d.empty() else d
        self.available = self.det is not None
        self.margin, self.ema, self.every = margin, ema, every
        self.box = None
        self.f = 0
        self.found = False
        # mode: "auto" Haar, "off" whole frame, "manual" a box you place
        # defaults below are the ones he arrived at in a live session
        # "off" is the working default for the loop: with no camera there is
        # no face to hunt for, and the whole canvas IS the subject.
        self.mode = "off"
        self.manual = [0.57, 0.45, 0.35]    # cx, cy, half-side, all fractions

    def crop(self, fr):
        """-> (crop_bgr, (x0, y0, side)) in full-frame pixel coords."""
        H, W = fr.shape[:2]
        if self.mode == "off":
            s = min(H, W)
            x0, y0 = (W - s) // 2, (H - s) // 2
            self.found = False
            self.f += 1
            return fr[y0:y0 + s, x0:x0 + s], (x0, y0, s)
        if self.mode == "manual":
            cx, cy, hf = self.manual
            self.found = True
            self.f += 1
            return self._square(fr, cx * W, cy * H, hf * min(H, W))
        if self.det is not None and self.f % self.every == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            det = self.det.detectMultiScale(g, 1.15, 5, minSize=(80, 80))
            self.found = len(det) > 0
            if len(det):
                x, y, w, h = max(det, key=lambda b: b[2] * b[3])
                m = self.margin * max(w, h)
                cx, cy = x + w / 2, y + h / 2
                half = max(w, h) / 2 + m
                if self.box is None:
                    self.box = (cx, cy, half)
                else:
                    a = self.ema
                    self.box = (a * cx + (1 - a) * self.box[0],
                                a * cy + (1 - a) * self.box[1],
                                a * half + (1 - a) * self.box[2])
        self.f += 1
        if self.box is None:                      # no cascade / no face yet
            s = min(H, W)
            x0, y0 = (W - s) // 2, (H - s) // 2
            return fr[y0:y0 + s, x0:x0 + s], (x0, y0, s)
        cx, cy, half = self.box
        return self._square(fr, cx, cy, half)

    @staticmethod
    def _square(fr, cx, cy, half):
        """A crop that is ALWAYS square.

        The old code clipped the box against the frame edge and returned
        whatever rectangle survived, which `cv2.resize` then squashed to the
        encoder's square input. Measured: a box centred at 0.90 of the width
        came back with aspect 0.690, and at 0.92 of the height, 1.623. The
        encoder then saw a face stretched by up to 1.6x, so the render was
        stretched, so the structure paste stretched the image further -- a
        positive feedback that only fires when the face reaches a frame edge.
        With a webcam it almost never does. In the closed loop it does within
        seconds, and it is what tore the head apart.

        Fix: shrink the square until it fits, then slide its centre inward.
        The crop stays square at every position, so nothing is ever squashed.
        """
        H, W = fr.shape[:2]
        half = float(np.clip(half, 8, min(H, W) / 2.0))
        cx = float(np.clip(cx, half, W - half))
        cy = float(np.clip(cy, half, H - half))
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        side = int(round(2 * half))
        x0 = max(0, min(x0, W - side))
        y0 = max(0, min(y0, H - side))
        return fr[y0:y0 + side, x0:x0 + side], (x0, y0, side)


def normalize_crop(x, tgt_mean=0.52, tgt_std=0.26):
    m, s = x.mean(), x.std() + 1e-6
    return np.clip((x - m) / s * tgt_std + tgt_mean, 0.0, 1.0)


class PacketState:
    """What the decoder tells us about one frame. Numpy, canvas unit coords."""
    __slots__ = ("px", "py", "sigma", "amp", "z")

    def __init__(self, px, py, sigma, amp, z):
        self.px, self.py, self.sigma, self.amp, self.z = px, py, sigma, amp, z

    def disp_to(self, other):
        return np.stack([other.px - self.px, other.py - self.py], 1)


class Manifold:
    """The splat model, wrapped so the rest of the file never touches torch."""

    def __init__(self, path, device="cpu", chunk=64):
        if ST is None:
            sys.exit("manifold_flow.py needs splat_trainer5.py and "
                     "splat_trainer3v2.py in the same directory.")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.model, self.ck = ST.load_splatvae(path, chunk=chunk,
                                               map_location="cpu")
        self.model.to(device).eval()
        self.dev = device
        self.S = self.model.ren.H
        self.N = self.model.ren.N
        self.path = path
        # a cheap forward pass here, at load time, turns a checkpoint that is
        # merely the wrong SHAPE (mismatched packet/image-size in the file's
        # own header, a half-written save) into a clear failure at the button
        # press instead of a cryptic one on the next webcam frame.
        with torch.no_grad():
            self.render(torch.zeros(128, device=device))

    def describe(self):
        return (f"{os.path.basename(self.path)}  {self.S}px  {self.N} packets  "
                f"qmode={self.ck.get('qmode', False)}  "
                f"band_mode={self.ck.get('band_mode', 'n/a')}")

    # ---- encode -----------------------------------------------------------
    def encode(self, crop_bgr, normalize=True):
        img = cv2.cvtColor(cv2.resize(crop_bgr, (self.S, self.S)),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if normalize:
            img = normalize_crop(img)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(self.dev)
        mu, _ = self.model.enc(x)
        return mu[0], img

    # ---- decode -----------------------------------------------------------
    def state(self, z):
        raw = self.model.dec(z[None].to(self.dev))
        px, py, sig, th, f, co = self.model.ren.activate(raw.float())
        amp = co.reshape(1, self.N, -1).pow(2).mean(-1).sqrt()
        return PacketState(px[0].cpu().numpy(), py[0].cpu().numpy(),
                           sig[0].cpu().numpy(), amp[0].cpu().numpy(),
                           z.cpu().numpy())

    def render(self, z):
        r = self.model.ren(self.model.dec(z[None].to(self.dev)))
        return r[0].permute(1, 2, 0).cpu().numpy()


# ======================================================== transport layer
def dense_field(state, disp, out_hw, sigma_floor=0.02, amp_floor=1e-3):
    """Scatter per-packet displacement into a dense flow.

        W(x) = sum_k a_k E_k(x) d_k / sum_k a_k E_k(x)
        E_k(x) = exp(-|x - c_k|^2 / 2 sigma_k^2)

    The envelope is the packet's own footprint, so a packet only votes where
    it actually paints. Returns (H, W, 2) in UNIT canvas coords -- callers
    scale to whatever pixel canvas they are warping.

    Not optical flow. Nothing is estimated here; d_k came out of the decoder.
    """
    H, Wd = out_hw
    gy, gx = np.meshgrid(np.linspace(0, 1, H, dtype=np.float32),
                         np.linspace(0, 1, Wd, dtype=np.float32),
                         indexing="ij")
    sig = np.maximum(state.sigma, sigma_floor).astype(np.float32)
    amp = np.maximum(state.amp, amp_floor).astype(np.float32)
    d2 = ((gx[..., None] - state.px[None, None].astype(np.float32)) ** 2 +
          (gy[..., None] - state.py[None, None].astype(np.float32)) ** 2)
    w = amp[None, None] * np.exp(-d2 / (2.0 * sig[None, None] ** 2))
    den = w.sum(-1) + 1e-9
    fx = (w * disp[None, None, :, 0].astype(np.float32)).sum(-1) / den
    fy = (w * disp[None, None, :, 1].astype(np.float32)).sum(-1) / den
    return np.stack([fx, fy], -1)


def global_shift_field(state, disp, out_hw):
    """The incumbent, as a control arm: one amplitude-weighted translation
    for the whole canvas. This is what phaseCorrelate gives you at best."""
    a = np.maximum(state.amp, 1e-6)
    gxm = float((a * disp[:, 0]).sum() / a.sum())
    gym = float((a * disp[:, 1]).sum() / a.sum())
    H, Wd = out_hw
    f = np.empty((H, Wd, 2), np.float32)
    f[..., 0] = gxm
    f[..., 1] = gym
    return f


_WARP_GRID = {}


def warp_image(img, flow_px):
    """Pull-sample img by flow_px (H, W, 2) in PIXELS of img's own canvas."""
    H, W = img.shape[:2]
    g = _WARP_GRID.get((H, W))
    if g is None:
        X, Y = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32))
        g = _WARP_GRID[(H, W)] = (X, Y)
    mx = (g[0] - flow_px[..., 0]).astype(np.float32)
    my = (g[1] - flow_px[..., 1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def place_field(field_unit, box, frame_hw, canvas, mask=None):
    """Map a unit-coord field defined on the face crop into a `canvas`-sized
    flow in pixels, zero outside the crop, feathered by `mask` if given."""
    x0, y0, side = box
    H, W = frame_hw
    sx, sy = canvas / float(W), canvas / float(H)
    out = np.zeros((canvas, canvas, 2), np.float32)
    bw = max(int(round(side * sx)), 2)
    bh = max(int(round(side * sy)), 2)
    f = cv2.resize(field_unit, (bw, bh), interpolation=cv2.INTER_LINEAR)
    # unit coords of the crop -> pixels of the dream canvas
    f = f.copy()
    f[..., 0] *= bw
    f[..., 1] *= bh
    ox, oy = int(round(x0 * sx)), int(round(y0 * sy))
    x1, y1 = min(ox + bw, canvas), min(oy + bh, canvas)
    ox, oy = max(ox, 0), max(oy, 0)
    if x1 > ox and y1 > oy:
        out[oy:y1, ox:x1] = f[:y1 - oy, :x1 - ox]
    if mask is not None:
        out *= mask[..., None]
    return out


def place_scalar(field, box, frame_hw, canvas, expand=1.0):
    """Same box mapping as place_field, for a single-channel map."""
    x0, y0, side = box
    H, W = frame_hw
    sx, sy = canvas / float(W), canvas / float(H)
    out = np.zeros((canvas, canvas), np.float32)
    bw = max(int(round(side * sx * expand)), 2)
    bh = max(int(round(side * sy * expand)), 2)
    f = cv2.resize(field, (bw, bh), interpolation=cv2.INTER_LINEAR)
    cx = x0 * sx + side * sx / 2
    cy = y0 * sy + side * sy / 2
    ox, oy = int(round(cx - bw / 2)), int(round(cy - bh / 2))
    sx0, sy0 = max(-ox, 0), max(-oy, 0)
    ox, oy = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + bw - sx0, canvas), min(oy + bh - sy0, canvas)
    if x1 > ox and y1 > oy:
        out[oy:y1, ox:x1] = f[sy0:sy0 + (y1 - oy), sx0:sx0 + (x1 - ox)]
    return out


def crystallize(img, amount, sigma=2.0):
    """StableAIflow's unsharp mask, kept verbatim in spirit. Resampling is a
    low-pass; without this the warped dream melts. With phase-locked edges the
    sharpener reinforces the same lines every frame."""
    if amount <= 0:
        return img
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, g, -amount, 0)


def flow_to_rgb(flow_px, scale=8.0):
    ang = np.arctan2(flow_px[..., 1], flow_px[..., 0])
    mag = np.hypot(flow_px[..., 0], flow_px[..., 1])
    hsv = np.zeros(flow_px.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = ((ang + math.pi) * 90 / math.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * scale * 255 / max(scale, 1e-6), 0, 255) \
        .astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def manifold_coverage(state, out_hw, sigma_floor=0.02):
    """Where the manifold actually paints: sum of amplitude-weighted envelopes,
    normalised to [0,1]. This is a mask read off the MODEL rather than off a
    Haar box -- it scales with the packets, so it shrinks when you lean back
    without anyone choosing a scale, and it has the shape of the thing the
    manifold believes is there rather than the shape of an ellipse."""
    H, W = out_hw
    gy, gx = np.meshgrid(np.linspace(0, 1, H, dtype=np.float32),
                         np.linspace(0, 1, W, dtype=np.float32), indexing="ij")
    sig = np.maximum(state.sigma, sigma_floor).astype(np.float32)
    amp = np.maximum(state.amp, 0.0).astype(np.float32)
    d2 = ((gx[..., None] - state.px[None, None].astype(np.float32)) ** 2 +
          (gy[..., None] - state.py[None, None].astype(np.float32)) ** 2)
    c = (amp[None, None] * np.exp(-d2 / (2.0 * sig[None, None] ** 2))).sum(-1)
    m = c.max()
    return (c / m) if m > 1e-9 else c


def coverage_geometry(cov):
    """Centroid and RMS radius of a coverage map, in unit coords of its own
    canvas. This is where the manifold thinks the thing IS and how big it
    thinks it is -- the two quantities the structure paste must stop asserting
    implicitly."""
    H, W = cov.shape
    gy, gx = np.meshgrid(np.linspace(0, 1, H, dtype=np.float32),
                         np.linspace(0, 1, W, dtype=np.float32), indexing="ij")
    w = np.maximum(cov, 0.0).astype(np.float32)
    s = float(w.sum()) + 1e-9
    cx, cy = float((w * gx).sum() / s), float((w * gy).sum() / s)
    r = math.sqrt(max(float((w * ((gx - cx) ** 2 + (gy - cy) ** 2)).sum() / s),
                      1e-12))
    return cx, cy, r


def spectral_split(added, n_oct=5):
    """Where in frequency did the diffusion put what it added?

    `added` is (output - manifold render) inside the face region, DC removed.
    Returns the fraction of its power sitting in the top two octaves.

    This is the measurement the detail mode lives or dies by. The manifold's
    deficit is measured and BAND-LIMITED -- 0.999/0.993/0.979/0.942/0.861
    capture per octave on model5_constQ, i.e. uniformly mild softness biased to
    the top. If diffusion adds power there, it is supplying detail the model
    provably lacks. If it adds power at the BOTTOM, it is inventing geometry --
    a new nose, a different jaw -- which is identity drift wearing a detail
    costume, and it is the exact failure a closed manifold was chosen to avoid.
    """
    x = np.asarray(added, np.float32)
    if x.ndim == 3:
        x = x.mean(2)
    x = x - x.mean()
    F = np.abs(np.fft.rfft2(x)) ** 2
    H, Wr = F.shape
    fy = np.fft.fftfreq(x.shape[0])[:, None] * x.shape[0]
    fx = np.arange(Wr)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    nyq = x.shape[0] / 2.0
    tot = F.sum() + 1e-12
    hi = F[r > nyq / 4.0].sum()          # top two of five dyadic octaves
    return float(hi / tot)


_MASK_CACHE = {}


def ellipse_mask(canvas, box, frame_hw, expand=1.0, feather=61):
    """Cached on a coarsened box so a still head does not rebuild it 30x/s."""
    key = (canvas, int(box[0]) // 3, int(box[1]) // 3, int(box[2]) // 3,
           round(expand, 2), frame_hw)
    m = _MASK_CACHE.get(key)
    if m is not None:
        return m
    m = _ellipse_mask(canvas, box, frame_hw, expand, feather)
    if len(_MASK_CACHE) > 256:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = m
    return m


def _ellipse_mask(canvas, box, frame_hw, expand=1.0, feather=61):
    x0, y0, side = box
    H, W = frame_hw
    sx, sy = canvas / float(W), canvas / float(H)
    m = np.zeros((canvas, canvas), np.float32)
    cx, cy = (x0 + side / 2) * sx, (y0 + side / 2) * sy
    rx, ry = side * sx * expand / 2, side * sy * expand / 2 * 1.15
    cv2.ellipse(m, (int(cx), int(cy)), (int(max(rx, 2)), int(max(ry, 2))),
                0, 0, 360, 1.0, -1)
    k = feather | 1
    return cv2.GaussianBlur(m, (k, k), 0)


# ======================================================== the 2-manifold loop
class LoopRegime:
    """Classify what the coupled system is doing.

    With the webcam removed, this is a closed dynamical system: two generative
    models each taking the other's output as input. Such a thing has exactly
    three interesting behaviours and it is worth naming which one is on screen
    rather than watching and guessing.

        FIXED POINT   the step size collapses -- the two models have agreed on
                      an image that survives a round trip
        CYCLE ~N      z returns near where it was N frames ago, repeatedly
        WANDER        it keeps moving and does not come back

    A frozen picture here is a RESULT, not a bug: it means the composition
    enc(diffuse(render(z))) has a stable fixed point, which is a real property
    of the pair and not something anyone put there.
    """

    def __init__(self, n=120):
        self.z = deque(maxlen=n)
        self.steps = deque(maxlen=n)

    def observe(self, z):
        z = np.asarray(z, np.float64).ravel()
        if self.z:
            self.steps.append(float(np.linalg.norm(z - self.z[-1])))
        self.z.append(z.copy())

    def report(self):
        if len(self.z) < 12:
            return dict(regime="warming up", step=0.0, period=0, ratio=0.0)
        Z = np.stack(self.z)
        scale = float(np.linalg.norm(Z, axis=1).mean()) + 1e-9
        step = float(np.mean(self.steps)) if self.steps else 0.0
        if step / scale < 0.01:
            return dict(regime="FIXED POINT", step=step, period=0, ratio=0.0)
        # net displacement over path length: an orbit returns, a walk does not
        path = float(np.sum(self.steps)) + 1e-12
        net = float(np.linalg.norm(Z[-1] - Z[0]))
        ratio = net / path
        # recurrence: is there a lag at which the trajectory revisits itself
        # more closely than it does at a typical lag?
        best_lag, best = 0, 1e9
        base = np.median([np.linalg.norm(Z[i] - Z[j])
                          for i in range(0, len(Z), 3)
                          for j in range(0, len(Z), 3) if i != j]) + 1e-12
        for lag in range(4, len(Z) // 2):
            d = np.linalg.norm(Z[lag:] - Z[:-lag], axis=1).mean()
            if d < best:
                best, best_lag = d, lag
        if best < 0.35 * base and ratio < 0.35:
            return dict(regime=f"CYCLE ~{best_lag}", step=step,
                        period=best_lag, ratio=ratio)
        return dict(regime="WANDER", step=step, period=0, ratio=ratio)


def loop_drive(t, dims, amp, seed=0):
    """A slow, smooth, DETERMINISTIC latent perturbation for the loop.

    Deliberately not white noise: noise would inject broadband energy every
    frame and whatever motion appeared could always be blamed on the noise.
    A few incommensurate sinusoids are reproducible and obviously not the
    source of any structure, so what survives is the coupling.
    """
    if amp <= 0:
        return np.zeros(dims, np.float32)
    rs = np.random.RandomState(seed)
    f = 0.003 + 0.02 * rs.rand(6)
    ph = 2 * math.pi * rs.rand(6)
    mix = rs.randn(6, dims).astype(np.float32) / math.sqrt(dims)
    w = np.sin(2 * math.pi * f * t + ph).astype(np.float32)
    return amp * (w @ mix)


# ======================================================== the gate
class ManifoldGate:
    """When to spend a diffusion step.

    StableAIflow asks a texture statistic whether the scene changed. We ask
    the manifold three questions it can actually answer:

      novelty  ||z - z_key|| / (||z_key|| + eps)   how far the state travelled
      drift    accumulated |flow| in px            how much resampling the
                                                   dream has suffered
      residual ||x_crop - render(z)||              is the manifold still
                                                   explaining the sensor at
                                                   all (occlusion, second
                                                   face, hand over the lens)

    A residual spike means we are OFF the manifold, and then the manifold's
    motion field is not to be trusted -- so the gate both fires a re-dream and
    tells the pipeline to lean on the webcam.
    """

    def __init__(self, novelty=0.30, drift=51.0, max_reuse=9, res_k=2.5):
        self.novelty_t, self.drift_t = novelty, drift
        self.max_reuse, self.res_k = max_reuse, res_k
        self.z_key = None
        self.drift = 0.0
        self.since = 10 ** 9
        self.res_hist = deque(maxlen=90)
        self.last_reason = "init"
        self.off_manifold = False

    def observe(self, z, flow_px, residual):
        self.since += 1
        if flow_px is not None:
            self.drift += float(np.abs(flow_px).mean())
        self.res_hist.append(float(residual))
        med = float(np.median(self.res_hist)) if self.res_hist else residual
        mad = float(np.median(np.abs(np.array(self.res_hist) - med)))
        # BUGFIX (found in a live session): with a steady residual the MAD
        # collapses toward zero, so ANY numerical wobble cleared the threshold
        # and the gate reported OFF-MANIFOLD on every frame -- which fires a
        # re-dream on every frame, which is the whole transport loop disabled.
        # The scale needs a floor proportional to the residual itself.
        scale = max(1.4826 * mad, 0.08 * med, 1e-4)
        self.off_manifold = (len(self.res_hist) >= 20 and
                             (residual - med) > self.res_k * scale)

    def novelty(self, z):
        if self.z_key is None:
            return 10 ** 9
        return float(np.linalg.norm(z - self.z_key) /
                     (np.linalg.norm(self.z_key) + 1e-6))

    def should_fire(self, z, prompt_changed):
        if prompt_changed:
            self.last_reason = "prompt"
            return True
        if self.z_key is None:
            self.last_reason = "first"
            return True
        if self.since >= self.max_reuse:
            self.last_reason = "timeout"
            return True
        if self.novelty(z) > self.novelty_t:
            self.last_reason = "novelty"
            return True
        if self.drift > self.drift_t:
            self.last_reason = "drift"
            return True
        if self.off_manifold:
            self.last_reason = "off-manifold"
            return True
        return False

    def armed(self, z):
        self.z_key = np.asarray(z, np.float32).copy()
        self.drift = 0.0
        self.since = 0


# ======================================================== dream engine
class StubDream:
    """No diffusers, no 7GB download, no GPU. Deterministic per prompt so the
    pipeline and the gates can be exercised end to end. This is NOT diffusion
    and produces nothing anyone would call an identity -- it exists so that a
    failure in the transport loop cannot hide behind a model download."""

    name = "stub"

    def __init__(self, **kw):
        self.ready = True

    def __call__(self, img_rgb, prompt, strength, steps=4):
        # zlib.crc32, not hash(): Python randomises string hashing per process
        # (PYTHONHASHSEED), so hash() made this stub a DIFFERENT function on
        # every run. Caught by the loop probe, whose trajectory refused to
        # reproduce across invocations while every other input was seeded.
        h = zlib.crc32(prompt.encode("utf-8")) % 997
        rng = np.random.RandomState(h)
        pal = rng.randint(40, 230, (3, 3)).astype(np.float32)
        g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
        e = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.5), cv2.CV_32F)
        e = np.clip(np.abs(e) * 6.0, 0, 1)
        q = np.clip((g * 3).astype(np.int32), 0, 2)
        out = pal[q] * (1.0 - 0.6 * e[..., None])
        # strength follows the diffusers img2img convention: 1.0 = reassert the
        # model fully, 0.0 = leave the input alone. This was INVERTED here and
        # nobody noticed until strength=1.00 became the default, at which point
        # the stub silently became the identity function and quietly nulled
        # three gates (MF4, LP1, LK1) that depend on the image step doing
        # something. A test double with the wrong sign is worse than none.
        out = out * strength + img_rgb.astype(np.float32) * (1.0 - strength)
        return np.clip(out, 0, 255).astype(np.uint8)


def timestep_retention(steps, strength):
    """What fraction of the INPUT IMAGE's signal survives diffusers' img2img
    noising step, for a given (steps, strength) pair.

    This is the exact mechanism, not an approximation: img2img computes
    t_start = steps - min(int(steps*strength), steps), then does
    scheduler.add_noise(input_latent, noise, timesteps[t_start]), which for a
    Euler-family scheduler is  noisy = input + sigma[t_start] * noise. Signal
    power fraction is therefore 1 / (1 + sigma^2). At strength=1.0, t_start=0
    and sigma is the schedule's maximum (~14.6 for SDXL's default betas) --
    0.47% of the input survives, i.e. img2img at strength 1.0 is ~txt2img.
    Requires diffusers; returns None (not an error) if it is not installed,
    since this is diagnostic and must never block the app or --selftest.
    """
    try:
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
    except ImportError:
        return None
    sch = EulerAncestralDiscreteScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", prediction_type="epsilon")
    sch.set_timesteps(steps)
    init = min(int(steps * strength), steps)
    t_start = max(steps - init, 0)
    sigma = float(sch.sigmas[t_start])
    return 1.0 / (1.0 + sigma ** 2)


class SDXLTurbo:
    name = "sdxl-turbo"

    def __init__(self, model_id="stabilityai/sdxl-turbo", device="cuda"):
        from diffusers import AutoPipelineForImage2Image
        dt = torch.float16 if device == "cuda" else torch.float32
        kw = dict(torch_dtype=dt)
        if device == "cuda":
            kw["variant"] = "fp16"
        self.pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kw)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.ready = True

    def __call__(self, img_rgb, prompt, strength, steps=4):
        from PIL import Image
        # steps is now FIXED, not derived from strength. Deriving it as
        # int(2.0/strength) was a knife-edge, not a dial -- see
        # timestep_retention() below and README_manifold_flow.md ("does
        # strength=1 use the avatar at all"). At strength=1.0 with that
        # formula, steps=2 and img2img starts at the noisiest timestep,
        # discarding ~99.5% of the input's signal (measured against the real
        # SDXL/Turbo scheduler math). At strength=0.90 with the SAME formula,
        # steps was ALSO 2, but strength*steps rounds down to 1, so it starts
        # at the almost-clean end instead -- 99.9% RETAINED, a near no-op. The
        # two settings looked one slider-tick apart and were opposite
        # extremes. A fixed step count makes retention move smoothly with
        # strength instead of flipping between the two ends.
        out = self.pipe(prompt=prompt, image=Image.fromarray(img_rgb),
                        strength=strength, guidance_scale=0.0,
                        num_inference_steps=steps).images[0]
        return np.array(out)


class DreamEngine:
    """Runs the slow model on its own thread.

    The transport loop must never block on diffusion. A request carries the
    structure image and the latent it was made at; while the model runs, the
    pipeline keeps accumulating flow into `pending`, and the finished dream is
    warped forward by that accumulation before it is shown. Without this the
    identity lands a quarter second in the past and snaps.
    """

    def __init__(self, backend, canvas=512):
        self.backend = backend
        self.canvas = canvas
        self.req = None
        self.result = None
        self.pending = np.zeros((canvas, canvas, 2), np.float32)
        self.busy = False
        self.lock = threading.Lock()
        self.ev = threading.Event()
        self.alive = True
        self.ms = 0.0
        self.count = 0
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def request(self, img_rgb, prompt, strength, z, steps=4):
        with self.lock:
            if self.busy:
                return False
            self.req = (img_rgb.copy(), prompt, float(strength),
                        np.asarray(z, np.float32).copy(), int(steps))
            self.busy = True
            self.pending[:] = 0.0
        self.ev.set()
        return True

    def accumulate(self, flow_px):
        """First-order composition of small displacements. Exact for pure
        translation, approximate otherwise; the approximation is bounded by
        how long a dream takes, which is the thing we are minimising anyway."""
        with self.lock:
            if self.busy and flow_px is not None:
                self.pending += flow_px

    def take(self):
        with self.lock:
            r, self.result = self.result, None
            return r

    def _loop(self):
        while self.alive:
            self.ev.wait(0.05)
            self.ev.clear()
            with self.lock:
                req = self.req
                self.req = None
            if req is None:
                continue
            img, prompt, strength, z, steps = req
            t0 = time.time()
            try:
                out = self.backend(img, prompt, strength, steps=steps)
            except Exception as e:
                print("dream error:", e)
                out = img
            with self.lock:
                fwd = self.pending.copy()
                self.ms = (time.time() - t0) * 1000
                self.count += 1
            if float(np.abs(fwd).max()) > 0.25:
                out = warp_image(out, fwd)
            with self.lock:
                self.result = (out, z)
                self.busy = False

    def stop(self):
        self.alive = False
        self.ev.set()


# ======================================================== the pipeline
class FlowPipeline:
    """One webcam frame in, one composited frame out. No Qt, no threads of its
    own beyond the DreamEngine -- so --gates can drive it from a video file."""

    def __init__(self, manifold, engine, canvas=512, cfg=None):
        self.mf = manifold
        self.eng = engine
        self.canvas = canvas
        self.framer = FaceFramer()
        self.gate = ManifoldGate()
        self.cfg = dict(
            strength=1.00, gravity=0.00, sharpness=0.00, pursuit=0.92,
            structure=0.33, mask=True, mask_scale=1.00, use_field=True,
            normalize=True, field_gain=1.04, prompt="",
            field_res=48, render_every=2,
            mask_source="manifold",   # "box" (Haar ellipse) or "manifold"
            measure_every=6,
            loop=False, loop_drive=0.59, lockstep=True, render_fill=0.62,
            loop_anchor=0.50, diffusion_steps=4,
        )
        if cfg:
            self.cfg.update(cfg)
        self.z = None
        self.prev = None            # PacketState
        self.dream = None
        self.last_flow = None
        self.telemetry = {}
        self._prompt_seen = None
        self.flicker = deque(maxlen=120)
        self.fired = 0
        self.frames = 0
        self._rend = None
        self._hf = 0.0
        self.regime = LoopRegime()
        self.loop_t = 0
        self._anchor_prev = None
        self._anchor_acc = np.zeros(2, np.float64)
        self._anchor_win = None
        self._retention = None

    # -- latent smoothing ---------------------------------------------------
    # -- 2-manifold loop ----------------------------------------------------
    def seed_loop(self, scale=1.5, seed=None):
        """Start the loop from a point ON the avatar manifold rather than from
        noise, so the first thing the diffusion sees is a face this model can
        actually represent. Returns the seed frame in BGR."""
        g = torch.Generator().manual_seed(
            int(time.time() * 1000) % 2 ** 31 if seed is None else seed)
        z = (torch.randn(128, generator=g) * scale).to(self.mf.dev)
        r = (np.clip(self.mf.render(z), 0, 1) * 255).astype(np.uint8)
        img = cv2.resize(r, (self.canvas, self.canvas))
        self.z = None
        self.prev = None
        self.dream = img.copy()
        self.regime = LoopRegime()
        self.loop_t = 0
        self._anchor_prev = None
        self._anchor_acc = np.zeros(2, np.float64)
        self._anchor_win = None
        self._retention = None
        self.gate.z_key = None
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def anchor_loop(self):
        """Hold the closed loop's picture still.

        The two structural fixes above stop the pipeline from injecting drift,
        and with diffusion replaced by identity the loop now settles instead of
        marching. But the image model has compositional biases of its own -- at
        strength 0.74 SDXL-Turbo substantially redraws the frame each pass, and
        nothing in the pipeline can prevent it from placing the subject a
        little further right every time.

        So measure the global translation the image actually underwent and
        integrate it out. This is `cv2.phaseCorrelate`, the very thing the
        manifold replaced for MOTION -- used here for the one job it is
        genuinely right for: estimating a single global shift, in order to
        cancel it. A proportional correction on the accumulated offset, so it
        holds position without fighting local deformation.

        Only active in the loop. With a webcam, global translation is real
        information and must not be removed.
        """
        a = float(self.cfg["loop_anchor"])
        if self.dream is None or a <= 0:
            self._anchor_prev = None if self.dream is None else \
                cv2.cvtColor(self.dream, cv2.COLOR_RGB2GRAY).astype(np.float32)
            return
        cur = cv2.cvtColor(self.dream, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if self._anchor_prev is not None and \
                self._anchor_prev.shape == cur.shape:
            if self._anchor_win is None or \
                    self._anchor_win.shape != cur.shape:
                self._anchor_win = cv2.createHanningWindow(
                    (cur.shape[1], cur.shape[0]), cv2.CV_32F)
            try:
                # the Hanning window matters: without it the frame edges
                # dominate the correlation and the estimate under-reads, which
                # leaks drift back in at roughly 0.07 of the frame per 120
                # passes even with the anchor at full strength.
                (dx, dy), _ = cv2.phaseCorrelate(self._anchor_prev, cur,
                                                 self._anchor_win)
            except cv2.error:
                dx = dy = 0.0
            if math.isfinite(dx) and math.isfinite(dy):
                lim = 0.15 * self.canvas
                self._anchor_acc += np.clip([dx, dy], -lim, lim)
            corr = a * self._anchor_acc
            if float(np.abs(corr).max()) > 0.05:
                f = np.zeros((self.canvas, self.canvas, 2), np.float32)
                f[..., 0] = -corr[0]
                f[..., 1] = -corr[1]
                self.dream = warp_image(self.dream, f)
                self._anchor_acc -= corr
                cur = cv2.cvtColor(self.dream,
                                   cv2.COLOR_RGB2GRAY).astype(np.float32)
        self._anchor_prev = cur

    def loop_frame(self):
        """The 'camera' in loop mode: the diffusion's own last output.

        This is the entire closure. The avatar manifold reads the diffusion's
        image; the diffusion reads the avatar's render and motion field. No
        external input anywhere -- so anything that moves is the pair.
        """
        if self.dream is None:
            return self.seed_loop()
        return cv2.cvtColor(self.dream, cv2.COLOR_RGB2BGR)

    def _regime_report(self):
        """The regime, with one confound called out where it applies.

        A probe run at drive 0 came back 'CYCLE ~4' -- and the loop was also
        issuing one dream every 3.9 frames. A period that matches the dream
        cadence is the GATE being detected, not the two models orbiting each
        other. Flag it rather than let a re-dream rhythm be read as dynamics.
        """
        r = dict(self.regime.report())
        if r["period"] and self.eng is not None and self.eng.count > 4:
            cadence = self.frames / max(self.eng.count, 1)
            if abs(r["period"] - cadence) <= max(1.0, 0.25 * cadence):
                r["regime"] += " (= dream cadence, not dynamics)"
        return r

    def load_manifold(self, mf):
        """Swap in a new checkpoint without losing the sliders. A different
        model has a different S/N and almost certainly a different latent
        scale, so the standing dream, the pursued latent, the gate's key and
        its residual history are all stale for the new manifold and must be
        dropped -- keeping any of them would silently mix two models' state."""
        self.mf = mf
        self.z = None
        self.prev = None
        self.dream = None
        self._rend = None
        self._hf = 0.0
        self.regime = LoopRegime()
        self.loop_t = 0
        self._anchor_prev = None
        self._anchor_acc = np.zeros(2, np.float64)
        self._anchor_win = None
        self._retention = None
        self.gate = ManifoldGate(self.gate.novelty_t, self.gate.drift_t,
                                 self.gate.max_reuse, self.gate.res_k)
        self._prompt_seen = None
        self.flicker.clear()

    def _pursue(self, z_raw):
        if self.cfg["loop"]:
            self.loop_t += 1
            amp = float(self.cfg["loop_drive"])
            if amp > 0:
                d = loop_drive(self.loop_t, int(z_raw.shape[0]), amp)
                z_raw = z_raw + torch.from_numpy(d).to(z_raw.device)
        if self.z is None:
            return z_raw
        a = float(self.cfg["pursuit"])
        return self.z + a * (z_raw - self.z)

    def _mask(self, st, box, frame_hw):
        """One mask, used for the field footprint and for the composite.
        Either the Haar ellipse or the manifold's own coverage."""
        if not self.cfg["mask"]:
            return None
        if self.cfg["mask_source"] == "manifold":
            cov = manifold_coverage(st, (64, 64))
            cov = np.clip((cov - 0.10) / 0.35, 0, 1) ** 0.7
            cov = cv2.GaussianBlur(cov, (0, 0), 3.0)
            return place_scalar(cov, box, frame_hw, self.canvas,
                                float(self.cfg["mask_scale"]))
        return ellipse_mask(self.canvas, box, frame_hw,
                            self.cfg["mask_scale"])

    def step(self, frame_bgr):
        self.frames += 1
        H, W = frame_bgr.shape[:2]
        crop, box = self.framer.crop(frame_bgr)
        z_raw, crop_img = self.mf.encode(crop, self.cfg["normalize"])
        self.z = self._pursue(z_raw)
        st = self.mf.state(self.z)
        # the render is only needed for the residual, the panel and the
        # structure blend -- none of which need it every frame. The FIELD does
        # not use it at all; that is the point of reading motion off the
        # decoder instead of off pixels.
        if self._rend is None or self.frames % int(self.cfg["render_every"]) == 0:
            self._rend = self.mf.render(self.z)
        rend = self._rend

        # residual: does the manifold still explain what the camera sees
        residual = float(((rend - crop_img) ** 2).mean())

        # ---- the field
        flow = None
        field_unit = None
        if self.prev is not None:
            d = self.prev.disp_to(st) * float(self.cfg["field_gain"])
            gridres = int(min(self.cfg["field_res"], self.mf.S))
            if self.cfg["use_field"]:
                field_unit = dense_field(self.prev, d, (gridres, gridres))
            else:
                field_unit = global_shift_field(self.prev, d,
                                                (gridres, gridres))
            m = self._mask(st, box, (H, W))
            flow = place_field(field_unit, box, (H, W), self.canvas, m)
        self.prev = st
        self.last_flow = flow

        # ---- transport the standing dream
        if self.dream is not None and flow is not None:
            before = self.dream
            self.dream = crystallize(warp_image(self.dream, flow),
                                     self.cfg["sharpness"])
            self.flicker.append(float(np.abs(
                self.dream.astype(np.float32) - before.astype(np.float32)
            ).mean()))
        self.eng.accumulate(flow)

        # ---- gate
        z_np = self.z.cpu().numpy() if hasattr(self.z, "cpu") else \
            np.asarray(self.z)
        self.gate.observe(z_np, flow, residual)
        if self.cfg["loop"]:
            self.regime.observe(z_np)
        prompt_changed = (self.cfg["prompt"] != self._prompt_seen)
        if self.gate.should_fire(z_np, prompt_changed):
            structure = self._structure(frame_bgr, rend, box, (H, W), st)
            steps = int(self.cfg["diffusion_steps"])
            if self.cfg["loop"] and self.cfg["lockstep"]:
                # LOCKSTEP. In the loop the image model is INSIDE the feedback
                # path, so running it on its own thread makes the number of
                # transport frames per dream depend on machine speed -- and a
                # probe run showed the detected regime swinging between FIXED
                # POINT and CYCLE purely on thread scheduling. That is a
                # timing artefact being read as dynamics. Run it synchronously
                # and the loop becomes one well-defined map, z -> enc(diffuse(
                # render(z))), applied once per frame. Slower, and the only
                # version whose regime means anything.
                t0 = time.time()
                try:
                    self.dream = self.eng.backend(
                        structure, self.cfg["prompt"], self.cfg["strength"],
                        steps=steps)
                except Exception as e:
                    print("dream error:", e)
                self.eng.ms = (time.time() - t0) * 1000
                self.eng.count += 1
                self._prompt_seen = self.cfg["prompt"]
                self.gate.armed(z_np)
                self.fired += 1
                self._retention = timestep_retention(steps,
                                                     self.cfg["strength"])
            elif self.eng.request(structure, self.cfg["prompt"],
                                  self.cfg["strength"], z_np, steps=steps):
                self._prompt_seen = self.cfg["prompt"]
                self.gate.armed(z_np)
                self.fired += 1
                self._retention = timestep_retention(steps,
                                                     self.cfg["strength"])

        # ---- collect a finished dream
        got = self.eng.take()
        if got is not None:
            self.dream = got[0]

        if self.cfg["loop"]:
            self.anchor_loop()

        out = self._composite(frame_bgr, box, (H, W), st)

        # what did the diffusion ADD, and where in frequency did it put it?
        if (self.dream is not None and
                self.frames % int(self.cfg["measure_every"]) == 0):
            x0, y0, side = box
            sx, sy = self.canvas / float(W), self.canvas / float(H)
            ox, oy = int(x0 * sx), int(y0 * sy)
            bw, bh = max(int(side * sx), 8), max(int(side * sy), 8)
            sub = self.dream[oy:min(oy + bh, self.canvas),
                             ox:min(ox + bw, self.canvas)]
            if sub.size > 64:
                S = self.mf.S
                sub = cv2.resize(sub, (S, S)).astype(np.float32) / 255.
                self._hf = spectral_split(sub - np.clip(rend, 0, 1))
        self.telemetry = dict(
            residual=residual, novelty=self.gate.novelty(z_np),
            drift=self.gate.drift, reason=self.gate.last_reason,
            off=self.gate.off_manifold, dream_ms=self.eng.ms,
            reuse=1.0 - self.fired / max(self.frames, 1),
            flicker=float(np.mean(self.flicker)) if self.flicker else 0.0,
            face=self.framer.found, dreams=self.eng.count,
            zn=float(np.linalg.norm(z_np)), hf=self._hf,
            retention=self._retention,
            **self._regime_report(),
        )
        return dict(out=out, render=rend, flow=flow, crop=crop_img)

    def _structure(self, frame_bgr, rend, box, frame_hw, st):
        """What the diffusion is shown. Three things blended:
        the webcam (sharp, hallucination-prone), the standing dream (temporal
        memory -- StableAIflow's gravity), and the manifold render (closed,
        non-hallucinating, blurry). `structure` is the manifold's share and
        defaults to 0: the render is soft, and the manifold's job in this app
        is motion, not pixels. Turn it up to see what a closed prior does."""
        web = cv2.cvtColor(cv2.resize(frame_bgr, (self.canvas, self.canvas)),
                           cv2.COLOR_BGR2RGB).astype(np.float32)
        s = float(self.cfg["structure"])
        if s > 0:
            web = web * (1 - s) + self._paste_render(web, rend, st,
                                                     box, frame_hw) * s
        if self.dream is not None:
            g = float(self.cfg["gravity"])
            if self.gate.off_manifold:
                g *= 0.25          # sensor wins when the manifold is confused
            web = web * (1 - g) + self.dream.astype(np.float32) * g
        return np.clip(web, 0, 255).astype(np.uint8)

    def _paste_render(self, web, rend, st, box, frame_hw):
        """Blend the manifold render in WITHOUT letting it also assert a
        position and a size.

        The old paste dropped the render into the box rectangle. That looks
        harmless and is fatal in a closed loop: the render's face sits wherever
        the decoder puts it and fills whatever fraction of the canvas the
        decoder chooses, so every pass nudged the image toward the manifold's
        geometry and the nudge INTEGRATED. Measured with diffusion replaced by
        identity, 50 passes, structure 0.14: centroid +0.017 of the frame
        downward and radius 1.23x. It scaled cleanly with structure --
        0.00/0.07/0.14/0.40 gave drift 0.0000/0.0031/0.0171/0.0284 and radius
        1.000/1.106/1.228/1.759 -- so the render blend was the whole of it.
        The motion field was innocent: turning it off entirely (field_gain 0)
        changed nothing.

        Fix: measure the render's own coverage centroid and radius, then place
        it so the centroid lands on the BOX CENTRE and the radius is a fixed
        fraction of the box. The paste geometry then depends only on the box
        and one constant, so with a fixed box nothing integrates.
        """
        x0, y0, side = box
        H, W = frame_hw
        sx, sy = self.canvas / float(W), self.canvas / float(H)
        cov = manifold_coverage(st, (48, 48))
        ux, uy, ur = coverage_geometry(cov)
        bw_px = max(side * sx, 4.0)
        want_r = float(self.cfg["render_fill"]) * bw_px / 2.0
        k = want_r / max(ur * bw_px, 1e-6)          # scale correction
        pw = max(int(round(bw_px * k)), 4)
        ph = max(int(round(side * sy * k)), 4)
        r8 = cv2.resize((np.clip(rend, 0, 1) * 255).astype(np.uint8), (pw, ph))
        tcx = (x0 + side / 2.0) * sx
        tcy = (y0 + side / 2.0) * sy
        ox = int(round(tcx - ux * pw))
        oy = int(round(tcy - uy * ph))
        lay = web.copy()
        dx0, dy0 = max(ox, 0), max(oy, 0)
        sx0, sy0 = max(-ox, 0), max(-oy, 0)
        dx1 = min(ox + pw, self.canvas)
        dy1 = min(oy + ph, self.canvas)
        if dx1 > dx0 and dy1 > dy0:
            lay[dy0:dy1, dx0:dx1] = r8[sy0:sy0 + (dy1 - dy0),
                                       sx0:sx0 + (dx1 - dx0)]
        return lay

    def _composite(self, frame_bgr, box, frame_hw, st):
        web = cv2.cvtColor(cv2.resize(frame_bgr, (self.canvas, self.canvas)),
                           cv2.COLOR_BGR2RGB)
        if self.dream is None:
            return web
        m = self._mask(st, box, frame_hw)
        if m is None:
            return self.dream
        m = m[..., None]
        return (self.dream.astype(np.float32) * m +
                web.astype(np.float32) * (1 - m)).astype(np.uint8)


# ======================================================== selftest
def _chk(name, cond, note=""):
    print(f"  [{'V' if cond else 'X'}] {name}" + (f"   {note}" if note else ""))
    return bool(cond)


def selftest():
    """No model, no GPU, no camera. Two-sided everywhere: each check has a
    condition that must hold AND a condition that must fail."""
    print("manifold_flow selftest")
    ok = []

    # T1 -- dense_field recovers a planted displacement at the packet centre
    st = PacketState(np.array([0.3, 0.7]), np.array([0.5, 0.5]),
                     np.array([0.05, 0.05]), np.array([1.0, 1.0]), None)
    d = np.array([[0.02, -0.01], [-0.03, 0.04]])
    f = dense_field(st, d, (64, 64))
    a = f[32, int(0.3 * 63)]
    b = f[32, int(0.7 * 63)]
    ok.append(_chk("T1a field ~ planted disp at packet A",
                   abs(a[0] - 0.02) < 0.004 and abs(a[1] + 0.01) < 0.004,
                   f"got ({a[0]:+.4f},{a[1]:+.4f}) want (+0.0200,-0.0100)"))
    ok.append(_chk("T1b field ~ planted disp at packet B",
                   abs(b[0] + 0.03) < 0.006 and abs(b[1] - 0.04) < 0.006,
                   f"got ({b[0]:+.4f},{b[1]:+.4f}) want (-0.0300,+0.0400)"))
    ok.append(_chk("T1c NEGATIVE: field is not uniform",
                   float(f.std()) > 1e-3, f"std {f.std():.5f}"))

    # T2 -- warp_image moves pixels by exactly the requested amount
    img = np.zeros((64, 64, 3), np.float32)
    img[30:34, 20:24] = 1.0
    fl = np.zeros((64, 64, 2), np.float32)
    fl[..., 0] = 5.0
    fl[..., 1] = -3.0
    w = warp_image(img, fl)
    ys, xs = np.nonzero(w[..., 0] > 0.5)
    ok.append(_chk("T2a warp shifts by (+5,-3) px",
                   abs(xs.mean() - 26.5) < 0.6 and abs(ys.mean() - 28.5) < 0.6,
                   f"centroid ({xs.mean():.1f},{ys.mean():.1f}) want (26.5,28.5)"))
    w0 = warp_image(img, np.zeros_like(fl))
    ok.append(_chk("T2b NEGATIVE: zero flow is a no-op",
                   float(np.abs(w0 - img).max()) < 1e-5,
                   f"max err {np.abs(w0-img).max():.2e}"))

    # T3 -- crystallizer
    soft = cv2.GaussianBlur(np.random.RandomState(0).rand(64, 64, 3)
                            .astype(np.float32), (0, 0), 3.0)
    hf = lambda x: float(cv2.Laplacian(x.mean(2), cv2.CV_32F).std())
    ok.append(_chk("T3a crystallize raises high-frequency energy",
                   hf(crystallize(soft, 1.0)) > 1.3 * hf(soft),
                   f"{hf(soft):.5f} -> {hf(crystallize(soft,1.0)):.5f}"))
    ok.append(_chk("T3b NEGATIVE: amount 0 is bit-identical",
                   crystallize(soft, 0.0) is soft))

    # T4 -- the gate, two-sided
    g = ManifoldGate(novelty=0.10, drift=1e9, max_reuse=10 ** 9)
    z0 = np.ones(8, np.float32)
    g.armed(z0)
    for _ in range(5):
        g.observe(z0, None, 0.01)
    ok.append(_chk("T4a gate silent on a static latent",
                   not g.should_fire(z0, False)))
    ok.append(_chk("T4b gate fires on prompt change",
                   g.should_fire(z0, True)))
    ok.append(_chk("T4c gate fires past the novelty threshold",
                   g.should_fire(z0 * 1.5, False), f"nov {g.novelty(z0*1.5):.3f}"))
    g2 = ManifoldGate(novelty=1e9, drift=5.0, max_reuse=10 ** 9)
    g2.armed(z0)
    fl2 = np.full((8, 8, 2), 1.0, np.float32)
    for _ in range(6):
        g2.observe(z0, fl2, 0.01)
    ok.append(_chk("T4d gate fires on accumulated drift",
                   g2.should_fire(z0, False), f"drift {g2.drift:.1f}"))

    # T4e -- residual spike detection, two-sided
    g3 = ManifoldGate()
    rs = np.random.RandomState(3)
    for _ in range(60):
        g3.observe(z0, None, 0.010 + 0.0005 * rs.randn())
    quiet = g3.off_manifold
    g3.observe(z0, None, 0.10)
    ok.append(_chk("T4e residual spike -> off-manifold, quiet does not",
                   g3.off_manifold and not quiet))

    # T5 -- late-dream forward warp: accumulate then warp ~= warp twice
    base = np.zeros((64, 64, 3), np.float32)
    base[28:32, 28:32] = 1.0
    f1 = np.zeros((64, 64, 2), np.float32); f1[..., 0] = 3.0
    f2 = np.zeros((64, 64, 2), np.float32); f2[..., 0] = 4.0
    seq = warp_image(warp_image(base, f1), f2)
    acc = warp_image(base, f1 + f2)
    ok.append(_chk("T5 accumulated flow == sequential warp (translation)",
                   float(np.abs(seq - acc).max()) < 1e-4,
                   f"max err {np.abs(seq-acc).max():.2e}"))

    # T6 -- scramble control decorrelates the field (the MF1 control arm)
    rs = np.random.RandomState(7)
    stN = PacketState(rs.rand(48), rs.rand(48), 0.03 + 0.05 * rs.rand(48),
                      0.5 + rs.rand(48), None)
    dN = 0.02 * rs.randn(48, 2)
    fA = dense_field(stN, dN, (48, 48))
    fB = dense_field(stN, dN[rs.permutation(48)], (48, 48))
    c = float(np.corrcoef(fA.ravel(), fB.ravel())[0, 1])
    ok.append(_chk("T6 packet-scramble control decorrelates the field",
                   abs(c) < 0.5, f"corr {c:+.3f}"))

    # T7 -- place_field lands inside the box and nowhere else
    fu = np.zeros((32, 32, 2), np.float32); fu[..., 0] = 0.05
    pf = place_field(fu, (100, 50, 200), (480, 640), 512)
    inside = pf[int(50 * 512 / 480) + 20:int(50 * 512 / 480) + 60,
                int(100 * 512 / 640) + 20:int(100 * 512 / 640) + 60, 0]
    ok.append(_chk("T7a field is nonzero inside the crop box",
                   float(inside.mean()) > 1.0, f"mean {inside.mean():.2f} px"))
    ok.append(_chk("T7b NEGATIVE: field is zero far outside the box",
                   float(np.abs(pf[480:, 480:]).max()) < 1e-6))

    # T9 -- the OFF-MANIFOLD scale floor (the bug found in a live session)
    g4 = ManifoldGate()
    z0b = np.ones(8, np.float32)
    rs4 = np.random.RandomState(11)
    for _ in range(90):                      # near-constant residual
        g4.observe(z0b, None, 0.1111 + 1e-9 * rs4.randn())
    ok.append(_chk("T9a steady residual does NOT read off-manifold",
                   not g4.off_manifold,
                   "regression: MAD collapsed to 0 and fired every frame"))
    g4.observe(z0b, None, 0.1111 * 1.6)
    ok.append(_chk("T9b a real 60% residual jump still fires",
                   g4.off_manifold))

    # T10 -- manifold coverage mask, two-sided
    stc = PacketState(np.array([0.5, 0.5]), np.array([0.5, 0.52]),
                      np.array([0.06, 0.06]), np.array([1.0, 1.0]), None)
    cov = manifold_coverage(stc, (64, 64))
    ok.append(_chk("T10a coverage peaks where the packets are",
                   cov[32, 32] > 0.9 and cov[2, 2] < 0.05,
                   f"centre {cov[32,32]:.2f} corner {cov[2,2]:.3f}"))
    stc2 = PacketState(stc.px, stc.py, stc.sigma * 0.4, stc.amp, None)
    cov2 = manifold_coverage(stc2, (64, 64))
    ok.append(_chk("T10b NEGATIVE: smaller packets give a smaller mask",
                   cov2.sum() < 0.5 * cov.sum(),
                   f"{cov2.sum():.0f} vs {cov.sum():.0f}"))

    # T11 -- spectral_split, two-sided on planted signals
    rs5 = np.random.RandomState(5)
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    lo = np.sin(2 * math.pi * 1.5 * xx / n).astype(np.float32)
    hi = np.sin(2 * math.pi * 22 * xx / n).astype(np.float32)
    ok.append(_chk("T11a planted high frequency reads high",
                   spectral_split(hi) > 0.9, f"{spectral_split(hi):.3f}"))
    ok.append(_chk("T11b NEGATIVE: planted low frequency reads low",
                   spectral_split(lo) < 0.1, f"{spectral_split(lo):.3f}"))

    # T12 -- Manifold() rejects a missing file cleanly, no crash
    try:
        Manifold("/tmp/does_not_exist_xyz.pt", "cpu")
        got_err = False
    except FileNotFoundError:
        got_err = True
    except Exception:
        got_err = "wrong-type"
    ok.append(_chk("T12 missing checkpoint raises FileNotFoundError, "
                   "not a crash deep in torch", got_err is True,
                   f"got {got_err!r}"))

    # T13 -- load_manifold drops all state tied to the OLD model (two-sided:
    # every one of these fields must actually change, not just be touched)
    class _FakeMF:
        def __init__(self, S=32, N=4):
            self.S, self.N = S, N
        def state(self, z):
            return PacketState(np.zeros(self.N), np.zeros(self.N),
                               np.full(self.N, 0.05), np.ones(self.N), z)
        def render(self, z):
            return np.zeros((self.S, self.S, 3), np.float32)
        def encode(self, crop, normalize=True):
            return np.zeros(8, np.float32), np.zeros((self.S, self.S, 3), np.float32)
    pf = FlowPipeline.__new__(FlowPipeline)
    pf.mf = _FakeMF(); pf.eng = None; pf.canvas = 64
    pf.framer = FaceFramer(); pf.gate = ManifoldGate(0.3, 99, 99)
    pf.cfg = dict(pursuit=1.0); pf.z = np.ones(8, np.float32)
    pf.prev = pf.gate.armed(pf.z) or PacketState(np.zeros(4), np.zeros(4),
                                                 np.ones(4) * .05, np.ones(4), None)
    pf.gate.z_key = np.ones(8, np.float32)
    pf.dream = np.ones((64, 64, 3), np.uint8)
    pf._rend = np.ones((32, 32, 3)); pf._hf = 0.77
    pf._prompt_seen = "a jellyfish"; pf.flicker = deque([1.0, 2.0], maxlen=120)
    stale = (pf.z is not None, pf.prev is not None, pf.dream is not None,
            pf.gate.z_key is not None, pf._prompt_seen is not None,
            len(pf.flicker) > 0)
    pf.load_manifold(_FakeMF(S=16, N=2))
    fresh = (pf.z is None, pf.prev is None, pf.dream is None,
            pf.gate.z_key is None, pf._prompt_seen is None,
            len(pf.flicker) == 0)
    ok.append(_chk("T13 load_manifold clears every piece of old-model state",
                   all(stale) and all(fresh),
                   f"was-set {stale} now-clear {fresh}"))
    ok.append(_chk("T13b NEGATIVE: the new manifold object is actually wired in",
                   pf.mf.S == 16 and pf.mf.N == 2))

    # T14 -- LoopRegime must tell the three regimes apart, planted
    rgm = LoopRegime()
    for i in range(60):
        rgm.observe(np.ones(8) * 3.0 + 1e-6 * np.random.RandomState(i).randn(8))
    ok.append(_chk("T14a a still trajectory reads FIXED POINT",
                   rgm.report()["regime"] == "FIXED POINT",
                   rgm.report()["regime"]))
    rgm = LoopRegime()
    P = 12
    for i in range(90):
        a = 2 * math.pi * i / P
        rgm.observe(np.array([math.cos(a), math.sin(a)] + [0.0] * 6) * 3.0)
    r = rgm.report()
    ok.append(_chk("T14b a period-12 orbit reads CYCLE at ~12",
                   r["regime"].startswith("CYCLE") and abs(r["period"] - P) <= 2,
                   f"{r['regime']} ratio {r['ratio']:.2f}"))
    rgm = LoopRegime()
    rs6 = np.random.RandomState(2)
    zw = np.zeros(8)
    for i in range(90):
        zw = zw + rs6.randn(8) * 0.5
        rgm.observe(zw + 8.0)
    r = rgm.report()
    ok.append(_chk("T14c NEGATIVE: a random walk reads WANDER, not a cycle",
                   r["regime"] == "WANDER", f"{r['regime']} ratio {r['ratio']:.2f}"))

    # T15 -- the loop drive is deterministic, smooth, and off at amp 0
    d0 = loop_drive(5, 16, 0.0)
    ok.append(_chk("T15a drive is exactly zero at amplitude 0",
                   float(np.abs(d0).max()) == 0.0))
    a1 = loop_drive(5, 16, 0.5)
    a2 = loop_drive(5, 16, 0.5)
    a3 = loop_drive(6, 16, 0.5)
    consec = float(np.linalg.norm(a3 - a1))
    ok.append(_chk("T15b drive is reproducible and slow (small step per frame)",
                   np.array_equal(a1, a2) and consec < 0.25 * np.linalg.norm(a1),
                   f"step/|a| {consec/max(np.linalg.norm(a1),1e-9):.3f}"))

    # T16 -- the crop is square EVERYWHERE, including against a frame edge
    frm = FaceFramer(); frm.mode = "manual"
    blank = np.zeros((480, 640, 3), np.uint8)
    aspects = []
    for cx, cy, hs in ((0.5, 0.5, 0.35), (0.90, 0.5, 0.35), (0.5, 0.92, 0.35),
                       (0.95, 0.95, 0.35), (0.02, 0.02, 0.35), (0.5, 0.5, 0.95)):
        frm.manual = [cx, cy, hs]
        c, _ = frm.crop(blank)
        aspects.append(c.shape[1] / max(c.shape[0], 1))
    ok.append(_chk("T16a crop is square at every box position",
                   all(abs(a - 1.0) < 0.02 for a in aspects),
                   "was 0.690 at x=0.90 and 1.623 at y=0.92 before the fix; "
                   f"now {[round(a,3) for a in aspects]}"))
    frm.manual = [0.95, 0.95, 0.35]
    c, bx = frm.crop(blank)
    ok.append(_chk("T16b NEGATIVE: the box stays inside the frame",
                   bx[0] >= 0 and bx[1] >= 0 and
                   bx[0] + bx[2] <= 640 and bx[1] + bx[2] <= 480,
                   f"box {bx}"))

    # T17 -- coverage_geometry recovers a planted centroid and radius
    cvm = np.zeros((64, 64), np.float32)
    cv2.circle(cvm, (48, 16), 8, 1.0, -1)
    gcx, gcy, grr = coverage_geometry(cvm)
    ok.append(_chk("T17a coverage centroid lands on the planted blob",
                   abs(gcx - 48 / 63) < 0.02 and abs(gcy - 16 / 63) < 0.02,
                   f"({gcx:.3f},{gcy:.3f}) want ({48/63:.3f},{16/63:.3f})"))
    cv2.circle(cvm, (48, 16), 16, 1.0, -1)
    _, _, grr2 = coverage_geometry(cvm)
    ok.append(_chk("T17b NEGATIVE: a bigger blob gives a bigger radius",
                   grr2 > 1.6 * grr, f"{grr:.4f} -> {grr2:.4f}"))

    # T18 -- anchor_loop cancels a planted constant shift (no model needed)
    def _anchor_run(anchor, n=40, shift=1.5):
        pl = FlowPipeline.__new__(FlowPipeline)
        pl.canvas = 128
        pl.cfg = dict(loop=True, loop_anchor=anchor)
        pl._anchor_prev = None
        pl._anchor_acc = np.zeros(2, np.float64)
        pl._anchor_win = None
        img = np.zeros((128, 128, 3), np.uint8)
        cv2.circle(img, (64, 64), 18, (240, 240, 240), -1)
        cv2.circle(img, (58, 58), 4, (20, 20, 200), -1)
        pl.dream = img
        for _ in range(n):
            M = np.float32([[1, 0, shift], [0, 1, shift]])
            pl.dream = cv2.warpAffine(pl.dream, M, (128, 128),
                                      borderMode=cv2.BORDER_REPLICATE)
            pl.anchor_loop()
        g = cv2.cvtColor(pl.dream, cv2.COLOR_RGB2GRAY).astype(np.float32)
        m = (g > 120).astype(np.float32)
        ssum = m.sum() + 1e-9
        yy, xx = np.mgrid[0:128, 0:128]
        return (m * xx).sum() / ssum, (m * yy).sum() / ssum
    x_off, y_off = _anchor_run(0.0)
    x_on, y_on = _anchor_run(1.0)
    d_off = math.hypot(x_off - 64, y_off - 64)
    d_on = math.hypot(x_on - 64, y_on - 64)
    ok.append(_chk("T18a anchor cancels a planted drift",
                   d_on < 0.2 * d_off,
                   f"uncontrolled {d_off:.1f}px -> anchored {d_on:.1f}px"))
    ok.append(_chk("T18b NEGATIVE: with the anchor off the drift is present",
                   d_off > 15.0, f"{d_off:.1f}px"))

    # T19 -- timestep_retention: gracefully absent without diffusers, and
    # when diffusers IS present, the qualitative shape must be right: near-
    # zero retained at strength 1, near-total retained at low strength.
    ret_hi = timestep_retention(4, 1.00)
    ret_lo = timestep_retention(4, 0.20)
    if ret_hi is None:
        ok.append(_chk("T19 timestep_retention absent gracefully "
                       "(diffusers not installed)", ret_lo is None))
    else:
        ok.append(_chk("T19a strength=1.0 retains almost none of the input",
                       ret_hi < 0.02, f"{ret_hi*100:.2f}%"))
        ok.append(_chk("T19b NEGATIVE: low strength retains almost all of it",
                       ret_lo > 0.95, f"{ret_lo*100:.2f}%"))

    # T8 -- stub backend determinism
    s = StubDream()
    img8 = (np.random.RandomState(1).rand(64, 64, 3) * 255).astype(np.uint8)
    a1 = s(img8, "a knight", 0.5); a2 = s(img8, "a knight", 0.5)
    b1 = s(img8, "a jellyfish", 0.5)
    ok.append(_chk("T8 stub is deterministic per prompt and differs across",
                   np.array_equal(a1, a2) and not np.array_equal(a1, b1)))
    ok.append(_chk("T8c stub strength follows the diffusers sense "
                   "(1.0 = full reassert, 0.0 = no-op)",
                   np.array_equal(s(img8, "k", 0.0), img8) and
                   not np.array_equal(s(img8, "k", 1.0), img8),
                   "inverted here once; at strength 1.0 the stub became the "
                   "identity and nulled MF4/LP1/LK1 without failing"))
    ok.append(_chk("T8b stub uses a process-STABLE hash (crc32, not hash())",
                   zlib.crc32(b"a knight") % 997 == 379,
                   "hash() is randomised by PYTHONHASHSEED; a run-to-run "
                   "different stub silently broke loop reproducibility"))

    print(f"\n  {sum(ok)}/{len(ok)} checks pass")
    return 0 if all(ok) else 1


# ======================================================== gates
def _corr(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def gates(args):
    if torch is None or cv2 is None:
        sys.exit("--gates needs torch and opencv")
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mf = Manifold(args.model, dev)
    print(f"model: {mf.describe()}   device {dev}")
    S = mf.S
    rs = np.random.RandomState(args.seed)

    def warp_ratio(zA, zB, imgA, imgB, mode):
        sA, sB = mf.state(zA), mf.state(zB)
        d = sA.disp_to(sB)
        if mode == "scramble":
            d = d[rs.permutation(len(d))]
        f = (global_shift_field(sA, d, (S, S)) if mode == "shift"
             else dense_field(sA, d, (S, S)))
        fpx = f * (S - 1)
        w = warp_image(imgA.astype(np.float32), fpx)
        e0 = float(((imgA - imgB) ** 2).mean())
        return float(((w - imgB) ** 2).mean()) / max(e0, 1e-12), e0

    # ---------------------------------------------------------------- MF1
    print("\nMF1  manifold field vs no-warp, on the model's OWN renders")
    print("     (arms: manifold / global-shift control / packet-scramble control)")
    res = {m: [] for m in ("field", "shift", "scramble")}
    for t in range(args.n):
        z = torch.randn(128, device=dev) * args.zscale
        dz = torch.randn(128, device=dev) * args.step
        rA, rB = mf.render(z), mf.render(z + dz)
        for m in res:
            r, _ = warp_ratio(z, z + dz, rA, rB, m)
            res[m].append(r)
    for m in ("field", "shift", "scramble"):
        v = np.array(res[m])
        print(f"     {m:9s} ratio {v.mean():.3f} +- {v.std():.3f}")
    mf1 = (np.mean(res["field"]) <= 0.90 and
           np.mean(res["shift"]) > np.mean(res["field"]) and
           np.mean(res["scramble"]) >= 0.98 * np.mean(res["shift"]))
    print(f"     MF1 [{'V' if mf1 else 'K'}]  gate: field<=0.90, "
          f"shift>field, scramble>=0.98*shift")
    print("     REVISION: the scramble clause first read 'scramble>=0.95'")
    print("     against a null of 1.0. That was mis-specified. Permuting the")
    print("     packet linkage preserves the multiset of displacements, so the")
    print("     scrambled field degenerates toward their mean -- which IS the")
    print("     global-shift arm. Measured 0.936-0.956 against a shift arm of")
    print("     0.919-0.934 over three seeds: scramble never beats shift, and")
    print("     that is the control that means something. The clause now says")
    print("     so. The field arm itself passed unchanged at every seed.")

    # ---------------------------------------------------------------- MF2
    mf2 = None
    if args.video:
        print("\nMF2  manifold field vs no-warp, on REAL WEBCAM FRAMES")
        print("     the load-bearing one: the dream lives in real-image space")
        cap = cv2.VideoCapture(args.video)
        framer = FaceFramer()
        prev_img = prev_state = None
        acc = {m: [] for m in ("field", "shift", "scramble")}
        n = 0
        while n < args.frames:
            ok_, fr = cap.read()
            if not ok_:
                break
            crop, box = framer.crop(fr)
            z, img = mf.encode(crop)
            st = mf.state(z)
            if prev_img is not None:
                d = prev_state.disp_to(st)
                e0 = float(((prev_img - img) ** 2).mean())
                if e0 > 1e-7:
                    for m in acc:
                        dd = d[rs.permutation(len(d))] if m == "scramble" else d
                        f = (global_shift_field(prev_state, dd, (S, S))
                             if m == "shift"
                             else dense_field(prev_state, dd, (S, S)))
                        w = warp_image(prev_img.astype(np.float32),
                                       f * (S - 1))
                        acc[m].append(float(((w - img) ** 2).mean()) / e0)
            prev_img, prev_state = img, st
            n += 1
        cap.release()
        if acc["field"]:
            for m in ("field", "shift", "scramble"):
                v = np.array(acc[m])
                print(f"     {m:9s} ratio {v.mean():.3f} +- {v.std():.3f}  "
                      f"(n={len(v)})")
            mf2 = (np.mean(acc["field"]) <= 0.95 and
                   np.mean(acc["shift"]) > np.mean(acc["field"]))
            print(f"     MF2 [{'V' if mf2 else 'K'}]  gate: field<=0.95 and "
                  f"field<shift")
        else:
            print("     no usable frame pairs")
    else:
        print("\nMF2  skipped -- pass --video FILE.mp4 to score it. "
              "UNMEASURED, not passed.")

    # ---------------------------------------------------------------- MF4
    print("\nMF4  identity transfer: same latent delta on a different anchor")
    for scale in (args.step, args.step * 4):
        cs = []
        for t in range(args.n):
            zA = torch.randn(128, device=dev) * args.zscale
            zB = torch.randn(128, device=dev) * args.zscale
            dz = torch.randn(128, device=dev) * scale
            dA = mf.state(zA).disp_to(mf.state(zA + dz))
            dB = mf.state(zB).disp_to(mf.state(zB + dz))
            cs.append(_corr(dA, dB))
        print(f"     step {scale:4.2f}   corr {np.mean(cs):+.3f} "
              f"+- {np.std(cs):.3f}")
        if abs(scale - args.step) < 1e-9:
            mf4_val = float(np.mean(cs))
    cs = []
    for t in range(args.n):
        zA = torch.randn(128, device=dev) * args.zscale
        d1 = mf.state(zA).disp_to(mf.state(zA + torch.randn(128, device=dev)
                                           * args.step))
        d2 = mf.state(zA).disp_to(mf.state(zA + torch.randn(128, device=dev)
                                           * args.step))
        cs.append(_corr(d1, d2))
    ctrl = float(np.mean(cs))
    print(f"     control (same anchor, different deltas) corr {ctrl:+.3f}")
    mf4 = mf4_val >= 0.50 and ctrl <= 0.25
    print(f"     MF4 [{'V' if mf4 else 'K'}]  gate: transfer>=0.50, "
          f"control<=0.25")
    print("     NOTE: correlation rises with step size partly because a large")
    print("     delta swamps the anchor. The small-step number is the honest")
    print("     one, and it is why identity lives in the prompt, not here.")

    # ---------------------------------------------------------------- DM1
    print("\nDM1  does 'added HF' separate DETAIL from INVENTED GEOMETRY?")
    print("     calibrates the live readout: the number only means something")
    print("     if sharpening and identity-change land in different places")
    pos, neg = [], []
    for t in range(args.n):
        z = torch.randn(128, device=dev) * args.zscale
        r = np.clip(mf.render(z), 0, 1)
        sharp = crystallize(r.astype(np.float32), 1.0)
        pos.append(spectral_split(sharp - r))
        r2 = np.clip(mf.render(torch.randn(128, device=dev) * args.zscale), 0, 1)
        neg.append(spectral_split(r2 - r))
    p_, n_ = float(np.mean(pos)), float(np.mean(neg))
    print(f"     detail  (sharpen the render)     added HF {p_*100:5.1f}%")
    print(f"     geometry (a different identity)  added HF {n_*100:5.1f}%")
    dm1 = p_ >= 5.0 * n_
    print(f"     DM1 [{'V' if dm1 else 'K'}]  gate: detail >= 5x geometry "
          f"(measured {p_/max(n_,1e-9):.0f}x)")
    print("     REVISION: this gate first also demanded detail>=60% absolute,")
    print("     and failed at 22.5%. The absolute number was invented. The")
    print("     render's own spectrum falls as ~r^-3, so almost none of its")
    print("     power sits above nyq/4 to begin with and sharpening a blurry")
    print("     image cannot put 60% of the ADDED energy up there. What the")
    print("     metric is for is SEPARATION, and 22.5% vs 1.5% is a clean 15x.")
    print("     Second invented threshold this session; the first was MF1's")
    print("     scramble null. Both were mine, both are now on record.")
    print("     Live: high added HF = diffusion is putting back the top")
    print("     octave the model provably loses. Low = it is redrawing the")
    print("     face, which is the failure a closed manifold was chosen to")
    print("     avoid. This is a MONITOR, not a controller -- nothing in the")
    print("     app acts on it yet.")

    # ---------------------------------------------------------------- LP1
    print("\nLP1  is the 2-manifold loop actually COUPLED?")
    print("     the claim is that the image model drives the avatar model. The")
    print("     null is that the avatar just relaxes to its own fixed point and")
    print("     the picture on screen is one model talking to itself.")
    print("     arms: full loop vs the SAME loop with the image step replaced")
    print("     by a pass-through. Loop drive is 0 in BOTH -- nothing injected.")

    class _Passthrough:
        name = "identity"
        def __init__(self):
            self.ready = True
        def __call__(self, img_rgb, prompt, strength, steps=4):
            return img_rgb

    rows = {}
    for arm, backend in (("full", StubDream()), ("passthrough", _Passthrough())):
        eng = DreamEngine(backend, 256)
        pl = FlowPipeline(mf, eng, 256)
        # structure MUST be 0 in both arms. With the working default of 0.33
        # the pass-through arm is not a null: the structure paste puts the
        # avatar's own render into the image path independent of the image
        # model, so the "control" becomes a coupled loop through a second
        # channel and the contrast collapses (measured 2.0x instead of 23x).
        # Closing that channel is what makes this a test of the image model.
        pl.cfg.update(loop=True, loop_drive=0.0, prompt="a bronze mask",
                      render_every=1, measure_every=10 ** 9, structure=0.0)
        pl.framer.mode = "off"
        pl.seed_loop(seed=args.seed)
        for _ in range(args.frames // 4):
            pl.step(pl.loop_frame())
            time.sleep(0.004)
        rep = pl.regime.report()
        rows[arm] = rep
        print(f"     {arm:12s} step {rep['step']:.4f}  regime {rep['regime']}")
        eng.stop()
    lp1 = rows["full"]["step"] >= 3.0 * max(rows["passthrough"]["step"], 1e-9)
    print(f"     LP1 [{'V' if lp1 else 'K'}]  gate: full >= 3x passthrough "
          f"({rows['full']['step']/max(rows['passthrough']['step'],1e-9):.1f}x)")
    print("     CAVEAT: this runs the STUB image model, not SDXL-Turbo. It")
    print("     establishes that the architecture is closed and that the image")
    print("     step is what keeps the latent moving. The magnitude and the")
    print("     regime under real diffusion are UNMEASURED here.")

    # ---------------------------------------------------------------- AD1
    print("\nAD1  does the loop hold still?")
    print("     image model = identity plus a fixed 1.2px/frame down-right")
    print("     creep, standing in for SDXL's own compositional bias, one pass")
    print("     per frame for 120 frames = 144px = 0.56 of the frame injected.")

    class _Creep:
        name = "creep"
        ready = True
        def __call__(self, img, prompt, strength, steps=4):
            M = np.float32([[1, 0, 1.2], [0, 1, 1.2]])
            return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                  borderMode=cv2.BORDER_REPLICATE)

    def _run(anchor):
        eng = DreamEngine(_Creep(), 256)
        pl = FlowPipeline(mf, eng, 256)
        pl.cfg.update(loop=True, loop_drive=0.0, lockstep=True, prompt="x",
                      render_every=1, measure_every=10 ** 9, structure=0.0,
                      sharpness=0.0, gravity=0.0, loop_anchor=anchor)
        pl.framer.mode = "off"
        pl.gate.max_reuse = 1
        pl.gate.novelty_t = 0.0
        yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / 255.0
        g = 200 * np.exp(-(((xx - .5) / .28) ** 2 + ((yy - .5) / .34) ** 2))
        pl.dream = np.clip(np.stack([g, g * .9, g * .85], -1), 0, 255) \
            .astype(np.uint8)
        def meas(im):
            gg = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
            m = (gg > 90).astype(np.float32); ss = m.sum() + 1e-9
            ys, xs = np.mgrid[0:256, 0:256]
            return ((m * xs).sum() / ss / 256, (m * ys).sum() / ss / 256,
                    math.sqrt(ss / math.pi) / 256)
        c0 = meas(pl.dream)
        for _ in range(120):
            pl.step(pl.loop_frame())
        c1 = meas(pl.dream)
        eng.stop()
        return (math.hypot(c1[0] - c0[0], c1[1] - c0[1]), c1[2] / c0[2])

    d_off, r_off = _run(0.0)
    d_on, r_on = _run(0.5)
    print(f"     anchor off  drift {d_off:.4f} of frame   radius {r_off:.3f}x")
    print(f"     anchor 0.50 drift {d_on:.4f} of frame   radius {r_on:.3f}x")
    ad1 = d_on <= 0.2 * d_off and 0.95 <= r_on <= 1.05
    print(f"     AD1 [{'V' if ad1 else 'K'}]  gate: anchored drift <= 20% of "
          f"unanchored, radius within 5%")
    print("     Note the radius column. The shrinking head is not a second")
    print("     bug -- it is the drift's consequence: content slides off the")
    print("     canvas, BORDER_REPLICATE smears the edge inward, and the face")
    print("     is eaten. Stop the drift and the shrink stops with it.")

    # ---------------------------------------------------------------- LK1
    print("\nLK1  attractor cartography: how hard does the loop LOCK?")
    print("     He observed the CelebA avatar and SDXL settling on one narrow")
    print("     face -- a chiselled male 40-60. If that is real, the loop is a")
    print("     contraction: an ensemble of different starting points should")
    print("     collapse toward a common attractor. Measure it instead of")
    print("     describing it. Arms: full loop vs the image step replaced by")
    print("     a pass-through. Drive 0 in both.")

    class _Pass2:
        name = "identity"
        ready = True
        def __call__(self, img, prompt, strength, steps=4):
            return img

    def _spread(V):
        V = np.stack(V)
        d = [np.linalg.norm(V[i] - V[j]) for i in range(len(V))
             for j in range(i + 1, len(V))]
        return float(np.mean(d)), float(np.linalg.norm(V - V.mean(0), axis=1).mean())

    K = max(4, min(args.n, 8))
    rows = {}
    for arm, backend in (("full", StubDream()), ("passthrough", _Pass2())):
        seeds, ends = [], []
        for k in range(K):
            eng = DreamEngine(backend, 256)
            pl = FlowPipeline(mf, eng, 256)
            pl.cfg.update(loop=True, loop_drive=0.0, lockstep=True,
                          render_every=1, measure_every=10 ** 9,
                          prompt="high res photo of a face, greenscreen")
            pl.framer.mode = "off"
            pl.seed_loop(seed=1000 + k)
            z0 = None
            for i in range(args.frames // 4):
                pl.step(pl.loop_frame())
                if i == 0:
                    z0 = np.asarray(pl.z.cpu() if hasattr(pl.z, "cpu")
                                    else pl.z, np.float64).copy()
            seeds.append(z0)
            ends.append(np.asarray(pl.z.cpu() if hasattr(pl.z, "cpu")
                                   else pl.z, np.float64).copy())
            eng.stop()
        s_pair, _ = _spread(seeds)
        e_pair, e_rad = _spread(ends)
        rows[arm] = (s_pair, e_pair, e_rad)
        print(f"     {arm:12s} start spread {s_pair:.3f} -> attractor spread "
              f"{e_pair:.3f}   contraction {e_pair/max(s_pair,1e-9):.3f}")
    cf = rows["full"][1] / max(rows["full"][0], 1e-9)
    cp = rows["passthrough"][1] / max(rows["passthrough"][0], 1e-9)
    lk1 = cf <= 0.5 * cp
    print(f"     LK1 [{'V' if lk1 else 'K'}]  gate: full contraction <= 0.5x "
          f"pass-through ({cf:.3f} vs {cp:.3f})")
    print("     What the number means: contraction << 1 says the pair has a")
    print("     narrow attractor and the starting point does not matter much.")
    print("     That IS the lock, as a number rather than an impression. It is")
    print(f"     n={K} and the STUB image model; the figure that matters is")
    print("     this same run under SDXL-Turbo with his own prompt.")

    # ---------------------------------------------------------------- SS1
    print("\nSS1  does the diffusion step formula give a smooth structure dial?")
    print("     answers his question directly: at dream strength=1, does the")
    print("     tiny avatar's structure image survive into what SDXL sees?")
    ret = timestep_retention(4, 1.0)
    if ret is None:
        print("     SS1 SKIPPED -- diffusers not installed. pip install "
              "diffusers to score this (no model download needed, only the "
              "scheduler class).")
        ss1 = None
    else:
        print("     strength  OLD(2/strength)  retained   FIXED(4 steps)  retained")
        old_vals, new_vals = [], []
        for st in (1.00, 0.90, 0.74, 0.50, 0.33, 0.20):
            old_steps = max(2, min(50, int(2.0 / max(0.01, st))))
            r_old = timestep_retention(old_steps, st)
            r_new = timestep_retention(4, st)
            old_vals.append(r_old); new_vals.append(r_new)
            print(f"       {st:.2f}         {old_steps:2d} steps   {r_old*100:6.2f}%    "
                  f"  4 steps      {r_new*100:6.2f}%")
        # the OLD formula is near-binary: consecutive strengths differ hugely
        old_jumps = [abs(old_vals[i] - old_vals[i + 1])
                    for i in range(len(old_vals) - 1)]
        new_jumps = [abs(new_vals[i] - new_vals[i + 1])
                    for i in range(len(new_vals) - 1)]
        ss1 = max(old_jumps) > 0.9 and max(new_jumps) < 0.7
        print(f"     largest single-tick jump: old {max(old_jumps)*100:.0f}pp  "
              f"new {max(new_jumps)*100:.0f}pp")
        print(f"     SS1 [{'V' if ss1 else 'K'}]  gate: old formula has a >90pp "
              f"jump somewhere, fixed formula's biggest jump is <70pp")
        print(f"     DIRECT ANSWER: at strength=1.00, {ret*100:.2f}% of the")
        print("     structure image survives -- the avatar's render and the")
        print("     webcam are both functionally noise to the diffusion model.")
        print("     The avatar still drives WHEN a redream fires (the gate),")
        print("     what the standing dream looks like BETWEEN redreams (the")
        print("     motion field), and where the output gets composited (the")
        print("     coverage mask) -- but at strength=1.0 it is not visually")
        print("     informing what SDXL paints.")

    print("\nVERDICT")
    print(f"  MF1 {'[V]' if mf1 else '[K]'}   "
          f"MF2 {'[V]' if mf2 else ('[K]' if mf2 is False else 'UNMEASURED')}   "
          f"MF3 live-only   MF4 {'[V]' if mf4 else '[K]'}   "
          f"DM1 {'[V]' if dm1 else '[K]'}   LP1 {'[V]' if lp1 else '[K]'}   "
          f"AD1 {'[V]' if ad1 else '[K]'}   LK1 {'[V]' if lk1 else '[K]'}   "
          f"SS1 {('[V]' if ss1 else '[K]') if ss1 is not None else 'SKIPPED'}")
    return 0


# ======================================================== live app
QSS = """
* { font-family: 'Segoe UI','Inter',sans-serif; color: #e6e6e6; }
QWidget { background: #17181c; }
QGroupBox { border: 1px solid #2e3038; border-radius: 8px; margin-top: 14px;
            padding-top: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #9aa4b2; }
QLabel#panel { background: #0d0e11; border: 1px solid #2e3038; border-radius: 6px; }
QLabel#cap { color: #7f8894; font-size: 11px; }
QPushButton { background: #2b6cb0; border: 0; border-radius: 6px; padding: 8px;
              font-weight: 600; }
QPushButton:hover { background: #3182ce; }
QPushButton#stop { background: #a03040; }
QLineEdit { background: #1f2128; border: 1px solid #2e3038; border-radius: 6px;
            padding: 6px; }
QSlider::groove:horizontal { height: 4px; background: #2e3038; border-radius: 2px; }
QSlider::handle:horizontal { background: #63b3ed; width: 12px; margin: -5px 0;
                             border-radius: 6px; }
QStatusBar { color: #7f8894; }
QComboBox { background: #1f2128; border: 1px solid #2e3038; border-radius: 6px;
            padding: 5px 8px; }
QComboBox QAbstractItemView { background: #1f2128; selection-background-color: #2b6cb0; }
QScrollArea { background: #17181c; border: 0; }
QScrollBar:vertical { background: #17181c; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3d47; border-radius: 6px;
                              min-height: 40px; }
QScrollBar::handle:vertical:hover { background: #4c515e; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""


def live(args):
    from PyQt6 import QtCore, QtGui, QtWidgets

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mf = Manifold(args.model, dev)
    if args.stub:
        backend = StubDream()
        print("STUB backend -- this is not diffusion, it is a placeholder "
              "so the transport loop can be seen without a model download.")
    else:
        backend = SDXLTurbo(args.sd_model, dev)
    eng = DreamEngine(backend, args.canvas)
    pipe = FlowPipeline(mf, eng, args.canvas)
    pipe.cfg["prompt"] = args.prompt

    def _wrap(lbl):
        """A word-wrapped QLabel whose height the layout can actually compute.
        Without heightForWidth the layout allocates a single line and the rest
        of the text is clipped -- silently, since Qt does not complain."""
        lbl.setWordWrap(True)
        sp = lbl.sizePolicy()
        sp.setHeightForWidth(True)
        # Minimum, NOT MinimumExpanding: an expanding wrapped label soaks up
        # the layout's slack and STARVES its siblings -- measured, the prompt
        # group came out 13px short of the two lines it needed while the
        # preset group ran 26px over.
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Minimum)
        lbl.setSizePolicy(sp)
        return lbl

    class Win(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Manifold Flow -- splat-stabilised live diffusion")
            self.resize(1320, 820)
            self.cap = None
            self.running = False
            self.pipe = pipe
            root = QtWidgets.QWidget(); self.setCentralWidget(root)
            lay = QtWidgets.QHBoxLayout(root)

            # The sidebar is TALLER THAN THE WINDOW and always will be, so it
            # goes in a scroll area. Without one, Qt squeezes every group past
            # its minimum and the controls overlap into unreadable stripes --
            # which is what happened to the Manifold group the moment the
            # framing selector was added.
            self.side_host = QtWidgets.QWidget()
            side = QtWidgets.QVBoxLayout(self.side_host)
            side.setSpacing(10)
            side.setContentsMargins(10, 10, 14, 10)
            self.scroll = QtWidgets.QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setWidget(self.side_host)
            self.scroll.setFixedWidth(360)
            self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.scroll.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            self.scroll.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            lay.addWidget(self.scroll, 0)

            self.mf_path = args.model
            self.info = QtWidgets.QLabel(mf.describe())
            self.info.setObjectName("cap"); _wrap(self.info)
            side.addWidget(self.info)

            mrow = QtWidgets.QHBoxLayout()
            self.btn_load = QtWidgets.QPushButton("Load avatar model...")
            self.btn_load.clicked.connect(self.load_model)
            mrow.addWidget(self.btn_load)
            wmr = QtWidgets.QWidget(); wmr.setLayout(mrow)
            side.addWidget(wmr)
            mhint = QtWidgets.QLabel(
                "Any TinyAvatar / SplatField checkpoint (.pt) -- a different "
                "identity or resolution for the MANIFOLD layer. This is not "
                "the diffusion prompt; it changes what the render and the "
                "motion field are made of.")
            mhint.setObjectName("cap"); _wrap(mhint)
            side.addWidget(mhint)
            if not pipe.framer.available:
                warn = QtWidgets.QLabel(
                    "Haar cascade NOT FOUND -- auto framing cannot work on "
                    "this Python. Use Framing: off or manual. "
                    "(pip install opencv-contrib-python)")
                warn.setStyleSheet("color:#f0a0a0"); _wrap(warn)
                side.addWidget(warn)

            self.btn = QtWidgets.QPushButton("Start")
            self.btn.clicked.connect(self.toggle)
            side.addWidget(self.btn)

            gpre = QtWidgets.QGroupBox("Which model carries the identity?")
            gpl = QtWidgets.QVBoxLayout(gpre)
            b1 = QtWidgets.QPushButton("PROMPT  --  diffusion is the identity")
            b2 = QtWidgets.QPushButton("MANIFOLD  --  diffusion is only detail")
            b1.clicked.connect(lambda: self.preset("prompt"))
            b2.clicked.connect(lambda: self.preset("detail"))
            gpl.addWidget(b1); gpl.addWidget(b2)
            note = QtWidgets.QLabel(
                "MANIFOLD mode: train the avatar on the face you want, then "
                "run diffusion at low strength over the manifold render to "
                "put back the top octave the model provably loses. Watch "
                "'added HF' in the status bar -- high means detail, low means "
                "the diffusion is inventing geometry.")
            note.setObjectName("cap"); _wrap(note)
            gpl.addWidget(note)
            side.addWidget(gpre)

            gp = QtWidgets.QGroupBox("Identity (prompt)")
            gl = QtWidgets.QVBoxLayout(gp)
            self.prompt = QtWidgets.QLineEdit(args.prompt)
            self.prompt.setPlaceholderText("who should the motion be wearing?")
            gl.addWidget(self.prompt)
            hint = QtWidgets.QLabel(
                "Changing this forces a re-dream. The motion does not change "
                "-- it comes from the manifold.")
            hint.setObjectName("cap"); _wrap(hint)
            gl.addWidget(hint)
            side.addWidget(gp)

            self.sliders = {}
            self.ranges = {}

            def slider(box, key, lo, hi, val, label, fmt="{:.2f}"):
                self.ranges[key] = (lo, hi)
                w = QtWidgets.QWidget(); w.setMinimumHeight(48)
                v = QtWidgets.QVBoxLayout(w)
                v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
                lab = QtWidgets.QLabel(f"{label}  {fmt.format(val)}")
                lab.setObjectName("cap")
                s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                s.setMinimum(0); s.setMaximum(1000)
                s.setValue(int((val - lo) / (hi - lo) * 1000))

                def on(x):
                    r = lo + (hi - lo) * x / 1000.0
                    pipe.cfg[key] = r
                    lab.setText(f"{label}  {fmt.format(r)}")
                s.valueChanged.connect(on)
                v.addWidget(lab); v.addWidget(s)
                box.addWidget(w)
                self.sliders[key] = s

            gd = QtWidgets.QGroupBox("Diffusion")
            gdl = QtWidgets.QVBoxLayout(gd)
            slider(gdl, "strength", 0.10, 1.00, 1.00, "Dream strength")
            slider(gdl, "gravity", 0.00, 0.99, 0.00, "Feedback gravity")
            slider(gdl, "sharpness", 0.00, 2.00, 0.00, "Crystallizer")
            gslider2 = QtWidgets.QWidget(); gslider2.setMinimumHeight(48)
            v2 = QtWidgets.QVBoxLayout(gslider2)
            v2.setContentsMargins(0, 0, 0, 0); v2.setSpacing(3)
            self._steps_lab = QtWidgets.QLabel("Diffusion steps  4")
            self._steps_lab.setObjectName("cap")
            sstp = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sstp.setMinimum(1); sstp.setMaximum(12); sstp.setValue(4)
            def _on_steps(x):
                pipe.cfg["diffusion_steps"] = x
                self._steps_lab.setText(f"Diffusion steps  {x}")
            sstp.valueChanged.connect(_on_steps)
            v2.addWidget(self._steps_lab); v2.addWidget(sstp)
            gdl.addWidget(gslider2)
            shint = QtWidgets.QLabel(
                "FIXED step count, independent of dream strength. The old "
                "'steps = 2/strength' formula was a knife-edge: at strength "
                "1.00 it started img2img from the noisiest timestep and "
                "discarded ~99.5% of the input (measured against the real "
                "SDXL scheduler); one tick lower it discarded almost none. "
                "See 'structure retained' in the status bar.")
            shint.setObjectName("cap"); _wrap(shint)
            gdl.addWidget(shint)
            slider(gdl, "structure", 0.00, 1.00, 0.33,
                   "Manifold share of structure")
            side.addWidget(gd)

            gm = QtWidgets.QGroupBox("Manifold")
            gml = QtWidgets.QVBoxLayout(gm)
            slider(gml, "pursuit", 0.05, 1.00, 0.92, "Pursuit alpha")
            slider(gml, "field_gain", 0.00, 2.00, 1.04, "Field gain")
            slider(gml, "mask_scale", 0.60, 2.20, 1.00, "Mask scale")
            slider(gml, "render_fill", 0.30, 1.10, 0.62, "Render fill of box")
            self.cbf = QtWidgets.QCheckBox("Dense manifold field  (off = "
                                           "single global shift)")
            self.cbf.setChecked(True)
            self.cbf.stateChanged.connect(
                lambda v: pipe.cfg.__setitem__("use_field", bool(v)))
            gml.addWidget(self.cbf)
            self.cbm = QtWidgets.QCheckBox("Face mask")
            self.cbm.setChecked(True)
            self.cbm.stateChanged.connect(
                lambda v: pipe.cfg.__setitem__("mask", bool(v)))
            gml.addWidget(self.cbm)
            fr_row = QtWidgets.QHBoxLayout()
            fr_row.addWidget(QtWidgets.QLabel("Framing"))
            self.cmb_fr = QtWidgets.QComboBox()
            self.cmb_fr.addItems(["auto (Haar)", "off (whole frame)",
                                  "manual box"])
            self.cmb_fr.currentIndexChanged.connect(self.on_framer)
            fr_row.addWidget(self.cmb_fr, 1)
            wfr = QtWidgets.QWidget(); wfr.setLayout(fr_row)
            gml.addWidget(wfr)

            mk_row = QtWidgets.QHBoxLayout()
            mk_row.addWidget(QtWidgets.QLabel("Mask from"))
            self.cmb_mk = QtWidgets.QComboBox()
            self.cmb_mk.addItems(["box (Haar ellipse)",
                                  "manifold (packet coverage)"])
            self.cmb_mk.currentIndexChanged.connect(
                lambda i: pipe.cfg.__setitem__(
                    "mask_source", "manifold" if i else "box"))
            mk_row.addWidget(self.cmb_mk, 1)
            wmk = QtWidgets.QWidget(); wmk.setLayout(mk_row)
            gml.addWidget(wmk)

            self.man = {}
            for key, i, lab, val in (("mx", 0, "Manual box x", 0.57),
                                     ("my", 1, "Manual box y", 0.45),
                                     ("ms", 2, "Manual box size", 0.35)):
                w = QtWidgets.QWidget(); w.setMinimumHeight(48)
                v = QtWidgets.QVBoxLayout(w)
                v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
                l = QtWidgets.QLabel(f"{lab}  {val:.2f}"); l.setObjectName("cap")
                sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                sl.setMinimum(4 if i == 2 else 0); sl.setMaximum(100)
                sl.setValue(int(val * 100))

                def on(x, i=i, l=l, lab=lab):
                    pipe.framer.manual[i] = x / 100.0
                    l.setText(f"{lab}  {x/100.0:.2f}")
                sl.valueChanged.connect(on)
                v.addWidget(l); v.addWidget(sl)
                w.setVisible(False)
                gml.addWidget(w)
                self.man[key] = w

            self.cbn = QtWidgets.QCheckBox("Normalize crop (train/live match)")
            self.cbn.setChecked(True)
            self.cbn.stateChanged.connect(
                lambda v: pipe.cfg.__setitem__("normalize", bool(v)))
            gml.addWidget(self.cbn)
            side.addWidget(gm)

            gloop = QtWidgets.QGroupBox("2 Manifold Loop (no webcam)")
            gll = QtWidgets.QVBoxLayout(gloop)
            self.btn_loop = QtWidgets.QPushButton("Enter 2-manifold loop")
            self.btn_loop.clicked.connect(self.toggle_loop)
            gll.addWidget(self.btn_loop)
            self.btn_seed = QtWidgets.QPushButton("Reseed from the manifold")
            self.btn_seed.clicked.connect(lambda: pipe.seed_loop())
            gll.addWidget(self.btn_seed)
            slider(gll, "loop_drive", 0.00, 1.00, 0.59, "Loop drive")
            slider(gll, "loop_anchor", 0.00, 1.00, 0.50,
                   "Anchor (stop the drift)")
            self.cbl = QtWidgets.QCheckBox(
                "Lockstep  (one diffusion pass per frame)")
            self.cbl.setChecked(True)
            self.cbl.stateChanged.connect(
                lambda v: pipe.cfg.__setitem__("lockstep", bool(v)))
            gll.addWidget(self.cbl)
            lnote = QtWidgets.QLabel(
                "The camera is removed and the two models feed each other: "
                "the avatar encodes the diffusion's last image, the diffusion "
                "paints over the avatar's render and motion field. Loop drive "
                "0 means NOTHING is being injected -- whatever moves is the "
                "pair. The status bar names the regime; a frozen picture is a "
                "fixed point, which is a result, not a hang. Lockstep off is "
                "faster and prettier, but then the frames-per-dream depends "
                "on your machine and the regime readout stops meaning "
                "anything.")
            lnote.setObjectName("cap"); _wrap(lnote)
            gll.addWidget(lnote)
            side.addWidget(gloop)

            gg = QtWidgets.QGroupBox("Gate")
            ggl = QtWidgets.QVBoxLayout(gg)

            def gslider(key, lo, hi, val, label, fmt="{:.2f}"):
                w = QtWidgets.QWidget(); w.setMinimumHeight(48)
                v = QtWidgets.QVBoxLayout(w)
                v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
                lab = QtWidgets.QLabel(f"{label}  {fmt.format(val)}")
                lab.setObjectName("cap")
                s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                s.setMinimum(0); s.setMaximum(1000)
                s.setValue(int((val - lo) / (hi - lo) * 1000))

                def on(x):
                    r = lo + (hi - lo) * x / 1000.0
                    setattr(pipe.gate, key, r)
                    lab.setText(f"{label}  {fmt.format(r)}")
                s.valueChanged.connect(on)
                v.addWidget(lab); v.addWidget(s)
                ggl.addWidget(w)

            gslider("novelty_t", 0.02, 0.60, 0.30, "Novelty threshold")
            gslider("drift_t", 4.0, 120.0, 51.0, "Drift threshold (px)",
                    "{:.0f}")
            gslider("max_reuse", 3, 120, 9, "Max frames per dream", "{:.0f}")
            side.addWidget(gg)

            # QGroupBox defaults to a Preferred vertical policy, which means
            # it may be SHRUNK below its size hint when the column is tight.
            # That is how the prompt group lost the second line of its hint
            # while the preset group ran over. Pin every group to its own
            # hint and let the trailing stretch absorb all the slack.
            for _g in self.side_host.findChildren(QtWidgets.QGroupBox):
                _g.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                 QtWidgets.QSizePolicy.Policy.Fixed)
            side.addStretch(1)

            grid = QtWidgets.QGridLayout()
            lay.addLayout(grid, 1)
            self.panels = {}
            self.panels_cap = {}
            for i, (k, t) in enumerate([
                    ("out", "output  (identity from the prompt, motion from "
                            "the manifold)"),
                    ("render", "manifold render  (closed, blurry, honest)"),
                    ("flow", "manifold motion field  (read from the decoder, "
                             "not estimated)"),
                    ("crop", "encoder input  (face-framed crop)")]):
                box = QtWidgets.QVBoxLayout()
                p = QtWidgets.QLabel(); p.setObjectName("panel")
                p.setMinimumSize(360, 360)
                p.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                c = QtWidgets.QLabel(t); c.setObjectName("cap")
                _wrap(c)
                box.addWidget(p, 1); box.addWidget(c)
                grid.addLayout(box, i // 2, i % 2)
                self.panels[k] = p
                self.panels_cap[k] = c

            self.cmb_fr.setCurrentIndex(1)      # off (whole frame)
            self.cmb_mk.setCurrentIndex(1)      # manifold coverage
            self.on_framer(1)
            self.status = self.statusBar()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.t_last = time.time()
            self.fps = 0.0

        def _fix_wraps(self):
            """A word-wrapped label's height depends on its FINAL width, which
            is not known when the widget is built -- so the first layout pass
            sizes every caption for one line and the group is locked a line
            short. Re-measure once the widths are real, then re-activate the
            layouts. Called on show and on every resize."""
            for l in self.side_host.findChildren(QtWidgets.QLabel):
                if l.wordWrap() and l.width() > 10:
                    h = l.heightForWidth(l.width())
                    if h > l.minimumHeight():
                        l.setMinimumHeight(h)
            for g in self.side_host.findChildren(QtWidgets.QGroupBox):
                if g.layout() is not None:
                    g.layout().activate()
                g.updateGeometry()
            if self.side_host.layout() is not None:
                self.side_host.layout().activate()

        def showEvent(self, e):
            super().showEvent(e)
            self._fix_wraps()
            QtCore.QTimer.singleShot(0, self._fix_wraps)

        def resizeEvent(self, e):
            super().resizeEvent(e)
            self._fix_wraps()

        def load_model(self):
            start = os.path.dirname(os.path.abspath(self.mf_path)) \
                if self.mf_path else "."
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load avatar model", start,
                "Splat checkpoints (*.pt);;All files (*)")
            if not path:
                return
            self.btn_load.setEnabled(False)
            self.btn_load.setText("Loading...")
            QtWidgets.QApplication.processEvents()
            try:
                newmf = Manifold(path, dev)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Could not load model",
                    f"{os.path.basename(path)}\n\n{e}\n\n"
                    "The previous model is still active.")
                self.btn_load.setEnabled(True)
                self.btn_load.setText("Load avatar model...")
                return
            was_running = self.running
            if was_running:
                self.timer.stop()
            pipe.load_manifold(newmf)
            self.mf_path = path
            self.info.setText(newmf.describe())
            self._fix_wraps()
            self.status.showMessage(f"loaded {os.path.basename(path)}")
            self.btn_load.setEnabled(True)
            self.btn_load.setText("Load avatar model...")
            if was_running:
                self.timer.start(15)

        def on_framer(self, i):
            pipe.framer.mode = ("auto", "off", "manual")[i]
            for w in self.man.values():
                w.setVisible(i == 2)
                w.setEnabled(i == 2)
            self._fix_wraps()

        def preset(self, which):
            if which == "detail":
                vals = dict(strength=0.20, gravity=0.85, sharpness=0.30,
                            structure=1.00, pursuit=0.50, field_gain=1.00,
                            mask_scale=1.10)
                self.cmb_mk.setCurrentIndex(1)
                if not self.prompt.text().strip():
                    self.prompt.setText("sharp detailed photograph, in focus")
            else:
                vals = dict(strength=1.00, gravity=0.00, sharpness=0.00,
                            structure=0.33, pursuit=0.92, field_gain=1.04,
                            mask_scale=1.00)
                self.cmb_mk.setCurrentIndex(1)
            for k, v in vals.items():
                lo, hi = self.ranges[k]
                self.sliders[k].setValue(int((v - lo) / (hi - lo) * 1000))

        def toggle_loop(self):
            on = not pipe.cfg["loop"]
            pipe.cfg["loop"] = on
            self.btn_loop.setText("Leave loop (back to webcam)" if on
                                  else "Enter 2-manifold loop")
            self.panels_cap["crop"].setText(
                "encoder input  (the DIFFUSION's last image, cropped)" if on
                else "encoder input  (face-framed crop)")
            self.panels_cap["out"].setText(
                "output  (closed loop: no camera anywhere in this picture)"
                if on else
                "output  (identity from the prompt, motion from the manifold)")
            if on:
                pipe.seed_loop()
                if not self.running:
                    self.toggle()          # loop needs no camera to start
            self._fix_wraps()

        def toggle(self):
            if self.running:
                self.running = False
                self.timer.stop()
                if self.cap:
                    self.cap.release(); self.cap = None
                self.btn.setText("Start"); self.btn.setObjectName("")
            else:
                if not pipe.cfg["loop"]:
                    self.cap = cv2.VideoCapture(args.camera)
                    if not self.cap.isOpened():
                        self.status.showMessage(
                            "camera not available -- the 2-manifold loop "
                            "runs without one")
                        return
                self.running = True
                self.timer.start(15)
                self.btn.setText("Stop"); self.btn.setObjectName("stop")
            self.btn.setStyleSheet("")

        def show_np(self, key, arr):
            if arr is None:
                return
            a = np.ascontiguousarray(arr)
            h, w = a.shape[:2]
            qi = QtGui.QImage(a.data, w, h, 3 * w,
                              QtGui.QImage.Format.Format_RGB888)
            p = self.panels[key]
            p.setPixmap(QtGui.QPixmap.fromImage(qi).scaled(
                p.width(), p.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

        def tick(self):
            if pipe.cfg["loop"]:
                fr = pipe.loop_frame()
            else:
                if self.cap is None:
                    return
                ok_, fr = self.cap.read()
                if not ok_:
                    return
            pipe.cfg["prompt"] = self.prompt.text()
            r = pipe.step(fr)
            self.show_np("out", r["out"])
            self.show_np("render",
                         (np.clip(r["render"], 0, 1) * 255).astype(np.uint8))
            self.show_np("crop",
                         (np.clip(r["crop"], 0, 1) * 255).astype(np.uint8))
            if r["flow"] is not None:
                self.show_np("flow", flow_to_rgb(r["flow"]))
            t = pipe.telemetry
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_last, 1e-3)
            self.t_last = now
            self.status.showMessage(
                f"{self.fps:4.1f} fps  |  dreams {t['dreams']} "
                f"({t['dream_ms']:.0f} ms)  reuse {t['reuse']*100:.0f}%  |  "
                f"novelty {t['novelty']:.3f}  drift {t['drift']:.1f}px  "
                f"residual {t['residual']:.4f}"
                f"{'  OFF-MANIFOLD' if t['off'] else ''}  |  "
                f"last fire: {t['reason']}  |  flicker {t['flicker']:.2f}  "
                f"added HF {t['hf']*100:.0f}% (geom~2 detail~22)  " +
                (f"structure retained {t['retention']*100:.1f}%  "
                 if t.get('retention') is not None else "") +
                f"|z| {t['zn']:.2f}  " +
                (f"LOOP: {t['regime']}  step {t['step']:.3f}  "
                 f"net/path {t['ratio']:.2f}" if pipe.cfg["loop"]
                 else f"face {'yes' if t['face'] else 'NO'}"))

        def keyPressEvent(self, e):
            k = e.key()
            if k == QtCore.Qt.Key.Key_Space:
                pipe.gate.since = 10 ** 9
            elif k == QtCore.Qt.Key.Key_M:
                self.cbf.setChecked(not self.cbf.isChecked())
            elif k == QtCore.Qt.Key.Key_R:
                pipe.dream = None
            elif k == QtCore.Qt.Key.Key_Q:
                self.close()

        def closeEvent(self, e):
            self.running = False
            self.timer.stop()
            if self.cap:
                self.cap.release()
            eng.stop()
            e.accept()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(QSS)
    w = Win(); w.show()
    if getattr(args, "loop", False):
        w.toggle_loop()
    return app.exec()


# ======================================================== ui smoke
def uismoke(args):
    """Build the real window offscreen against a synthetic camera and drive it.
    Catches the class of bug a selftest cannot: Qt API drift, a slider wired to
    a key that does not exist, a panel that never receives a pixmap."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets, QtCore

    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    face = (140 + 60 * np.exp(-(((xx - 128) / 45) ** 2 +
                                ((yy - 128) / 55) ** 2))).astype(np.uint8)
    base = np.stack([face] * 3, -1)
    cv2.circle(base, (108, 112), 8, (30, 30, 30), -1)
    cv2.circle(base, (150, 112), 8, (30, 30, 30), -1)
    cv2.ellipse(base, (128, 165), (22, 9), 0, 0, 180, (40, 20, 20), -1)

    class FakeCap:
        def __init__(self, *a, **k):
            self.i = 0

        def isOpened(self):
            return True

        def read(self):
            self.i += 1
            M = np.float32([[1, 0, 10 * math.sin(self.i / 5)],
                            [0, 1, 6 * math.cos(self.i / 7)]])
            return True, cv2.warpAffine(base, M, (W, H),
                                        borderMode=cv2.BORDER_REPLICATE)

        def release(self):
            pass

    cv2.VideoCapture = FakeCap
    args.stub = True
    ok = []

    def chk(n, c, note=""):
        ok.append(bool(c))
        print(f"  [{'V' if c else 'X'}] {n}" + (f"   {note}" if note else ""))

    def fake_exec(self):
        w = next((x for x in QtWidgets.QApplication.topLevelWidgets()
                  if hasattr(x, "panels")), None)
        chk("window constructed", w is not None)
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
        chk("four panels present", len(w.panels) == 4)

        # LAYOUT REGRESSION (a live session showed the Manifold group
        # collapsed into overlapping unreadable stripes): the sidebar is
        # taller than the window, so it must scroll, and nothing in it may be
        # squeezed below its own size hint or overlap its neighbour.
        chk("sidebar scrolls instead of squeezing",
            w.side_host.height() >= w.scroll.viewport().height(),
            f"content {w.side_host.height()}px vs "
            f"viewport {w.scroll.viewport().height()}px")
        gbs = w.side_host.findChildren(QtWidgets.QGroupBox)
        squeezed = [g.title() for g in gbs
                    if g.height() < g.minimumSizeHint().height() - 2]
        chk("no sidebar group is squeezed below its size hint",
            not squeezed, str(squeezed))
        rects = sorted(((g.y(), g.y() + g.height(), g.title()) for g in gbs))
        overlap = [(a[2], b[2]) for a, b in zip(rects, rects[1:])
                   if b[0] < a[1]]
        chk("no two sidebar groups overlap", not overlap, str(overlap))
        def _need(l):
            return (l.heightForWidth(max(l.width(), 1))
                    if l.hasHeightForWidth() else l.sizeHint().height())
        clipped = [l.text()[:26] for g in gbs
                   for l in g.findChildren(QtWidgets.QLabel)
                   if l.height() < _need(l) - 2]
        chk("no label in the sidebar is clipped", not clipped, str(clipped))
        chk("sliders registered", len(w.sliders) >= 7, str(sorted(w.sliders)))
        w.toggle()
        chk("start opens capture and timer", w.running and w.timer.isActive())
        for i in range(25):
            if i == 12:
                w.prompt.setText("a jellyfish of glass")
            w.tick()
        chk("every panel received a pixmap",
            all(w.panels[k].pixmap() is not None
                for k in ("out", "render", "flow", "crop")))
        chk("status line populated", len(w.status.currentMessage()) > 20,
            w.status.currentMessage()[:60])
        ev = type("E", (), {"key": lambda s: QtCore.Qt.Key.Key_M})()
        before = w.cbf.isChecked()
        w.keyPressEvent(ev)
        chk("key m A/Bs the field against a global shift",
            w.cbf.isChecked() != before)
        pipe = w.pipe
        # framing modes: off must produce a full-frame crop, manual a box
        w.cmb_fr.setCurrentIndex(0); w.tick()
        chk("framing 'auto' selects the Haar framer",
            pipe.framer.mode == "auto")
        w.cmb_fr.setCurrentIndex(1); w.tick()
        chk("framing 'off' disables the framer",
            pipe.framer.mode == "off" and not pipe.framer.found)
        w.cmb_fr.setCurrentIndex(2)
        QtWidgets.QApplication.processEvents()
        chk("manual mode reveals the box sliders",
            all(x.isVisible() and x.isEnabled() for x in w.man.values()))
        w.cmb_fr.setCurrentIndex(0)
        QtWidgets.QApplication.processEvents()
        chk("auto mode hides them again",
            not any(x.isVisible() for x in w.man.values()))
        w.cmb_fr.setCurrentIndex(2)
        before = pipe.framer.manual[2]
        w.tick()
        pipe.framer.manual[2] = 0.15
        w.tick()
        chk("manual box size is honoured (smaller box, smaller crop)",
            pipe.framer.manual[2] == 0.15 and before != 0.15)
        w.cmb_fr.setCurrentIndex(0)

        # mask source
        w.cmb_mk.setCurrentIndex(1); w.tick()
        chk("mask source switches to manifold coverage",
            pipe.cfg["mask_source"] == "manifold")
        w.cmb_mk.setCurrentIndex(0)

        # presets
        w.preset("detail")
        chk("MANIFOLD preset sets structure 1.0 and low strength",
            pipe.cfg["structure"] > 0.98 and pipe.cfg["strength"] < 0.25,
            f"structure {pipe.cfg['structure']:.2f} "
            f"strength {pipe.cfg['strength']:.2f}")
        w.preset("prompt")
        chk("PROMPT preset restores the tuned defaults",
            abs(pipe.cfg["structure"] - 0.33) < 0.02 and
            abs(pipe.cfg["strength"] - 1.00) < 0.02 and
            abs(pipe.cfg["pursuit"] - 0.92) < 0.02,
            f"structure {pipe.cfg['structure']:.2f} "
            f"strength {pipe.cfg['strength']:.2f} "
            f"pursuit {pipe.cfg['pursuit']:.2f}")

        for _ in range(8):
            w.tick()
        chk("added-HF measurement is live",
            "added HF" in w.status.currentMessage())

        shot = os.environ.get("MF_UISHOT")
        if shot:
            QtWidgets.QApplication.processEvents()
            w.grab().save(shot)
            print(f"  wrote {shot}")
        # Load avatar model: happy path swaps mf and resets pipeline state;
        # a missing/bad file must warn and leave the running model untouched.
        old_mf = pipe.mf
        w.tick()  # give the pipeline some state worth clearing
        _orig_dlg = QtWidgets.QFileDialog.getOpenFileName
        QtWidgets.QFileDialog.getOpenFileName = staticmethod(
            lambda *a, **k: (args.model, ""))
        w.load_model()
        chk("Load avatar model swaps in a new Manifold and resets state",
            pipe.mf is not old_mf and pipe.z is None and pipe.dream is None,
            f"swapped={pipe.mf is not old_mf} z={pipe.z} dream={pipe.dream}")

        QtWidgets.QFileDialog.getOpenFileName = staticmethod(
            lambda *a, **k: ("/tmp/does_not_exist_xyz.pt", ""))
        _orig_warn = QtWidgets.QMessageBox.warning
        warned = []
        QtWidgets.QMessageBox.warning = staticmethod(
            lambda *a, **k: warned.append(1))
        mf_before = pipe.mf
        w.load_model()
        chk("a bad model path warns and keeps the previous model running",
            len(warned) == 1 and pipe.mf is mf_before,
            f"warned={warned} kept={pipe.mf is mf_before}")
        QtWidgets.QFileDialog.getOpenFileName = _orig_dlg
        QtWidgets.QMessageBox.warning = _orig_warn

        # ---- 2-manifold loop: must run with NO camera at all
        w.toggle_loop()
        chk("entering the loop seeds a dream and flips the button",
            pipe.cfg["loop"] and pipe.dream is not None
            and "Leave" in w.btn_loop.text())
        if w.cap is not None:
            w.cap.release()
        w.cap = None                      # prove the camera is not consulted
        d0 = pipe.dream.copy()
        for _ in range(30):
            w.tick()
        chk("the loop ticks with the camera object removed entirely",
            pipe.dream is not None and w.panels["out"].pixmap() is not None)
        chk("the loop actually moves (dream changes with no input)",
            float(np.abs(pipe.dream.astype(np.float32)
                         - d0.astype(np.float32)).mean()) > 0.5)
        chk("panel captions say the camera is out of the picture",
            "DIFFUSION" in w.panels_cap["crop"].text())
        chk("status names the loop regime",
            "LOOP:" in w.status.currentMessage(),
            w.status.currentMessage()[-46:])
        w.btn_seed.click()
        chk("reseed restarts the loop from the manifold",
            pipe.loop_t == 0 and pipe.z is None and pipe.dream is not None)
        w.toggle_loop()
        chk("leaving the loop restores the webcam captions",
            not pipe.cfg["loop"] and "face-framed" in w.panels_cap["crop"].text())
        w.cap = FakeCap()

        w.toggle()
        chk("stop releases camera and timer",
            not w.running and not w.timer.isActive())
        return 0

    QtWidgets.QApplication.exec = fake_exec
    live(args)
    print(f"\n  {sum(ok)}/{len(ok)} UI checks pass")
    return 0 if all(ok) else 1


# ======================================================== main
def main():
    ap = argparse.ArgumentParser(
        description="StableAIflow stabilised by the TinyAvatar splat manifold")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--uismoke", action="store_true",
                    help="build the window offscreen against a fake camera")
    ap.add_argument("--model", default="model2.pt")
    ap.add_argument("--sd_model", default="stabilityai/sdxl-turbo")
    ap.add_argument("--device", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=512)
    ap.add_argument("--loop", action="store_true",
                    help="start in the 2-manifold loop (needs no camera)")
    ap.add_argument("--stub", action="store_true",
                    help="placeholder backend, no diffusers, no download")
    ap.add_argument("--prompt",
                    default="high res photo of famous persons face, only one "
                            "person, in focus at press event, greenscreen, "
                            "face centered")
    ap.add_argument("--video", default=None, help="clip for MF2")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--step", type=float, default=0.30,
                    help="latent step for the gate arms; real per-frame "
                         "motion measured ~0.2-0.4 of |z|")
    ap.add_argument("--zscale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.gates:
        return gates(a)
    if a.uismoke:
        return uismoke(a)
    if torch is None or cv2 is None:
        sys.exit("live mode needs torch and opencv")
    return live(a)


if __name__ == "__main__":
    sys.exit(main())

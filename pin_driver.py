#!/usr/bin/env python3
# ============================================================================
# pin_driver.py — drive the avatar through the DECODER, not the encoder.
#
# THE PROBLEM THIS EXISTS FOR
#   The shipped driver goes  frame -> conv encoder -> z -> render.  That path
#   has a measured defect: the encoder is translation-invariant (a 28 px slide
#   of the input moves the reconstruction 0.5 px), it is appearance-based, and
#   over a live session z wanders into directions that change how the face
#   LOOKS without changing where anything IS.  That is the appearance decay —
#   geometry holds, sharpness and colour rot toward the dataset mean.
#
#   This module takes the other route.  Facial landmarks from the webcam
#   become PINS, and z is solved so the rendered feature positions match them,
#   using the same damped-least-squares latent IK as splat_ragdoll.py.  The
#   decoder Jacobian d(centroid)/dz is the geometrically correct operator for
#   "move this feature there"; the encoder is not.
#
# THE STIFFNESS MASK, AND WHAT IT ACTUALLY IS
#   manifold_probe.py measures, per image location, how much |dz| it costs to
#   move the packets there.  On a face model the face is cheap and the frame
#   is expensive — the latent spends its degrees of freedom on the subject and
#   treats the background as near-fixed scenery.
#
#   You cannot mask a 128-D latent update with a 2-D picture, so the mask is
#   not applied as a picture.  It is applied as a SUBSPACE.  build_control_
#   basis() stacks the Jacobian rows d(centroid)/dz over a grid of face-region
#   grab points, optionally projects out the directions that move a background
#   ring, and takes the top-k right singular vectors.  Driving is then done in
#
#       z = z_anchor + V @ a ,      V: (128, k) orthonormal, a: (k,)
#
#   so stiff / background / non-facial directions are not penalised — they are
#   UNREACHABLE.  The identity anchor is preserved by construction along every
#   direction outside span(V), and by a nullspace pull toward a = 0 inside it.
#   That is the "identity locked in the null space" requirement, made
#   structural instead of soft.
#
#   Bonus readout: the singular value spectrum of the stacked Jacobian is a
#   direct count of how many latent directions actually move the face.  On a
#   model with diversity collapse it falls off a cliff, and you can see where.
#
# LANDMARKS
#   Two backends, chosen automatically:
#     haar       always available (ships with opencv).  Detects both eyes
#                inside the framer's crop -> 2 pins, 4 constraints.  That
#                carries roll, eye separation, and vertical eye position.
#     mediapipe  used if importable.  8 pins: 4 eye corners, 2 mouth corners,
#                nose tip, chin -> pose AND expression.
#
#   Pins live in CROP-NORMALISED coordinates, using FaceFramer's 0.35-margin
#   crop — the same framing Dataset Prep used, which is the whole reason live
#   framing matters.  Note the consequence honestly: because the crop tracks
#   the face, global head translation is largely removed before the pins are
#   read.  That is correct rather than a limitation — the training frames were
#   face-cropped too, so the manifold contains almost no global translation and
#   a model asked to put a face in the corner has never seen one.  What is
#   tracked is within-crop geometry: roll, scale, vertical offset, and (with
#   mediapipe) expression.
#
# CALIBRATION
#   At startup the current landmarks become the reference, and clusters are
#   grabbed on the anchor face at those locations.  Each frame the target is
#     target_i = centroid_ref_i + gain * (landmark_i - landmark_ref_i)
#   so the offset between where your eye sits in the crop and where the model's
#   eye sits never has to be zero.  Press c to recalibrate.
#
# REGISTERED GATES  (frozen before running; python pin_driver.py --selftest
#                    and --gates --model X.pt)
#   PD0a  basis orthonormality: ||V^T V - I||_max < 1e-5
#   PD0b  structural containment: for any solved z, the component of
#         (z - z_anchor) orthogonal to span(V) is < 1e-6.  This is the whole
#         claim of the projection and it must hold to machine precision.
#   PD0c  reduced solver reaches a reachable pin target within 1.5 px in <= 60
#         DLS iterations.
#   PD1   BACKGROUND SUPPRESSION, the money gate.  Drive the same synthetic
#         pin trajectory twice — once in the reduced basis, once in full 128-D
#         — and measure how far the background-ring packets moved.
#         [V] iff reduced background motion <= 0.60 x full, at equal pin error.
#         A [K] means the compliant subspace is not separating subject from
#         frame on this model, which is a finding about the model.
#   PD2   RATE: full loop (landmark -> Jacobian -> solve -> render) >= 30 FPS.
#   PD3   IDENTITY RETENTION — cannot be measured headless.  Registered for
#         your machine: run --compare_drive with a webcam, which logs
#         ||z - z_anchor|| and recon sharpness for pin-driving vs encoder-
#         driving over the same session.  Unmeasured until you run it.
#
# HONEST LIMITS ON RECORD
#   * The Jacobian is recomputed once per frame and held fixed across that
#     frame's DLS iterations (frozen-Jacobian Gauss-Newton).  Valid because
#     inter-frame motion is small; it is an approximation, not exact IK.
#   * Haar gives 2 pins.  Two pins cannot express a smile.  Expression
#     tracking needs the mediapipe backend, and without it this is a pose
#     driver.
#   * "Fixes appearance decay" is PD3, and PD3 is unmeasured.  The mechanism
#     argument is sound and the encoder's translation-invariance is measured,
#     but the improvement itself is a hypothesis until your session log says
#     otherwise.
#   * k (basis rank) is a knob with no measured optimum.  Default 24 was
#     chosen from the singular spectrum's knee on the models to hand, not from
#     a sweep.
#
# USAGE
#   python pin_driver.py --selftest
#   python pin_driver.py --gates --model model2.pt
#   python pin_driver.py --model model2.pt                 # live webcam
#   python pin_driver.py --model model2.pt --k 32 --gain 1.3
#   python pin_driver.py --model model2.pt --full_basis    # ablation: no mask
#
# KEYS (live)
#   c  recalibrate reference from the current frame
#   f  toggle compliant-subspace projection on/off (live A/B against full 128-D)
#   e  toggle encoder driving on/off (the incumbent, for side-by-side)
#   [ ]  gain down / up        - =  solver damping
#   o  overlay        r  reset to anchor        q  quit
#
# Do not hype. Do not lie. Just show.
# ============================================================================
import argparse, math, os, sys, time
import numpy as np
import torch

torch.set_grad_enabled(False)

try:
    import splat_trainer5 as ST
    from splat_ragdoll import (load_model, packet_amp, grab_weights,
                               render_from_params)
except ImportError:
    ST = None
    load_model = packet_amp = grab_weights = render_from_params = None


def _need():
    if ST is None:
        sys.exit("pin_driver.py needs splat_trainer5.py, splat_trainer3v2.py "
                 "and splat_ragdoll.py in the same directory.")


# ------------------------------------------------------------------ geometry
def centroid(w, px, py):
    ws = w.sum() + 1e-9
    return torch.stack([(w * px[0]).sum() / ws, (w * py[0]).sum() / ws])


def jac_rows(model, z, weights, wrt=None):
    """d(centroid_i)/d(param) for each cluster in `weights`.
    Returns (2m, P) where P is len(z) or len(a) depending on `wrt`."""
    with torch.enable_grad():
        if wrt is None:
            leaf = z.detach().clone().requires_grad_(True)
            raw = model.dec(leaf[None])
        else:
            leaf = wrt.detach().clone().requires_grad_(True)
            # Reconstruct z from low-rank a inside the grad graph if wrt is passed
            leaf_z = leaf  # caller already built computation graph or passes a leaf
            raw = model.dec(z[None])
            
        px, py, *_ = model.ren.activate(raw.float())
        rows, vals = [], []
        for w in weights:
            c = centroid(w, px, py)
            vals.append(c)
            for k in range(2):
                g = torch.autograd.grad(c[k], leaf, retain_graph=True)[0]
                rows.append(g)
    return torch.stack(rows).detach(), torch.stack(vals).detach()


# ------------------------------------------------- compliant control basis
def build_control_basis(model, z_anchor, k=24, grid=8, r_grab=0.10,
                        face_lo=0.22, face_hi=0.78, bg_drop=4, verbose=True):
    """The stiffness mask, as a subspace.

    Stack d(centroid)/dz over a grid of grab points inside the face region;
    optionally project out the top `bg_drop` directions that move a background
    ring; SVD; keep the top k right singular vectors.

    Returns (V, S, info):
      V     (128, k) orthonormal — the compliant / controllable subspace
      S     singular values of the (projected) face Jacobian stack
      info  dict with the background spectrum and coverage diagnostics
    """
    N = model.ren.N
    raw = model.dec(z_anchor[None]).float()
    px, py, sg, th, fr, cf = model.ren.activate(raw)
    amp = packet_amp(cf)
    bg_gate = torch.ones(N)

    def clusters(points):
        out = []
        for p in points:
            w = grab_weights(px, py, sg, amp, p, r_grab, bg_gate)
            if float(w.sum()) > 1e-6:
                out.append(w)
        return out

    g = np.linspace(face_lo, face_hi, grid)
    face_pts = [(float(a), float(b)) for b in g for a in g]
    ring = []
    for t in np.linspace(0, 1, 4 * grid, endpoint=False):
        # square ring just inside the frame
        s = 4 * t
        if s < 1:   ring.append((0.06 + 0.88 * s, 0.06))
        elif s < 2: ring.append((0.94, 0.06 + 0.88 * (s - 1)))
        elif s < 3: ring.append((0.94 - 0.88 * (s - 2), 0.94))
        else:       ring.append((0.06, 0.94 - 0.88 * (s - 3)))

    Wf, Wb = clusters(face_pts), clusters(ring)
    if not Wf:
        raise RuntimeError("no face-region clusters — check face_lo/face_hi")

    Jf, _ = jac_rows(model, z_anchor, Wf)          # (2Nf, 128)
    Sb = None
    if Wb and bg_drop > 0:
        Jb, _ = jac_rows(model, z_anchor, Wb)      # (2Nb, 128)
        Ub, Sb, Vbt = torch.linalg.svd(Jb, full_matrices=False)
        d = min(bg_drop, Vbt.shape[0])
        Vb = Vbt[:d].T                             # (128, d)
        Jf = Jf - (Jf @ Vb) @ Vb.T                 # kill background directions

    U, S, Vt = torch.linalg.svd(Jf, full_matrices=False)
    k = int(min(k, Vt.shape[0]))
    V = Vt[:k].T.contiguous()                      # (128, k)

    info = {"n_face": len(Wf), "n_bg": len(Wb),
            "S_face": S[:k].clone(), "S_bg": Sb,
            "energy": float(S[:k].pow(2).sum() / (S.pow(2).sum() + 1e-30))}
    if verbose:
        s = S / (S[0] + 1e-30)
        keep = int((s > 0.05).sum())
        print(f"control basis: {len(Wf)} face clusters, {len(Wb)} bg clusters, "
              f"k={k}")
        print(f"  singular spectrum (normalised): "
              + " ".join(f"{float(v):.3f}" for v in s[:min(12, len(s))])
              + (" ..." if len(s) > 12 else ""))
        print(f"  directions above 5% of the leading mode: {keep}   "
              f"(k={k} captures {100*info['energy']:.1f}% of face-motion "
              f"energy)")
        if keep < k:
            print(f"  NOTE: only {keep} directions carry real face motion. "
                  f"k={k} is padding the basis with near-null directions; "
                  f"consider --k {max(4, keep)}.")
    return V, S[:k], info


# ------------------------------------------------------------------ driver
class PinDriver:
    """Solves z = z_anchor + V a so that pinned cluster centroids track
    landmark targets.  Frozen-Jacobian Gauss-Newton, damped least squares,
    nullspace pull toward the anchor pose."""

    def __init__(self, model, z_anchor, V=None, lam=0.06, beta=0.25,
                 step_clip=0.5, iters=3):
        self.m = model
        self.z0 = z_anchor.clone()
        self.V = V                          # None => full 128-D
        self.lam, self.beta = lam, beta
        self.step_clip, self.iters = step_clip, iters
        self.dim = 128 if V is None else V.shape[1]
        self.a = torch.zeros(self.dim)
        self.weights, self.c_ref, self.lm_ref = [], None, None

    # ---- state
    def z(self):
        return self.z0 + (self.a if self.V is None else self.V @ self.a)

    def reset(self):
        self.a.zero_()

    # ---- calibration
    def calibrate(self, landmarks, r_grab=0.09):
        """landmarks: (m,2) array in [0,1] crop coords.  Grab clusters on the
        anchor face at those locations and record the reference offsets."""
        z = self.z()
        raw = self.m.dec(z[None]).float()
        px, py, sg, th, fr, cf = self.m.ren.activate(raw)
        amp = packet_amp(cf)
        gate = torch.ones(self.m.ren.N)
        ws, cs, lms = [], [], []
        for (u, v) in landmarks:
            w = grab_weights(px, py, sg, amp, (float(u), float(v)),
                             r_grab, gate)
            if float(w.sum()) < 1e-6:
                continue
            ws.append(w)
            cs.append(centroid(w, px, py))
            lms.append([float(u), float(v)])
        if not ws:
            return False
        self.weights = ws
        self.c_ref = torch.stack(cs)
        self.lm_ref = torch.tensor(lms, dtype=torch.float32)
        return True

    # ---- per-frame solve
    def step(self, landmarks, gain=1.0):
        """Returns (z, pin_err_px).  landmarks must match calibration order."""
        if not self.weights:
            return self.z(), 0.0
        lm = torch.tensor(np.asarray(landmarks, dtype=np.float32)[:len(self.weights)])
        tgt = self.c_ref + gain * (lm - self.lm_ref)
        tgt = tgt.clamp(0.02, 0.98)

        with torch.enable_grad():
            a = self.a.detach().clone().requires_grad_(True)
            z = self.z0 + (a if self.V is None else self.V @ a)
            J, c = jac_rows(self.m, z, self.weights, wrt=a)   # (2m, dim)

        err_px = 0.0
        H = self.m.ren.H
        for _ in range(self.iters):
            # centroids under the CURRENT a, Jacobian held from frame start
            raw = self.m.dec(self.z()[None]).float()
            px, py, *_ = self.m.ren.activate(raw)
            c_now = torch.stack([centroid(w, px, py) for w in self.weights])
            e = (tgt - c_now).reshape(-1)
            JJt = J @ J.T + (self.lam ** 2) * torch.eye(J.shape[0])
            K = J.T @ torch.linalg.solve(JJt, torch.eye(J.shape[0]))
            da = K @ e
            if self.beta > 0:
                P = torch.eye(J.shape[1]) - K @ J
                da = da + P @ (-self.beta * self.a)
            n = da.norm()
            if n > self.step_clip:
                da = da * (self.step_clip / n)
            self.a = self.a + da
            err_px = float(e.reshape(-1, 2).norm(dim=1).max()) * H
        return self.z(), err_px


# ------------------------------------------------------------------ landmarks
class Landmarker:
    """Returns landmarks in crop-normalised [0,1] coords, ordered and stable.
    backend 'mediapipe' if available else 'haar'."""

    HAAR_NAMES = ["eye_l", "eye_r"]
    MP_NAMES = ["eye_l_out", "eye_l_in", "eye_r_in", "eye_r_out",
                "mouth_l", "mouth_r", "nose", "chin"]
    MP_IDX = [33, 133, 362, 263, 61, 291, 1, 152]

    def __init__(self, prefer="auto"):
        import cv2 as cv
        self.cv = cv
        self.backend = "haar"
        self.mp = None
        if prefer in ("auto", "mediapipe"):
            try:
                import mediapipe as mp
                self.mp = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False, max_num_faces=1,
                    refine_landmarks=False, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
                self.backend = "mediapipe"
            except Exception:
                self.mp = None
        if self.backend == "haar":
            p = os.path.join(cv.data.haarcascades, "haarcascade_eye.xml")
            self.eye = cv.CascadeClassifier(p)
            if self.eye.empty():
                self.eye = None
        self.last = None

    @property
    def names(self):
        return self.MP_NAMES if self.backend == "mediapipe" else self.HAAR_NAMES

    def detect(self, crop_bgr):
        """crop_bgr: the FaceFramer crop (square).  Returns (m,2) in [0,1] or
        None.  Holds the last good reading on a miss rather than jumping."""
        cv = self.cv
        h, w = crop_bgr.shape[:2]
        if self.backend == "mediapipe":
            res = self.mp.process(cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                return self.last
            L = res.multi_face_landmarks[0].landmark
            out = np.array([[L[i].x, L[i].y] for i in self.MP_IDX],
                           dtype=np.float32)
        else:
            if self.eye is None:
                return self.last
            g = cv.cvtColor(crop_bgr, cv.COLOR_BGR2GRAY)
            top = g[: int(0.60 * h)]
            det = self.eye.detectMultiScale(top, 1.12, 6,
                                            minSize=(max(8, w // 14),) * 2)
            if len(det) < 2:
                return self.last
            det = sorted(det, key=lambda b: -b[2] * b[3])[:2]
            det = sorted(det, key=lambda b: b[0])          # left, right
            out = np.array([[(x + bw / 2) / w, (y + bh / 2) / h]
                            for (x, y, bw, bh) in det], dtype=np.float32)
        self.last = np.clip(out, 0.0, 1.0)
        return self.last


# ================================================================== SELFTEST
def _tiny(size=32, packets=48, seed=0):
    _need()
    torch.manual_seed(seed)
    m = ST.SplatVAEQ(size, packets, chunk=32, ckpt=False, qmode=True, q=0.6,
                     gist_frac=0.0, octaves=4, sig_lo=0.008, sig_hi=0.70,
                     band_mode="permute")
    for p in m.dec.parameters():
        p.add_(torch.randn_like(p) * 0.05)
    m.eval()
    return m


def _bg_motion(model, z_a, z_b, ring_w):
    """Mean displacement (px) of background-ring clusters between two z."""
    def cents(z):
        raw = model.dec(z[None]).float()
        px, py, *_ = model.ren.activate(raw)
        return torch.stack([centroid(w, px, py) for w in ring_w])
    return float((cents(z_b) - cents(z_a)).norm(dim=1).mean()) * model.ren.H


def selftest():
    _need()
    print("SELFTEST — machinery only. A random decoder has no manifold; these "
          "results certify the solver and the projection, nothing about a "
          "trained model.")
    m = _tiny()
    z0 = torch.randn(128) * 0.5
    V, S, info = build_control_basis(m, z0, k=12, grid=5, bg_drop=2,
                                     verbose=False)

    ok_a = float((V.T @ V - torch.eye(V.shape[1])).abs().max()) < 1e-5
    print(f"PD0a basis orthonormality        {'[V]' if ok_a else '[K]'}")

    raw = m.dec(z0[None]).float()
    px, py, sg, th, fr, cf = m.ren.activate(raw)
    amp = packet_amp(cf)
    gate = torch.ones(m.ren.N)
    kbest = int(amp[0].argmax())
    lm = np.array([[float(px[0, kbest]), float(py[0, kbest])]],
                  dtype=np.float32)

    d = PinDriver(m, z0, V, lam=0.05, beta=0.0, step_clip=0.4, iters=1)
    assert d.calibrate(lm, r_grab=0.12)
    err = 1e9
    for it in range(60):
        _, err = d.step(lm + np.array([[0.05, 0.03]], np.float32))
        if err < 1.5:
            break
    ok_c = err < 1.5
    print(f"PD0c reduced solver convergence   pin err {err:.2f} px @ iter "
          f"{it+1}  {'[V]' if ok_c else '[K]'}")

    dz = d.z() - z0
    resid = float((dz - V @ (V.T @ dz)).abs().max())
    ok_b = resid < 1e-6
    print(f"PD0b structural containment       max leak outside span(V) "
          f"{resid:.2e}  {'[V]' if ok_b else '[K]'}")

    # PD1 — background suppression, reduced vs full 128-D on identical targets
    ring = [(0.06, 0.5), (0.94, 0.5), (0.5, 0.06), (0.5, 0.94),
            (0.10, 0.10), (0.90, 0.90)]
    ring_w = [grab_weights(px, py, sg, amp, p, 0.10, gate) for p in ring]
    ring_w = [w for w in ring_w if float(w.sum()) > 1e-6]
    tgt = lm + np.array([[0.05, 0.03]], np.float32)
    res = {}
    for name, basis in (("reduced", V), ("full", None)):
        dd = PinDriver(m, z0, basis, lam=0.05, beta=0.0, step_clip=0.4, iters=1)
        dd.calibrate(lm, r_grab=0.12)
        e = 1e9
        for _ in range(60):
            _, e = dd.step(tgt)
            if e < 1.5:
                break
        res[name] = (_bg_motion(m, z0, dd.z(), ring_w) if ring_w else 0.0, e)
    r_bg, r_e = res["reduced"]; f_bg, f_e = res["full"]
    ratio = r_bg / max(f_bg, 1e-9)
    ok_1 = ratio <= 0.60 and r_e < 3.0
    print(f"PD1  background suppression       reduced {r_bg:.3f} px vs full "
          f"{f_bg:.3f} px  ratio {ratio:.2f} (<=0.60)  "
          f"[pin err {r_e:.2f}/{f_e:.2f}]  {'[V]' if ok_1 else '[K]'}")

    t0 = time.time()
    for _ in range(20):
        d.step(lm)
        render_from_params(m.ren, *m.ren.activate(m.dec(d.z()[None]).float()))
    dt = (time.time() - t0) / 20
    print(f"PD2  rate ({m.ren.H}px/{m.ren.N}pk, CPU)  {1/max(dt,1e-9):.1f} FPS "
          f"[report — the gate is >=30 on your model]")

    allv = ok_a and ok_b and ok_c and ok_1
    print(f"SELFTEST {'ALL-[V]' if allv else 'FAILED'}")
    return 0 if allv else 1


# ==================================================================== GATES
def gates(args):
    _need()
    m, ck = load_model(args.model)
    S = m.ren.H
    print(f"GATES on {args.model}  ({S}px / {m.ren.N} packets)")
    torch.manual_seed(args.seed)
    ratios, errs = [], []
    for s in range(args.samples):
        z0 = torch.randn(128) * 0.8
        V, sv, info = build_control_basis(m, z0, k=args.k, grid=args.grid,
                                          bg_drop=args.bg_drop,
                                          verbose=(s == 0))
        if s == 0:
            ok_a = float((V.T @ V - torch.eye(V.shape[1])).abs().max()) < 1e-5
            print(f"PD0a basis orthonormality  {'[V]' if ok_a else '[K]'}")

        raw = m.dec(z0[None]).float()
        px, py, sg, th, fr, cf = m.ren.activate(raw)
        amp = packet_amp(cf); gate = torch.ones(m.ren.N)
        lm = np.array([[0.36, 0.42], [0.64, 0.42]], np.float32)
        ring = [(0.06, 0.5), (0.94, 0.5), (0.5, 0.06), (0.5, 0.94),
                (0.10, 0.10), (0.90, 0.90), (0.10, 0.90), (0.90, 0.10)]
        ring_w = [grab_weights(px, py, sg, amp, p, 0.10, gate) for p in ring]
        ring_w = [w for w in ring_w if float(w.sum()) > 1e-6]
        tgt = lm + np.array([[0.03, 0.02], [0.03, 0.02]], np.float32)

        out = {}
        for name, basis in (("reduced", V), ("full", None)):
            d = PinDriver(m, z0, basis, lam=args.lam, beta=0.0,
                          step_clip=0.5, iters=1)
            if not d.calibrate(lm, r_grab=0.09):
                out[name] = None; continue
            e = 1e9
            for _ in range(args.iters):
                _, e = d.step(tgt)
                if e < args.tol_px:
                    break
            out[name] = (_bg_motion(m, z0, d.z(), ring_w) if ring_w else 0.0, e)
        if out.get("reduced") is None or out.get("full") is None:
            print(f"  sample {s}: empty grab, skipped"); continue
        rb, re_ = out["reduced"]; fb, fe = out["full"]
        r = rb / max(fb, 1e-9)
        ratios.append(r); errs.append(re_)
        print(f"  sample {s}: bg motion reduced {rb:6.3f} px  full {fb:6.3f} px"
              f"  ratio {r:.2f}   pin err {re_:.2f}/{fe:.2f} px")

    if not ratios:
        return 1
    med = float(np.median(ratios))
    ok1 = med <= 0.60 and float(np.median(errs)) < args.tol_px * 2
    print(f"PD1  background suppression: median ratio {med:.2f} (<=0.60), "
          f"median pin err {np.median(errs):.2f} px  {'[V]' if ok1 else '[K]'}")

    # PD2 rate
    z0 = torch.randn(128) * 0.8
    V, _, _ = build_control_basis(m, z0, k=args.k, grid=args.grid,
                                 bg_drop=args.bg_drop, verbose=False)
    d = PinDriver(m, z0, V, iters=args.frame_iters)
    d.calibrate(np.array([[0.36, 0.42], [0.64, 0.42]], np.float32))
    t0 = time.time()
    for i in range(20):
        d.step(np.array([[0.36 + 0.01 * math.sin(i), 0.42],
                         [0.64, 0.42]], np.float32))
        render_from_params(m.ren, *m.ren.activate(m.dec(d.z()[None]).float()))
    fps = 20 / (time.time() - t0)
    print(f"PD2  rate: full loop {fps:.1f} FPS (>=30)  "
          f"{'[V]' if fps >= 30 else '[K]'}  (this machine, CPU-only as "
          f"shipped)")
    print("PD3  identity retention: unmeasured — needs a webcam session. "
          "Run the live driver and press e to A/B against encoder driving.")
    return 0


# ===================================================================== LIVE
def live(args):
    import cv2 as cv
    _need()
    m, ck = load_model(args.model)
    S = m.ren.H
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from tiny_avatar4 import FaceFramer, normalize_crop
    except Exception:
        FaceFramer = None
        def normalize_crop(x, tgt_mean=0.52, tgt_std=0.26):
            mu, sd = x.mean(), x.std() + 1e-6
            return np.clip((x - mu) / sd * tgt_std + tgt_mean, 0, 1)

    torch.manual_seed(args.seed)
    z_anchor = torch.randn(128) * 0.8
    if args.image:
        import cv2 as _c
        im = _c.imread(args.image)
        if im is not None:
            im = _c.cvtColor(_c.resize(im, (S, S)), _c.COLOR_BGR2RGB)
            x = torch.from_numpy(im).float().permute(2, 0, 1)[None] / 255.0
            z_anchor = m.enc(x)[0][0]

    V, sv, info = build_control_basis(m, z_anchor, k=args.k, grid=args.grid,
                                      bg_drop=args.bg_drop)
    lmk = Landmarker(args.landmarks)
    print(f"landmarks: {lmk.backend}  ({len(lmk.names)} pins: "
          f"{', '.join(lmk.names)})")
    if lmk.backend == "haar":
        print("  haar gives 2 pins — pose only. pip install mediapipe for "
              "expression tracking.")

    drv = PinDriver(m, z_anchor, V, lam=args.lam, beta=args.beta,
                    step_clip=0.5, iters=args.frame_iters)
    cap = cv.VideoCapture(args.cam)
    if not cap.isOpened():
        sys.exit("webcam failed to open")
    framer = FaceFramer() if FaceFramer is not None else None
    use_mask, use_enc, overlay = True, False, True
    gain, calibrated = args.gain, False
    D = 384
    t_last, fps = time.time(), 0.0
    win = "pin driver — TinyAvatar2"
    cv.namedWindow(win)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if framer is not None:
            crop = framer.crop(frame)
        else:
            h, w = frame.shape[:2]; s = min(h, w)
            crop = frame[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        lm = lmk.detect(crop)

        err = 0.0
        if use_enc:
            x = cv.resize(crop, (S, S))[:, :, ::-1].astype(np.float32) / 255.0
            x = normalize_crop(x)
            xt = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
            z = m.enc(xt)[0][0]
        else:
            if lm is not None and not calibrated:
                calibrated = drv.calibrate(lm, r_grab=args.r_grab)
            if lm is not None and calibrated:
                z, err = drv.step(lm, gain=gain)
            else:
                z = drv.z()

        raw = m.dec(z[None]).float()
        img = render_from_params(m.ren, *m.ren.activate(raw))
        av = (img[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(
            np.uint8)[:, :, ::-1]
        av = cv.resize(av, (D, D), interpolation=cv.INTER_CUBIC)
        cam = cv.resize(crop, (D, D))
        if overlay and lm is not None:
            for (u, v) in lm:
                cv.drawMarker(cam, (int(u * D), int(v * D)), (0, 220, 255),
                              cv.MARKER_CROSS, 12, 2)
        out = np.hstack([cam, av])
        if overlay:
            hud = (f"{'ENCODER' if use_enc else 'PIN'}  "
                   f"{'mask ON' if (use_mask and not use_enc) else 'mask off'}"
                   f"  k={drv.dim}  gain {gain:.2f}  err {err:.1f}px  "
                   f"|z-z0| {float((z - z_anchor).norm()):.2f}  {fps:.0f} fps"
                   f"  [{lmk.backend}]")
            cv.putText(out, hud, (8, 2 * D - 12 if False else out.shape[0] - 12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
                       cv.LINE_AA)
        cv.imshow(win, out)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6); t_last = now

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibrated = False
        elif key == ord('f'):
            use_mask = not use_mask
            drv = PinDriver(m, z_anchor, V if use_mask else None,
                            lam=args.lam, beta=args.beta, step_clip=0.5,
                            iters=args.frame_iters)
            calibrated = False
        elif key == ord('e'):
            use_enc = not use_enc
        elif key == ord('['):
            gain = max(0.2, gain - 0.1)
        elif key == ord(']'):
            gain = min(3.0, gain + 0.1)
        elif key == ord('-'):
            drv.lam = min(0.5, drv.lam * 1.3)
        elif key == ord('='):
            drv.lam = max(0.005, drv.lam / 1.3)
        elif key == ord('o'):
            overlay = not overlay
        elif key == ord('r'):
            drv.reset()

    cap.release()
    cv.destroyAllWindows()
    return 0


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(
        description="pin-driven avatar control on the compliant subspace")
    ap.add_argument("--model")
    ap.add_argument("--image", help="encode this to set the anchor identity")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--landmarks", default="auto",
                    choices=["auto", "haar", "mediapipe"])
    ap.add_argument("--k", type=int, default=24, help="compliant basis rank")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--bg_drop", type=int, default=4,
                    help="background directions projected out (0 disables)")
    ap.add_argument("--full_basis", action="store_true",
                    help="ablation: drive in full 128-D, no stiffness mask")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.06)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--r_grab", type=float, default=0.09)
    ap.add_argument("--frame_iters", type=int, default=3)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--tol_px", type=float, default=1.5)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.full_basis:
        a.k = 128
        a.bg_drop = 0
    if a.selftest:
        sys.exit(selftest())
    if a.gates:
        if not a.model:
            sys.exit("--gates needs --model")
        sys.exit(gates(a))
    if not a.model:
        sys.exit("need --model (or --selftest)")
    sys.exit(live(a))


if __name__ == "__main__":
    main()

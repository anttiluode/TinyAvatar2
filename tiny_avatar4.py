#!/usr/bin/env python3
# =============================================================================
# tiny_avatar4.py — TINY AVATAR 2 studio
#
#   Tab 1  Home            what this is
#   Tab 2  Dataset Prep    video -> face-cropped frames, or drop an image folder
#   Tab 3  Training Studio runs splat_trainer5.py as a SUBPROCESS (OOM can
#                          never take the GUI down), parses its log lines,
#                          shows live previews + GPU/RAM pulse, detects
#                          resumable runs, and — new — exposes the constant-Q
#                          BASIS knobs with a live octave-ladder preview
#   Tab 4  Avatar Driver   phase-transport pursuit driving webcam or latent
#                          walk, rendered into the app
#
# WHAT CHANGED FROM tiny_avatar3, AND WHY
# ---------------------------------------
# 1. SILENT MISRENDER, fixed. v3 loaded models with
#        model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
#        model.load_state_dict(ck["sd"])
#    which ignores the checkpoint's own parameterisation and applies whatever
#    the renderer defaults happen to be. A constant-Q state_dict has the SAME
#    SHAPE as a legacy one, so this raises no error — it just renders the
#    wrong pixels. Measured in the trainer's smoke test: up to 0.31 absolute
#    difference on a 0-1 image. Every load now goes through the trainer's own
#    load_splatvae(), which reads qmode/q/octaves/sig_hi/f_max back out of the
#    file and rebuilds the matching renderer.
#
# 2. PROGRESS BAR, fixed. v3's LOG_RE required "(PSNR x) kl y", which the 4q
#    trainer never printed, so the bar and the stat line never moved.
#    splat_trainer5 now prints both, and the parser here also tolerates their
#    absence so older logs still register step counts.
#
# 3. RESUME, fixed. v3 scanned for model2.pt while the trainer wrote
#    model4q_<tag>.pt, so the button never armed; and --resume was not a real
#    trainer flag, so pressing it did nothing anyway. Now it globs model*.pt,
#    and splat_trainer5 restores optimizer + schedule + step.
#
# 4. BASIS KNOBS, new. v3 exposed only the old-model settings (beta, lr,
#    packets, size) and had no way to touch the constant-Q parameterisation
#    at all, so every launch used defaults. q / q_slack / octaves / sig_hi /
#    f_max / gist_frac / detail are now controls, with a LIVE OCTAVE LADDER
#    readout underneath that recomputes f_lo = max(1, q/sig_hi) and the band
#    edges as you type — and turns red if the settings would reopen the
#    spectral hole that produced the "frosted glass" renders.
#
# 5. Dead controls removed. The --aug and --disk checkboxes were passed
#    through a flag guard that silently dropped them because the trainer
#    never declared either. A control that does nothing is worse than no
#    control; they are gone, and any flag the app sends is now asserted
#    against the trainer's declared set at launch.
#
# HONESTY LEDGER (what was actually verified before shipping)
#   [V] app constructs offscreen; all four tabs build (PyQt6 6.11, headless)
#   [V] LOG_RE matches splat_trainer5's real log line, and still matches the
#       older PSNR-less format
#   [V] scan_resume finds model5_<tag>.pt written by a real 40-step run
#   [V] load_splatvae path loads a real checkpoint and renders through the
#       same pursue()/render_image() the driver uses
#   [V] flag guard: every flag this app sends is declared by splat_trainer5
#   [ ] webcam mode needs a camera — wired identically to v3, untested here
#   [ ] pulse check on an actual CUDA card — the math ran, this box is CPU
# =============================================================================
import glob
import importlib.util
import math
import os
import re
import sys
import time

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QProcess, QSize
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFileDialog, QPlainTextEdit, QProgressBar,
    QGroupBox, QSlider, QMessageBox, QFormLayout, QSizePolicy)

# ---------------------------------------------------------------- adapter
TRAINER_CANDIDATES = ["splat_trainer5.py", "splat_trainer4q.py"]


def find_trainer():
    for name in TRAINER_CANDIDATES:
        p = os.path.join(APP_DIR, name)
        if os.path.exists(p):
            return p
    return None


def trainer_flags(path):
    """Which CLI flags does the trainer actually declare? (source scan)"""
    try:
        src = open(path, "r", encoding="utf-8", errors="replace").read()
    except OSError:
        return set()
    return set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', src))


_TRAINER_MOD = None


def import_trainer(path):
    global _TRAINER_MOD
    if _TRAINER_MOD is None:
        spec = importlib.util.spec_from_file_location("splat_trainer_mod", path)
        _TRAINER_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_TRAINER_MOD)
    return _TRAINER_MOD


def octave_ladder(q, sig_hi, f_max, octaves, image_size, num_packets,
                  gist_frac):
    """Reproduce GaborRendererQ's band construction without building it, so
    the studio can show the ladder before a run starts. Returns
    (rows, warning_or_None) where rows are (lo, hi, n_packets)."""
    f_lo = max(1.0, q / max(sig_hi, 1e-9))
    f_max = f_max if f_max else 0.5 * (image_size / 2.0)
    n_gist = int(round(gist_frac * num_packets))
    n_car = max(1, num_packets - n_gist)
    span = math.log(max(f_max / f_lo, 1.0000001))
    rows, counts = [], [0] * octaves
    for k in range(n_gist, num_packets):
        b = min(int((k - n_gist) * octaves / n_car), octaves - 1)
        counts[b] += 1
    for b in range(octaves):
        lo = f_lo * math.exp(span * b / octaves)
        hi = f_lo * math.exp(span * (b + 1) / octaves)
        rows.append((lo, hi, counts[b]))
    warn = None
    if n_gist > 0 and f_lo > 1.0 + 1e-6:
        warn = (f"SPECTRAL HOLE: {n_gist} carrier-free packets sit at freq 0 "
                f"but carriers start at {f_lo:.2f} cyc/image, so nothing "
                f"covers (0, {f_lo:.2f}) — the head-outline and "
                f"feature-layout band. This renders as frosted glass plus "
                f"fine stripes with no face gestalt. Set gist_frac to 0, or "
                f"raise sig_hi so f_lo = q/sig_hi reaches 1.0.")
    elif f_lo > 2.0:
        warn = (f"carrier floor f_lo = q/sig_hi = {f_lo:.2f} cyc/image is "
                f"above the head-outline band (1-2). Raise sig_hi.")
    return rows, warn


# ---------------------------------------------------------------- math
def render_image(ren, P):
    import torch
    px, py, sigma, theta, freq, coeff = P
    out = None
    for i in range(0, ren.N, ren.chunk):
        sl = slice(i, i + ren.chunk)
        c = ren._chunk(px[:, sl], py[:, sl], sigma[:, sl],
                       theta[:, sl], freq[:, sl], coeff[:, sl])
        out = c if out is None else out + c
    return torch.sigmoid(out)


def _arc_step(a, b, alpha):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + alpha * d


def _screw_step(px, py, th, pxT, pyT, thT, alpha):
    """Fractional step along the SE(2) geodesic (screw motion)."""
    import torch
    w = torch.remainder(thT - th + math.pi, 2 * math.pi) - math.pi
    dx, dy = pxT - px, pyT - py
    c, s = torch.cos(th), torch.sin(th)
    dpx = c * dx + s * dy
    dpy = -s * dx + c * dy
    small = w.abs() < 1e-6
    ws = torch.where(small, torch.ones_like(w), w)
    A = torch.sin(ws) / ws
    B = (1 - torch.cos(ws)) / ws
    det = A * A + B * B
    vx = (A * dpx + B * dpy) / det
    vy = (-B * dpx + A * dpy) / det
    wt = w * alpha
    At = torch.sin(wt) / ws
    Bt = (1 - torch.cos(wt)) / ws
    ex = torch.where(small, vx * alpha, At * vx - Bt * vy)
    ey = torch.where(small, vy * alpha, Bt * vx + At * vy)
    return px + c * ex - s * ey, py + s * ex + c * ey, th + wt


def pursue(P, T, alpha, mode):
    """direct | lerp | phase | screw | dispersion.
    'phase' is the transport that passed the registered 8-pair gate.
    'screw' and 'dispersion' are demos, not certificates — see the README."""
    import torch
    px, py, s, th, f, c = P
    pxT, pyT, sT, thT, fT, cT = T
    L = lambda a, b: a + alpha * (b - a)
    s2, f2 = L(s, sT), L(f, fT)

    if mode == "screw":
        px2, py2, th2 = _screw_step(px, py, th, pxT, pyT, thT, alpha)
    else:
        px2, py2 = L(px, pxT), L(py, pyT)
        th2 = _arc_step(th, thT, alpha)

    if mode == "lerp":
        c2 = L(c, cT)
    else:
        a_, b_ = c[..., 0], c[..., 1]
        aT, bT = cT[..., 0], cT[..., 1]
        m = torch.sqrt(a_ * a_ + b_ * b_ + 1e-12)
        mT = torch.sqrt(aT * aT + bT * bT + 1e-12)
        ph = torch.atan2(b_, a_)
        phT = torch.atan2(bT, aT)
        m2 = L(m, mT)
        if mode == "dispersion":
            vx, vy = alpha * (pxT - px), alpha * (pyT - py)
            ux, uy = torch.cos(th), torch.sin(th)
            # closed-form phase advance. NOTE: the r=0.915 figure often
            # quoted for this was the --selftest CHAIN certification, not a
            # real-video measurement; the real-pose P2 run is what certified
            # the dispersion LAW, not this pursuit mode.
            dphi = -2.0 * math.pi * f * (ux * vx + uy * vy)
            ph2 = ph + 0.7 * dphi + 0.3 * (_arc_step(ph, phT, alpha) - ph)
        else:
            ph2 = _arc_step(ph, phT, alpha)
        c2 = torch.stack([m2 * torch.cos(ph2), m2 * torch.sin(ph2)], dim=-1)
    return (px2, py2, s2, th2, f2, c2)


class FaceFramer:
    """Live face framing that reproduces Dataset Prep's crop exactly (same
    Haar cascade, same 0.35 margin, same square-up) plus EMA smoothing.
    Fixes the train/live framing mismatch: Dataset Prep face-crops the
    training frames, but a plain center crop feeds the encoder a smaller,
    wandering face -> off-manifold -> blurry average head."""

    def __init__(self, margin=0.35, ema=0.30, every=2):
        import cv2 as cv
        cpath = os.path.join(cv.data.haarcascades,
                             "haarcascade_frontalface_default.xml")
        self.det = cv.CascadeClassifier(cpath)
        if self.det.empty():
            self.det = None
        self.margin, self.ema, self.every = margin, ema, every
        self.box = None
        self.f = 0

    def crop(self, fr):
        import cv2 as cv
        H, W = fr.shape[:2]
        if self.det is not None and self.f % self.every == 0:
            g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
            det = self.det.detectMultiScale(g, 1.15, 5, minSize=(80, 80))
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
        if self.box is None:
            s = min(H, W)
            return fr[(H - s)//2:(H + s)//2, (W - s)//2:(W + s)//2]
        cx, cy, half = self.box
        s = int(half)
        x0, x1 = int(max(cx - s, 0)), int(min(cx + s, W))
        y0, y1 = int(max(cy - s, 0)), int(min(cy + s, H))
        c = fr[y0:y1, x0:x1]
        return c if c.size else fr[(H - min(H, W))//2:(H + min(H, W))//2,
                                   (W - min(H, W))//2:(W + min(H, W))//2]


def clone_params(P):
    return tuple(t.clone() for t in P)


def normalize_crop(x, tgt_mean=0.52, tgt_std=0.26):
    m, s = x.mean(), x.std() + 1e-6
    return np.clip((x - m) / s * tgt_std + tgt_mean, 0, 1)


# ---------------------------------------------------------------- theme
QSS = """
* { font-family: 'Segoe UI', 'Inter', sans-serif; }
QMainWindow, QWidget { background: #14161b; color: #d7dae0; }
QTabWidget::pane { border: 1px solid #262a33; border-radius: 6px; }
QTabBar::tab { background: #1a1d24; color: #8b91a0; padding: 9px 22px;
               border-top-left-radius: 6px; border-top-right-radius: 6px;
               margin-right: 2px; font-size: 13px; }
QTabBar::tab:selected { background: #232733; color: #e8b44c; font-weight: 600; }
QGroupBox { border: 1px solid #2a2f3a; border-radius: 8px; margin-top: 12px;
            padding-top: 16px; font-weight: 600; color: #a9b0bf; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QPushButton { background: #2a3040; color: #e6e9ef; border: 1px solid #39415a;
              border-radius: 6px; padding: 7px 16px; font-size: 13px; }
QPushButton:hover { background: #353d52; }
QPushButton:pressed { background: #232838; }
QPushButton:disabled { background: #1c1f27; color: #565c69; }
QPushButton#accent { background: #b8862b; color: #14161b; font-weight: 700;
                     border: none; }
QPushButton#accent:hover { background: #d19c35; }
QPushButton#accent:disabled { background: #4a3d1e; color: #7a715c; }
QPushButton#danger { background: #7a2e2e; border: none; }
QPushButton#danger:hover { background: #944040; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #1b1e26; border: 1px solid #2c313d; border-radius: 5px;
    padding: 5px 8px; color: #d7dae0; }
QPlainTextEdit { background: #0e1013; border: 1px solid #262a33;
                 border-radius: 6px; color: #9fd08a;
                 font-family: 'Consolas', 'DejaVu Sans Mono', monospace;
                 font-size: 12px; }
QProgressBar { background: #1b1e26; border: 1px solid #2c313d;
               border-radius: 5px; text-align: center; color: #d7dae0; }
QProgressBar::chunk { background: #b8862b; border-radius: 4px; }
QSlider::groove:horizontal { height: 5px; background: #2c313d; border-radius: 2px; }
QSlider::handle:horizontal { width: 15px; margin: -6px 0; border-radius: 7px;
                             background: #e8b44c; }
QLabel#h1 { font-size: 30px; font-weight: 800; color: #e8b44c; }
QLabel#h2 { font-size: 15px; color: #a9b0bf; }
QLabel#stat { font-family: 'Consolas', monospace; font-size: 13px;
              color: #9fd08a; }
QLabel#ladder { font-family: 'Consolas', monospace; font-size: 12px;
                color: #8fb6d0; }
QLabel#warn { color: #d98e5f; }
QLabel#bad { color: #e06666; font-weight: 600; }
QLabel#imgpane { background: #0e1013; border: 1px solid #262a33;
                 border-radius: 6px; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; background: #1b1e26;
    border: 1px solid #39415a; border-radius: 4px; }
QCheckBox::indicator:hover { border: 1px solid #e8b44c; }
QCheckBox::indicator:checked { background: #e8b44c; border: 1px solid #e8b44c; }
"""


def np_to_pixmap(arr, target=None):
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    im = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(im)
    if target is not None:
        pm = pm.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
    return pm


# =============================================================================
# TAB 1 — HOME
# =============================================================================
class HomeTab(QWidget):
    def __init__(self, trainer_path):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 40, 48, 40)
        lay.setSpacing(14)
        t = QLabel("TINY AVATAR 2"); t.setObjectName("h1")
        s = QLabel("a constant-Q wave-interference face model, end to end")
        s.setObjectName("h2")
        lay.addWidget(t); lay.addWidget(s)
        lay.addSpacing(10)
        body = QLabel(
            "A ~7 MB generative model: a VAE maps a 128-dimensional latent to "
            "a few hundred Gabor wave packets, rendered by nothing but "
            "additive wave interference. No pixels stored, no convolutions in "
            "the decoder — the face IS the interference pattern.\n\n"
            "What is new in 2 is the BASIS. Measuring the old models showed "
            "they had abandoned their own carrier: median Q = sigma*freq was "
            "0.22, under half a cycle across the envelope, so the packets "
            "were signed Gaussian blobs, not Gabor wavelets. A blob cannot "
            "make an edge. The constant-Q coupling pins sigma to Q/freq and "
            "forces every packet to oscillate; in a matched comparison that "
            "bought +1.7 dB PSNR and cut invented mid-band structure 44-fold.\n\n"
            "The avatar side runs on phase-transport pursuit: between encoder "
            "keyframes, packets glide along the complex-phasor geodesic "
            "instead of crossfading, which is what keeps the image from "
            "dissolving mid-motion.\n\n"
            "Workflow:  Dataset Prep -> Training Studio -> Avatar Driver.\n"
            "Record one to two minutes of yourself talking and turning your "
            "head, extract face-cropped frames, train (hours, not minutes), "
            "then drive the result live from your webcam.")
        body.setWordWrap(True)
        body.setStyleSheet("font-size: 13.5px; line-height: 150%; color: #c3c8d2;")
        lay.addWidget(body)
        lay.addSpacing(8)
        tp = trainer_path or "NO TRAINER FOUND — put splat_trainer5.py next to this app"
        eng = QLabel(f"engine: {os.path.basename(tp) if trainer_path else tp}")
        eng.setObjectName("stat" if trainer_path else "warn")
        lay.addWidget(eng)
        lay.addStretch(1)


# =============================================================================
# TAB 2 — DATASET PREP
# =============================================================================
class ExtractWorker(QThread):
    progress = pyqtSignal(int, int)
    preview = pyqtSignal(np.ndarray)
    finished_ok = pyqtSignal(int, str)
    failed = pyqtSignal(str)

    def __init__(self, video, out, stride, size, use_face):
        super().__init__()
        self.video, self.out = video, out
        self.stride, self.size, self.use_face = stride, size, use_face
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import cv2 as cv
            os.makedirs(self.out, exist_ok=True)
            cap = cv.VideoCapture(self.video)
            if not cap.isOpened():
                self.failed.emit(f"could not open {self.video}"); return
            rot_op, meta = None, 0
            try:
                cap.set(cv.CAP_PROP_ORIENTATION_AUTO, 0)
                meta = int(round(cap.get(cv.CAP_PROP_ORIENTATION_META))) % 360
                rot_op = {90: cv.ROTATE_90_CLOCKWISE, 180: cv.ROTATE_180,
                          270: cv.ROTATE_90_COUNTERCLOCKWISE}.get(meta)
            except Exception:
                pass
            total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or -1
            face = None
            if self.use_face:
                cpath = os.path.join(cv.data.haarcascades,
                                     "haarcascade_frontalface_default.xml")
                face = cv.CascadeClassifier(cpath)
                if face.empty():
                    face = None
            last_box = None
            i = n = 0
            while not self._stop:
                ok, fr = cap.read()
                if not ok:
                    break
                if rot_op is not None:
                    fr = cv.rotate(fr, rot_op)
                if i % self.stride == 0:
                    box = None
                    if face is not None:
                        g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
                        det = face.detectMultiScale(g, 1.15, 5, minSize=(80, 80))
                        if len(det):
                            x, y, w, h = max(det, key=lambda b: b[2] * b[3])
                            m = int(0.35 * max(w, h))
                            box = (max(x - m, 0), max(y - m, 0),
                                   min(x + w + m, fr.shape[1]),
                                   min(y + h + m, fr.shape[0]))
                            last_box = box
                        else:
                            box = last_box
                    if box is None:
                        H, W = fr.shape[:2]; s = min(H, W)
                        box = ((W - s) // 2, (H - s) // 2,
                               (W + s) // 2, (H + s) // 2)
                    x0, y0, x1, y1 = box
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    s = max(x1 - x0, y1 - y0) // 2
                    H, W = fr.shape[:2]
                    x0, x1 = max(cx - s, 0), min(cx + s, W)
                    y0, y1 = max(cy - s, 0), min(cy + s, H)
                    crop = fr[y0:y1, x0:x1]
                    if crop.size == 0:
                        i += 1; continue
                    crop = cv.resize(crop, (self.size, self.size),
                                     interpolation=cv.INTER_AREA)
                    cv.imwrite(os.path.join(self.out, f"f{n:05d}.jpg"), crop)
                    if n % 25 == 0:
                        self.preview.emit(crop[:, :, ::-1].copy())
                        self.progress.emit(i, total)
                    n += 1
                i += 1
            cap.release()
            if self._stop:
                self.failed.emit("stopped by user"); return
            if n == 0:
                self.failed.emit("no frames extracted"); return
            note = f" (rotated {meta}\u00b0 from phone metadata)" if rot_op else ""
            self.finished_ok.emit(n, self.out + note)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class DatasetTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        lay = QVBoxLayout(self); lay.setContentsMargins(24, 20, 24, 20)
        note = QLabel(
            "One identity, good coverage: record 1-2 minutes of yourself "
            "TALKING, TURNING your head through angles, changing expression, "
            "with a little lighting variation. Coverage of pose + expression "
            "is what lets the single-identity manifold lock and track.")
        note.setWordWrap(True); note.setObjectName("h2")
        lay.addWidget(note)

        vg = QGroupBox("Video  ->  face-cropped frames")
        f = QFormLayout(vg)
        row = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("path to your video (mp4/avi/mkv...)")
        b = QPushButton("Browse"); b.clicked.connect(self.pick_video)
        row.addWidget(self.video_edit); row.addWidget(b)
        f.addRow("video file", row)
        row2 = QHBoxLayout()
        self.out_edit = QLineEdit(os.path.join(APP_DIR, "faces1"))
        b2 = QPushButton("Browse"); b2.clicked.connect(self.pick_out)
        row2.addWidget(self.out_edit); row2.addWidget(b2)
        f.addRow("output folder", row2)
        self.stride = QSpinBox(); self.stride.setRange(1, 30); self.stride.setValue(2)
        f.addRow("keep every Nth frame", self.stride)
        self.size = QSpinBox(); self.size.setRange(64, 512); self.size.setValue(178)
        f.addRow("saved frame size (px)", self.size)
        self.face_chk = QCheckBox(
            "face-detect crop (recommended — otherwise your face is a blob "
            "in a wide frame)")
        self.face_chk.setChecked(True)
        f.addRow(self.face_chk)
        warn = QLabel("Extraction walks the whole video. Training afterwards "
                      "takes HOURS. Both are normal.\n"
                      "Phone videos: portrait clips are stored sideways with "
                      "a rotation tag. The app reads the tag and un-rotates; "
                      "if the preview is still sideways, record in LANDSCAPE.")
        warn.setObjectName("warn"); warn.setWordWrap(True)
        f.addRow(warn)
        hb = QHBoxLayout()
        self.go = QPushButton("Extract frames"); self.go.setObjectName("accent")
        self.go.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop)
        hb.addWidget(self.go); hb.addWidget(self.stop_btn); hb.addStretch(1)
        f.addRow(hb)
        self.bar = QProgressBar(); f.addRow(self.bar)
        lay.addWidget(vg)

        ig = QGroupBox("Already have images?")
        il = QHBoxLayout(ig)
        self.img_dir = QLineEdit()
        self.img_dir.setPlaceholderText(
            "folder of jpg/png of ONE person — used directly by the trainer")
        b3 = QPushButton("Browse"); b3.clicked.connect(self.pick_imgdir)
        b4 = QPushButton("Check folder"); b4.clicked.connect(self.check_folder)
        il.addWidget(self.img_dir); il.addWidget(b3); il.addWidget(b4)
        lay.addWidget(ig)

        bottom = QHBoxLayout()
        self.preview = QLabel("frame preview"); self.preview.setObjectName("imgpane")
        self.preview.setFixedSize(QSize(260, 260))
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status = QLabel(""); self.status.setObjectName("stat")
        self.status.setWordWrap(True)
        bottom.addWidget(self.preview); bottom.addWidget(self.status, 1)
        lay.addLayout(bottom)
        lay.addStretch(1)

    def pick_video(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Video", "", "Video (*.mp4 *.avi *.mkv *.mov *.webm);;All (*)")
        if p: self.video_edit.setText(p)

    def pick_out(self):
        p = QFileDialog.getExistingDirectory(self, "Output folder")
        if p: self.out_edit.setText(p)

    def pick_imgdir(self):
        p = QFileDialog.getExistingDirectory(self, "Image folder")
        if p: self.img_dir.setText(p)

    def check_folder(self):
        d = self.img_dir.text().strip()
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        n = sum(len(glob.glob(os.path.join(d, e))) for e in exts)
        self.status.setText(
            f"{n} images in {d}\n" +
            ("Fine to train on directly — set this as data_dir in the "
             "Training tab." if n >= 200 else
             "Under ~200 images is thin for pose+expression coverage; the "
             "model will still train but tracking range will be narrow."))

    def start(self):
        v = self.video_edit.text().strip()
        if not os.path.exists(v):
            QMessageBox.warning(self, "Tiny Avatar", "Video file not found."); return
        self.worker = ExtractWorker(v, self.out_edit.text().strip(),
                                    self.stride.value(), self.size.value(),
                                    self.face_chk.isChecked())
        self.worker.progress.connect(self.on_prog)
        self.worker.preview.connect(self.on_prev)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.go.setEnabled(False); self.stop_btn.setEnabled(True)
        self.status.setText("extracting...")
        self.worker.start()

    def stop(self):
        if self.worker: self.worker.stop()

    def on_prog(self, done, total):
        if total > 0:
            self.bar.setMaximum(total); self.bar.setValue(done)

    def on_prev(self, rgb):
        self.preview.setPixmap(np_to_pixmap(rgb, self.preview.size()))

    def on_done(self, n, out):
        self.go.setEnabled(True); self.stop_btn.setEnabled(False)
        self.bar.setValue(self.bar.maximum())
        self.status.setText(
            f"wrote {n} frames -> {out}\nNext: Training Studio, set data_dir "
            f"to this folder. First run builds a one-time .npy cache.")

    def on_fail(self, msg):
        self.go.setEnabled(True); self.stop_btn.setEnabled(False)
        self.status.setText(f"extraction failed: {msg}")


# =============================================================================
# TAB 3 — TRAINING STUDIO
# =============================================================================
# PSNR and kl are optional groups: splat_trainer5 prints them, 4q did not.
LOG_RE = re.compile(
    r"step\s+(\d+)\s*/\s*(\d+)\s+rec\s+([\d.eE+-]+)"
    r"(?:\s+\(PSNR\s+([\d.]+)\))?"
    r"(?:\s+kl\s+([\d.eE+-]+))?")


class TrainTab(QWidget):
    def __init__(self, trainer_path, flags):
        super().__init__()
        self.trainer_path, self.flags = trainer_path, flags
        self.proc = None

        outer = QHBoxLayout(self); outer.setContentsMargins(20, 16, 20, 16)
        left = QVBoxLayout(); right = QVBoxLayout()
        outer.addLayout(left, 0); outer.addLayout(right, 1)

        # ---- run settings
        sg = QGroupBox("Run settings"); f = QFormLayout(sg)
        row = QHBoxLayout()
        self.data_dir = QLineEdit(os.path.join(APP_DIR, "faces1"))
        b = QPushButton("..."); b.setFixedWidth(30); b.clicked.connect(self.pick_data)
        row.addWidget(self.data_dir); row.addWidget(b)
        f.addRow("data_dir", row)
        row2 = QHBoxLayout()
        self.out_dir = QLineEdit(os.path.join(APP_DIR, "runs", "tiny2"))
        b2 = QPushButton("..."); b2.setFixedWidth(30); b2.clicked.connect(self.pick_out)
        row2.addWidget(self.out_dir); row2.addWidget(b2)
        f.addRow("out dir", row2)
        self.res = QComboBox(); self.res.addItems(["64", "96", "128", "160", "192"])
        self.res.setCurrentText("128")
        self.res.currentTextChanged.connect(self.refresh_ladder)
        f.addRow("image_size", self.res)
        self.packets = QSpinBox(); self.packets.setRange(32, 2048)
        self.packets.setSingleStep(64); self.packets.setValue(512)
        self.packets.valueChanged.connect(self.refresh_ladder)
        f.addRow("num_packets", self.packets)
        self.batch = QSpinBox(); self.batch.setRange(4, 512); self.batch.setValue(32)
        f.addRow("batch", self.batch)
        self.steps = QSpinBox(); self.steps.setRange(100, 500000)
        self.steps.setSingleStep(1000); self.steps.setValue(30000)
        f.addRow("steps", self.steps)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1e-1); self.lr.setValue(3e-4)
        f.addRow("lr", self.lr)
        self.beta = QDoubleSpinBox(); self.beta.setDecimals(6)
        self.beta.setRange(0.0, 10.0); self.beta.setValue(0.0005)
        f.addRow("beta (low = sharp single identity)", self.beta)
        self.beta_warm = QSpinBox(); self.beta_warm.setRange(1, 100000)
        self.beta_warm.setSingleStep(500); self.beta_warm.setValue(2000)
        f.addRow("beta warmup steps (beta ramps 0 -> beta)", self.beta_warm)
        self.free_bits = QDoubleSpinBox(); self.free_bits.setDecimals(3)
        self.free_bits.setRange(0.0, 1.0); self.free_bits.setSingleStep(0.01)
        f.addRow("free_bits (KL floor)", self.free_bits)
        self.gamma = QDoubleSpinBox(); self.gamma.setDecimals(4)
        self.gamma.setRange(0.0, 1.0); self.gamma.setValue(0.0)
        f.addRow("gamma_floater (0 = off)", self.gamma)
        self.log_every = QSpinBox(); self.log_every.setRange(10, 5000)
        self.log_every.setValue(250)
        f.addRow("log/save every N steps", self.log_every)
        self.ckpt_chk = QCheckBox("--checkpointing (trade ~30% speed for VRAM)")
        self.ckpt_chk.setEnabled("--checkpointing" in self.flags)
        f.addRow(self.ckpt_chk)
        left.addWidget(sg)

        # ---- basis (constant-Q) — new in TinyAvatar2
        bg = QGroupBox("Basis — constant-Q Gabor frame"); bf = QFormLayout(bg)
        self.legacy_chk = QCheckBox(
            "train the LEGACY parameterisation instead (blob-collapse regime)")
        self.legacy_chk.toggled.connect(self.refresh_ladder)
        bf.addRow(self.legacy_chk)
        self.q = QDoubleSpinBox(); self.q.setDecimals(3)
        self.q.setRange(0.05, 4.0); self.q.setSingleStep(0.05); self.q.setValue(0.6)
        self.q.valueChanged.connect(self.refresh_ladder)
        bf.addRow("q  (cycles per envelope sigma)", self.q)
        self.q_slack = QDoubleSpinBox(); self.q_slack.setDecimals(3)
        self.q_slack.setRange(0.0, 3.0); self.q_slack.setSingleStep(0.05)
        self.q_slack.setValue(math.log(2.0))
        bf.addRow("q_slack (ln-octaves of slack)", self.q_slack)
        self.octaves = QSpinBox(); self.octaves.setRange(1, 10)
        self.octaves.setValue(5)
        self.octaves.valueChanged.connect(self.refresh_ladder)
        bf.addRow("octaves", self.octaves)
        self.sig_hi = QDoubleSpinBox(); self.sig_hi.setDecimals(3)
        self.sig_hi.setRange(0.01, 2.0); self.sig_hi.setSingleStep(0.05)
        self.sig_hi.setValue(0.70)
        self.sig_hi.valueChanged.connect(self.refresh_ladder)
        bf.addRow("sig_hi  (sets carrier floor q/sig_hi)", self.sig_hi)
        self.sig_lo = QDoubleSpinBox(); self.sig_lo.setDecimals(4)
        self.sig_lo.setRange(0.0001, 0.5); self.sig_lo.setSingleStep(0.002)
        self.sig_lo.setValue(0.008)
        bf.addRow("sig_lo", self.sig_lo)
        self.f_max = QDoubleSpinBox(); self.f_max.setDecimals(1)
        self.f_max.setRange(0.0, 256.0); self.f_max.setSingleStep(1.0)
        self.f_max.setValue(0.0)
        self.f_max.setSpecialValueText("auto (half pixel Nyquist)")
        self.f_max.valueChanged.connect(self.refresh_ladder)
        bf.addRow("f_max (cyc/image; run spectrum_audit first)", self.f_max)
        self.gist = QDoubleSpinBox(); self.gist.setDecimals(2)
        self.gist.setRange(0.0, 0.9); self.gist.setSingleStep(0.05)
        self.gist.setValue(0.0)
        self.gist.valueChanged.connect(self.refresh_ladder)
        bf.addRow("gist_frac (carrier-free; 0 — see warning)", self.gist)
        self.band_mode = QComboBox()
        self.band_mode.addItems(["permute", "interleave", "striped"])
        bf.addRow("band_mode (octave -> anchor lattice mapping)",
                  self.band_mode)
        self.detail = QDoubleSpinBox(); self.detail.setDecimals(2)
        self.detail.setRange(0.0, 5.0); self.detail.setSingleStep(0.25)
        self.detail.setValue(1.0)
        bf.addRow("detail (gradient-weighted recon loss)", self.detail)
        self.ladder = QLabel(""); self.ladder.setObjectName("ladder")
        self.ladder.setWordWrap(True)
        bf.addRow(self.ladder)
        self.ladder_warn = QLabel(""); self.ladder_warn.setObjectName("bad")
        self.ladder_warn.setWordWrap(True)
        bf.addRow(self.ladder_warn)
        left.addWidget(bg)

        # ---- pulse
        pg = QGroupBox("Pulse check"); pl = QVBoxLayout(pg)
        self.pulse_btn = QPushButton("Take the pulse")
        self.pulse_btn.clicked.connect(self.pulse)
        self.pulse_lbl = QLabel("—"); self.pulse_lbl.setObjectName("stat")
        self.pulse_lbl.setWordWrap(True)
        pl.addWidget(self.pulse_btn); pl.addWidget(self.pulse_lbl)
        left.addWidget(pg)

        # ---- controls
        cg = QGroupBox("Run"); cl = QVBoxLayout(cg)
        self.resume_lbl = QLabel(""); self.resume_lbl.setObjectName("warn")
        self.resume_lbl.setWordWrap(True)
        cl.addWidget(self.resume_lbl)
        hb = QHBoxLayout()
        self.start_btn = QPushButton("Start training"); self.start_btn.setObjectName("accent")
        self.start_btn.clicked.connect(lambda: self.start(resume=False))
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(lambda: self.start(resume=True))
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop)
        hb.addWidget(self.start_btn); hb.addWidget(self.resume_btn)
        hb.addWidget(self.stop_btn)
        cl.addLayout(hb)
        self.prog = QProgressBar(); cl.addWidget(self.prog)
        self.stat_lbl = QLabel("idle"); self.stat_lbl.setObjectName("stat")
        self.stat_lbl.setWordWrap(True)
        cl.addWidget(self.stat_lbl)
        self.sys_lbl = QLabel(""); self.sys_lbl.setObjectName("stat")
        cl.addWidget(self.sys_lbl)
        left.addWidget(cg)
        left.addStretch(1)

        # ---- right: previews + console
        prow = QHBoxLayout()
        self.recon_lbl = QLabel("recon preview\n(appears at first log step)")
        self.sample_lbl = QLabel("sample preview")
        for w in (self.recon_lbl, self.sample_lbl):
            w.setObjectName("imgpane")
            w.setMinimumSize(QSize(300, 300))
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            w.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Expanding)
            prow.addWidget(w)
        right.addLayout(prow, 1)
        self.console = QPlainTextEdit(); self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        right.addWidget(self.console, 1)

        self.sys_timer = QTimer(self); self.sys_timer.timeout.connect(self.sys_tick)
        self.sys_timer.start(1500)
        self.img_timer = QTimer(self); self.img_timer.timeout.connect(self.img_tick)
        self.out_dir.textChanged.connect(self.scan_resume)
        self.scan_resume()
        self.refresh_ladder()

    # -- pickers
    def pick_data(self):
        p = QFileDialog.getExistingDirectory(self, "Data dir")
        if p: self.data_dir.setText(p)

    def pick_out(self):
        p = QFileDialog.getExistingDirectory(self, "Out dir")
        if p: self.out_dir.setText(p)

    # -- live octave ladder
    def refresh_ladder(self, *_):
        if self.legacy_chk.isChecked():
            self.ladder.setText(
                "LEGACY parameterisation: sigma and freq are sampled "
                "independently (sigma 0.012-0.152, freq 1-16 cyc/image). No "
                "octave ladder, no Q coupling. This is the regime that "
                "measured median Q = 0.22 — signed blobs, not wavelets.")
            self.ladder_warn.setText("")
            return
        rows, warn = octave_ladder(
            self.q.value(), self.sig_hi.value(),
            self.f_max.value() or None, self.octaves.value(),
            int(self.res.currentText()), self.packets.value(), self.gist.value())
        nyq = int(self.res.currentText()) / 2
        head = (f"octave ladder (pixel Nyquist {nyq:.0f} cyc/image):\n")
        body = "\n".join(
            f"  {lo:7.2f} - {hi:7.2f} cyc/img   {n:4d} packets   "
            f"sigma {self.q.value()/hi:.4f}-{self.q.value()/max(lo,1e-9):.4f}"
            for lo, hi, n in rows)
        self.ladder.setText(head + body)
        self.ladder_warn.setText(warn or "")

    # -- resume detection
    def scan_resume(self):
        out = self.out_dir.text().strip()
        pts = sorted(glob.glob(os.path.join(out, "model*.pt")))
        caches = glob.glob(os.path.join(out, "faces_cache_*.npy"))
        bits = []
        self._resume_path = None
        if pts:
            p = max(pts, key=os.path.getmtime)
            self._resume_path = p
            age = (time.time() - os.path.getmtime(p)) / 3600
            bits.append(f"checkpoint found: {os.path.basename(p)} "
                        f"({age:.1f} h old) — Resume continues it")
        if caches:
            bits.append(f"cache present ({os.path.basename(caches[0])}) — "
                        "no re-preprocessing needed")
        self.resume_lbl.setText("\n".join(bits) if bits
                                else "fresh run — no checkpoint in this out dir")
        self.resume_btn.setEnabled(bool(pts) and "--resume" in self.flags)

    # -- pulse
    def pulse(self):
        import torch
        S = int(self.res.currentText())
        B, N = self.batch.value(), self.packets.value()
        chunk = 64
        lines = []
        d = self.data_dir.text().strip()
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        n_img = sum(len(glob.glob(os.path.join(d, e))) for e in exts)
        cache_gb = n_img * 3 * S * S / 1e9
        lines.append(f"dataset: {n_img} images -> cache {cache_gb:.2f} GB "
                     f"(uint8 {S}px)")
        work_gb = B * chunk * S * S * 4 * 6 / 1e9
        lines.append(f"renderer working set ~{work_gb:.2f} GB "
                     f"(batch {B} x chunk {chunk} @ {S}px, heuristic x6)")
        ips_guess = 40.0 if S >= 128 else 160.0
        hrs = self.steps.value() * B / ips_guess / 3600
        lines.append(f"time estimate: {self.steps.value()} steps x batch {B} "
                     f"= {self.steps.value()*B/1e6:.2f}M images -> "
                     f"~{hrs:.1f} h at ~{ips_guess:.0f} img/s")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            lines.append(f"VRAM: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")
            if cache_gb > free / 1e9 - 3.0:
                lines.append("dataset will NOT sit in VRAM — the trainer "
                             "falls back to pinned/pageable RAM")
            else:
                lines.append("dataset fits resident in VRAM")
            if work_gb > free / 1e9 * 0.5:
                lines.append(f"renderer estimate is over half your free VRAM "
                             f"— consider batch {max(8, B//2)} or "
                             f"--checkpointing")
            else:
                lines.append("renderer estimate looks comfortable")
        else:
            lines.append("no CUDA visible from here — CPU training works but "
                         "is 10-100x slower; the numbers above still apply "
                         "to RAM")
        self.pulse_lbl.setText("\n".join(lines))

    # -- launch
    def build_args(self, resume=False):
        out = self.out_dir.text().strip()
        a = ["-u", self.trainer_path,
             "--data_dir", self.data_dir.text().strip(),
             "--out", out,
             "--image_size", self.res.currentText(),
             "--num_packets", str(self.packets.value()),
             "--batch", str(self.batch.value()),
             "--steps", str(self.steps.value()),
             "--beta", f"{self.beta.value():g}",
             "--beta_warmup_steps", str(self.beta_warm.value()),
             "--lr", f"{self.lr.value():g}",
             "--free_bits", f"{self.free_bits.value():g}",
             "--gamma_floater", f"{self.gamma.value():g}",
             "--log_every", str(self.log_every.value())]
        if self.legacy_chk.isChecked():
            a += ["--legacy"]
        else:
            a += ["--q", f"{self.q.value():g}",
                  "--q_slack", f"{self.q_slack.value():g}",
                  "--octaves", str(self.octaves.value()),
                  "--sig_lo", f"{self.sig_lo.value():g}",
                  "--sig_hi", f"{self.sig_hi.value():g}",
                  "--gist_frac", f"{self.gist.value():g}",
                  "--band_mode", self.band_mode.currentText(),
                  "--detail", f"{self.detail.value():g}"]
            if self.f_max.value() > 0:
                a += ["--f_max", f"{self.f_max.value():g}"]
        if self.ckpt_chk.isChecked():
            a += ["--checkpointing"]
        if resume and self._resume_path:
            a += ["--resume", self._resume_path]
        # every flag we send must be one the trainer declares. v3 relied on a
        # guard that silently dropped unknown flags, which is how --aug and
        # --disk became controls that did nothing.
        unknown = [t for t in a if t.startswith("--") and t not in self.flags]
        return a, unknown

    def start(self, resume=False):
        if self.proc is not None:
            return
        if not self.trainer_path:
            QMessageBox.critical(self, "Tiny Avatar",
                                 "No trainer script found next to the app.")
            return
        if self.legacy_chk.isChecked():
            r = QMessageBox.question(
                self, "Tiny Avatar",
                "Legacy parameterisation trains the blob-collapse regime "
                "(median Q ~0.22 — packets abandon their carrier). It is "
                "here as the control arm for comparisons, not as a way to "
                "make a better avatar.\n\nTrain legacy anyway?")
            if r != QMessageBox.StandardButton.Yes:
                return
        args, unknown = self.build_args(resume)
        if unknown:
            QMessageBox.critical(
                self, "Tiny Avatar",
                f"This app would send flags {os.path.basename(self.trainer_path)} "
                f"does not declare: {', '.join(unknown)}.\n\nRefusing to "
                f"launch rather than have them silently ignored.")
            return
        os.makedirs(self.out_dir.text().strip(), exist_ok=True)
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_out)
        self.proc.finished.connect(self.on_fin)
        self.console.appendPlainText(f"$ {sys.executable} {' '.join(args)}\n")
        self.proc.start(sys.executable, args)
        self.start_btn.setEnabled(False); self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stat_lbl.setText("launching trainer process...")
        self.prog.setMaximum(self.steps.value()); self.prog.setValue(0)
        self.img_timer.start(2000)

    def stop(self):
        if self.proc:
            self.console.appendPlainText(
                "\n[stopping trainer — the checkpoint is written every log "
                "step with optimizer and schedule state, so Resume continues "
                "exactly where this left off]")
            self.proc.kill()

    def on_out(self):
        txt = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in txt.splitlines():
            if line.strip():
                self.console.appendPlainText(line)
            m = LOG_RE.search(line)
            if m:
                step, tot = int(m.group(1)), int(m.group(2))
                self.prog.setMaximum(tot); self.prog.setValue(step)
                bits = [f"step {step}/{tot}", f"rec {m.group(3)}"]
                if m.group(4):
                    bits.append(f"PSNR {m.group(4)} dB")
                if m.group(5):
                    bits.append(f"kl {m.group(5)}")
                self.stat_lbl.setText("   ".join(bits))

    def on_fin(self, code, _status):
        self.console.appendPlainText(f"\n[trainer exited, code {code}]")
        self.proc = None
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.img_timer.stop()
        self.img_tick()
        self.scan_resume()
        self.stat_lbl.setText(f"stopped (exit {code})" if code else "done")

    def sys_tick(self):
        bits = []
        try:
            import psutil
            vm = psutil.virtual_memory()
            bits.append(f"RAM {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB "
                        f"({vm.percent:.0f}%)  CPU {psutil.cpu_percent():.0f}%")
        except Exception:
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            mi = pynvml.nvmlDeviceGetMemoryInfo(h)
            bits.append(f"GPU {u.gpu}%  VRAM {mi.used/1e9:.1f}/"
                        f"{mi.total/1e9:.1f} GB")
        except Exception:
            try:
                import torch
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    bits.append(f"VRAM {(total-free)/1e9:.1f}/"
                                f"{total/1e9:.1f} GB used")
            except Exception:
                pass
        self.sys_lbl.setText("   ".join(bits))

    def img_tick(self):
        out = self.out_dir.text().strip()
        for pat, lbl in (("recon_*.png", self.recon_lbl),
                         ("sample_*.png", self.sample_lbl)):
            files = sorted(glob.glob(os.path.join(out, pat)))
            if files and files[-1] != getattr(lbl, "_shown", None):
                pm = QPixmap(files[-1])
                if not pm.isNull():
                    lbl.setPixmap(pm.scaled(
                        lbl.size(), Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    lbl._shown = files[-1]


# =============================================================================
# TAB 4 — AVATAR DRIVER
# =============================================================================
def load_model(trainer_path, model_path, dev):
    """THE fix. Always through the trainer's load_splatvae, which reads the
    checkpoint's own qmode/q/octaves/sig_hi/f_max and rebuilds the matching
    renderer. Constructing SplatVAE from (image_size, num_packets) alone
    applies the current defaults instead — no error, wrong pixels, measured
    at up to 0.31 absolute on a 0-1 image."""
    import torch
    ST = import_trainer(trainer_path)
    if hasattr(ST, "load_splatvae"):
        model, ck = ST.load_splatvae(model_path)
    else:                      # very old trainer without the loader
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
        model.load_state_dict(ck["sd"]); model.eval()
    return model.to(dev), ck


def describe_ckpt(ck):
    if ck.get("qmode", False):
        return (f"{ck['image_size']}px / {ck['num_packets']} packets  "
                f"constant-Q  q={ck.get('q', '?')}  "
                f"octaves={ck.get('octaves', '?')}  "
                f"f_max={ck.get('f_max', '?')}  "
                f"band_mode={ck.get('band_mode', 'striped (pre-key)')}  "
                f"trainer={ck.get('trainer', '?')}")
    return (f"{ck['image_size']}px / {ck['num_packets']} packets  "
            f"LEGACY parameterisation (sigma and freq independent) — "
            f"expect the blob-collapse look")


class AvatarWorker(QThread):
    frame = pyqtSignal(np.ndarray, np.ndarray, float)
    status = pyqtSignal(str)

    def __init__(self, trainer_path, model_path, source):
        super().__init__()
        self.trainer_path, self.model_path = trainer_path, model_path
        self.source = source            # "webcam" | "walk"
        self.mode = "phase"
        self.kf = 8
        self.alpha = 0.35
        self.norm = True
        self.align = True
        self.walk_step, self.z_max = 2.5, 12.0
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, ck = load_model(self.trainer_path, self.model_path, dev)
            ren = model.ren
            self.status.emit(describe_ckpt(ck) + f"  on {dev}")
            if self.source == "webcam":
                self._webcam(model, ren, dev, torch)
            else:
                self._walk(model, ren, dev, torch)
        except Exception as e:
            self.status.emit(f"avatar failed: {type(e).__name__}: {e}")

    def _webcam(self, model, ren, dev, torch):
        import cv2 as cv
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            self.status.emit("webcam failed to open — try Latent walk instead")
            return
        P = T = None
        f = 0
        framer = FaceFramer() if self.align else None
        t_last, fps = time.time(), 0.0
        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                break
            if framer is not None:
                crop = framer.crop(frame)
            else:
                h, w = frame.shape[:2]; s = min(h, w)
                crop = frame[(h - s) // 2:(h + s) // 2,
                             (w - s) // 2:(w + s) // 2]
            x = cv.resize(crop, (ren.H, ren.W))[:, :, ::-1].astype(
                np.float32) / 255.0
            if self.norm:
                x = normalize_crop(x)
            xt = torch.from_numpy(np.ascontiguousarray(
                x.transpose(2, 0, 1)))[None].to(dev)
            need = (self.mode == "direct") or (f % max(1, self.kf) == 0) \
                or (T is None)
            if need:
                with torch.no_grad():
                    mu, _ = model.enc(xt)
                    T = ren.activate(model.dec(mu).float())
                if P is None:
                    P = clone_params(T)
            with torch.no_grad():
                P = clone_params(T) if self.mode == "direct" else \
                    pursue(P, T, self.alpha, self.mode)
                img = render_image(ren, P)
            av = (img[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            cam = cv.resize(crop, (256, 256))[:, :, ::-1].copy()
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6); t_last = now
            self.frame.emit(cam, av, fps)
            f += 1
        cap.release()

    def _walk(self, model, ren, dev, torch):
        LATENT = getattr(import_trainer(self.trainer_path), "LATENT", 128)
        g = torch.Generator().manual_seed(0)
        z = torch.randn(1, LATENT, generator=g).to(dev)

        def keyframe(z):
            with torch.no_grad():
                return ren.activate(model.dec(z).float())

        T = keyframe(z); P = clone_params(T)
        f = 0
        t_last, fps = time.time(), 0.0
        blank = np.zeros((256, 256, 3), np.uint8)
        while not self._stop:
            if f % max(1, self.kf) == 0:
                z = z + self.walk_step * torch.randn(
                    1, LATENT, generator=g).to(dev)
                z = z * min(1.0, self.z_max / (z.norm() + 1e-9))
                T = keyframe(z)
            with torch.no_grad():
                mode = self.mode if self.mode != "direct" else "phase"
                P = pursue(P, T, self.alpha, mode)
                img = render_image(ren, P)
            av = (img[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6); t_last = now
            self.frame.emit(blank, av, fps)
            f += 1
            time.sleep(max(0.0, 1 / 30 - (time.time() - now)))


class AvatarTab(QWidget):
    def __init__(self, trainer_path):
        super().__init__()
        self.trainer_path = trainer_path
        self.worker = None
        lay = QVBoxLayout(self); lay.setContentsMargins(20, 16, 20, 16)

        top = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.describe)
        rb = QPushButton("Rescan"); rb.clicked.connect(self.refresh_models)
        mb = QPushButton("Browse..."); mb.clicked.connect(self.pick_model)
        top.addWidget(QLabel("model")); top.addWidget(self.model_combo, 1)
        top.addWidget(rb); top.addWidget(mb)
        lay.addLayout(top)
        self.ckpt_lbl = QLabel(""); self.ckpt_lbl.setObjectName("stat")
        self.ckpt_lbl.setWordWrap(True)
        lay.addWidget(self.ckpt_lbl)

        panes = QHBoxLayout()
        self.cam_lbl = QLabel("camera"); self.av_lbl = QLabel("avatar")
        for w in (self.cam_lbl, self.av_lbl):
            w.setObjectName("imgpane")
            w.setMinimumSize(QSize(360, 360))
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            w.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Expanding)
            panes.addWidget(w)
        lay.addLayout(panes, 1)

        ctl = QGroupBox("Drive"); g = QGridLayout(ctl)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "phase pursuit (certified transport)",
            "lerp pursuit (baseline)",
            "direct (encode every frame)",
            "screw pursuit (demo, not certified)",
            "dispersion pursuit (demo, not certified)"])
        self.mode_combo.currentIndexChanged.connect(self.push_params)
        g.addWidget(QLabel("mode"), 0, 0); g.addWidget(self.mode_combo, 0, 1)
        self.kf_sl = QSlider(Qt.Orientation.Horizontal)
        self.kf_sl.setRange(1, 30); self.kf_sl.setValue(8)
        self.kf_val = QLabel("8")
        self.kf_sl.valueChanged.connect(
            lambda v: (self.kf_val.setText(str(v)), self.push_params()))
        g.addWidget(QLabel("keyframe every N frames"), 1, 0)
        g.addWidget(self.kf_sl, 1, 1); g.addWidget(self.kf_val, 1, 2)
        self.al_sl = QSlider(Qt.Orientation.Horizontal)
        self.al_sl.setRange(5, 100); self.al_sl.setValue(35)
        self.al_val = QLabel("0.35")
        self.al_sl.valueChanged.connect(
            lambda v: (self.al_val.setText(f"{v/100:.2f}"), self.push_params()))
        g.addWidget(QLabel("pursuit alpha"), 2, 0)
        g.addWidget(self.al_sl, 2, 1); g.addWidget(self.al_val, 2, 2)
        self.norm_chk = QCheckBox("normalize input (fights the dark-head "
                                  "domain gap)")
        self.norm_chk.setChecked(True); self.norm_chk.toggled.connect(self.push_params)
        g.addWidget(self.norm_chk, 3, 0, 1, 3)
        self.align_chk = QCheckBox("face-align input (crop live face the same "
                                   "way Dataset Prep cropped the training "
                                   "frames)")
        self.align_chk.setChecked(True); self.align_chk.toggled.connect(self.push_params)
        g.addWidget(self.align_chk, 4, 0, 1, 3)
        hb = QHBoxLayout()
        self.cam_btn = QPushButton("Start webcam"); self.cam_btn.setObjectName("accent")
        self.cam_btn.clicked.connect(lambda: self.start("webcam"))
        self.walk_btn = QPushButton("Latent walk (no camera)")
        self.walk_btn.clicked.connect(lambda: self.start("walk"))
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop)
        hb.addWidget(self.cam_btn); hb.addWidget(self.walk_btn)
        hb.addWidget(self.stop_btn); hb.addStretch(1)
        g.addLayout(hb, 5, 0, 1, 3)
        lay.addWidget(ctl)
        self.status = QLabel("load a model, then Start"); self.status.setObjectName("stat")
        lay.addWidget(self.status)
        self.refresh_models()

    def refresh_models(self):
        self.model_combo.clear()
        pats = [os.path.join(APP_DIR, "runs", "*", "*.pt"),
                os.path.join(APP_DIR, "runs", "*", "*", "*.pt"),
                os.path.join(APP_DIR, "*.pt")]
        seen = []
        for p in pats:
            for f in sorted(glob.glob(p)):
                if f not in seen:
                    seen.append(f)
        for f in seen:
            self.model_combo.addItem(os.path.relpath(f, APP_DIR), f)
        if not seen:
            self.model_combo.addItem("(no .pt found — train one, or Browse)", "")

    def describe(self, *_):
        mp = self.model_combo.currentData()
        if not mp or not os.path.exists(mp) or not self.trainer_path:
            self.ckpt_lbl.setText(""); return
        try:
            import torch
            ck = torch.load(mp, map_location="cpu", weights_only=False)
            self.ckpt_lbl.setText(describe_ckpt(ck))
        except Exception as e:
            self.ckpt_lbl.setText(f"could not read checkpoint: {e}")

    def pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Model", APP_DIR,
                                           "PyTorch (*.pt)")
        if p:
            self.model_combo.insertItem(0, os.path.basename(p), p)
            self.model_combo.setCurrentIndex(0)

    def _mode(self):
        idx = self.mode_combo.currentIndex()
        modes = ["phase", "lerp", "direct", "screw", "dispersion"]
        return modes[idx] if idx < len(modes) else "phase"

    def push_params(self):
        if self.worker:
            self.worker.mode = self._mode()
            self.worker.kf = self.kf_sl.value()
            self.worker.alpha = self.al_sl.value() / 100
            self.worker.norm = self.norm_chk.isChecked()
            self.worker.align = self.align_chk.isChecked()

    def start(self, source):
        mp = self.model_combo.currentData()
        if not mp or not os.path.exists(mp):
            QMessageBox.warning(self, "Tiny Avatar",
                                "Pick a trained .pt model first (Training "
                                "Studio produces model5_constQ.pt).")
            return
        self.stop()
        self.worker = AvatarWorker(self.trainer_path, mp, source)
        self.push_params()
        self.worker.frame.connect(self.on_frame)
        self.worker.status.connect(self.status.setText)
        self.worker.start()
        self.cam_btn.setEnabled(False); self.walk_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop(self):
        if self.worker:
            self.worker.stop(); self.worker.wait(2000)
            self.worker = None
        self.cam_btn.setEnabled(True); self.walk_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_frame(self, cam, av, fps):
        self.cam_lbl.setPixmap(np_to_pixmap(cam, self.cam_lbl.size()))
        self.av_lbl.setPixmap(np_to_pixmap(av, self.av_lbl.size()))
        self.status.setText(
            f"{self._mode()}  kf {self.kf_sl.value()}  "
            f"alpha {self.al_sl.value()/100:.2f}  {fps:.0f} fps")


# =============================================================================
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tiny Avatar 2")
        self.resize(1220, 800)
        trainer = find_trainer()
        flags = trainer_flags(trainer) if trainer else set()
        tabs = QTabWidget()
        tabs.addTab(HomeTab(trainer), "Home")
        tabs.addTab(DatasetTab(), "Dataset Prep")
        self.train_tab = TrainTab(trainer, flags)
        tabs.addTab(self.train_tab, "Training Studio")
        self.avatar_tab = AvatarTab(trainer)
        tabs.addTab(self.avatar_tab, "Avatar Driver")
        self.setCentralWidget(tabs)

    def closeEvent(self, ev):
        try:
            self.avatar_tab.stop()
            if self.train_tab.proc:
                self.train_tab.proc.kill()
        except Exception:
            pass
        ev.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

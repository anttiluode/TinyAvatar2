# TinyAvatar 2

![ragdoll](ragdoll.png)

A generative face model built out of nothing but wave interference — plus the
trainer, the studio that drives it, and now **splat_ragdoll.py**: grab the
face with the mouse and pull it, and the rest of the head follows along the
learned manifold.

A VAE maps a 128-dimensional latent to a few hundred **Gabor wave packets**
(oriented sinusoids under Gaussian envelopes). The image is their additive
interference on the canvas. No pixels are stored and there are no
convolutions in the decoder — the face *is* the interference pattern.

> Do not hype. Do not lie. Just show.

---

## Play with it now

A legacy checkpoint (`model2.pt`, 96px / 256 packets) is included in this
repo so you can try it without training anything.

```bash
pip install torch opencv-python numpy
python splat_ragdoll.py --model model2.pt          # the ragdoll
python splat_ragdoll.py --gates --model model2.pt  # the science run
```

`splat_ragdoll.py` needs `splat_trainer5.py` and `splat_trainer3v2.py` next
to it (both in this repo). It loads legacy and constant-Q checkpoints through
the same loader and prints which parameterisation it found.

### What you are doing when you drag

**MANIFOLD mode** (default). A click grabs a soft cluster of packets
(amplitude × envelope × grab-radius weighted); dragging creates a *pin*. The
app then solves, live,

```
min_z  Σ_i || centroid_i(z) − target_i ||²    (+ identity bias)
```

by damped least squares on the decoder Jacobian ∂c/∂z — a (2m × 128)
Jacobian for m pins, so the per-iteration solve is a tiny (2m × 2m) system.
The identity-retention pull toward the anchor face is applied in the **null
space** of the pin Jacobian (the standard secondary-task trick from robot
IK): pins are honored exactly, and the face relaxes back toward itself only
in directions the pins don't constrain. The VAE's learned covariance is the
rig. There is no skeleton, no blendshapes, no weight painting — if the
training data correlated two features, pulling one drags the other.

**DIRECT mode** (`m` to toggle). The same grab, applied straight to the
activated packet parameters after the decoder: rigid SE(2) translate, rotate
(`,` / `.`), and a scale gesture (`<` / `>`) that preserves constant-Q
**exactly** — σ scales up, carrier frequency scales down, Q = σ·f invariant
to machine precision. No solver, no manifold; parameter-space surgery.
Keys `1`–`5` gate octave bands out of the grab, so you can pull the coarse
head-outline band while leaving fine texture untouched, or vice versa.

Other keys: `p` make the held pin persistent (drop several, drag them
against each other) · `c` clear · `[` `]` grab radius · `s` spring-back ·
`w` wobble (latent velocity — release and the face flops) · `b` bake the
current state as the new identity · `r` reset · `n` new random identity ·
`--image me.png` starts from an encoding · `--record out.mp4`.

### The ragdoll gates, measured

![pic](pic.png)

The claim "pull one eye and the head moves with it" is a registered gate
(RG1), not marketing. Drag a pin at the left-eye region by +0.08 in x,
solve to tolerance, and measure how far the **un-pinned** mirror cluster
moved — against a direct-mode control, which moves only the grabbed weights.

| model | RG1 mirror motion | direct control | ratio (gate ≥ 3×) | verdict |
|---|---|---|---|---|
| `model5_constQ.pt` (128px/512, constant-Q) | 9.7 px | 0.2 px | **9.7×** | [V] |
| `model2.pt` (96px/256, legacy, own-face) | 6.2 px | 0.04 px | **6.2×** | [V] |
| `model2.pt` (96px/256, legacy, CelebA-30k) | 4.4 px | 0.01 px | **4.4×** | [V] |

Negative control, found for free during development: on a **random-weight**
decoder the same gate returns ratio 0.41 — **[K]**. The machinery cannot
pass its own gate; a pass is the trained manifold talking, not the solver
flattering itself.

RG4 ramps the drag from 0.02 to 0.30 of the frame and logs pin error, ‖z‖,
and moiré. The three models answer differently, and the difference is the
most interesting number in the run:

- The **CelebA model refuses the pin**: pin error climbs to 13.6 px while
  moiré stays ≈ 0. A *stiff* manifold — it would rather miss your target
  than leave the data distribution. (This is also why webcam driving worked
  out of the box: the manifold hosts poses directionally.)
- The **constant-Q model obeys the pin** all the way out and pays in moiré
  (0.48 by drag 0.20). A *compliant* manifold — it follows you off the
  distribution and shatters there.
- A model trained on ~1900 frames of one person sits between: you can drag
  that person around and the non-moving parts hold still.

Stiffness versus compliance of a learned manifold, measured with a mouse.

Remaining honest caveats: RG2 (nullspace identity retention) passed 0.84 on
one model and returned "no converged pairs" on two — the convergence
tolerance is too tight for those models, which is a limitation of the gate,
not a pass. RG3 (≥ 20 FPS) failed on the 128px model because the app
currently runs **CPU-only**; 96px runs at ~59 FPS on CPU. And the grab is
spatial soft-selection at a chosen radius and band — the model has **no
learned hierarchy** (see the tree section below), so "grab the eye" means
"grab what is painted there", nothing more.

---

## The size, honestly

The old headline said "~7 MB model". A shipped training checkpoint measures
**96 MB**. Both numbers are real; they measure different things. Exact
arithmetic from the shipped architectures:

| piece | 96px/256 line | 128px/512 line |
|---|---|---|
| decoder (z → packets) — *the generator* | **7.1 MB** | 12.9 MB |
| renderer | 0 learned parameters (buffers only) | 0 |
| encoder (image → z) | 15.9 MB | 19.5 MB |
| all weights | 23.0 MB | 32.4 MB |
| training checkpoint on disk (+ Adam moments, 2 extra copies of every weight) | ~69 MB | **~97 MB** |

So: the thing that *generates* a face from a 128-float latent is the 7 MB
decoder plus a parameter-free renderer. The 96 MB file additionally carries
the encoder and the optimizer state needed to resume training. To ship an
inference-only checkpoint:

```python
import torch
ck = torch.load("model5_constQ.pt", map_location="cpu", weights_only=False)
ck.pop("opt", None); ck.pop("sched", None)
torch.save(ck, "model5_constQ_infer.pt")     # ~32 MB, loads identically
```

Per-frame state is still 128 floats — the z-vector — regardless of any of
the above.

---

## The main science: constant-Q

### The old models had abandoned their own carrier

A Gabor packet's character is captured by `Q = sigma * freq` — cycles of
carrier across one envelope sigma. `splat_trainer3v2` sampled sigma and freq
independently, so the model could pick any Q. Measured on the shipped
`model2.pt`: **median Q = 0.22**, under half a cycle across the visible
envelope, with zero packets above Q = 1.5. The trained basis was a mixture
of signed Gaussian *blobs*, not a Gabor frame. That is the blur — one Gabor
can represent an edge; one blob cannot.

This falsified the hypothesis the trainer was written to test (that moiré
came from *too much* Q; it was the opposite regime), so the coupling was
inverted from a ceiling into a **floor**:

```
sigma = clamp( (q / freq) * exp(q_slack * tanh(raw)), sig_lo, sig_hi )
```

With `q = 0.6` and one octave of slack, Q is confined to roughly
[0.3, 1.2] — blob collapse is forbidden and every packet must oscillate.

### The result, in a matched comparison

96px / 256 packets, 1937 own-face images, 3000 steps per arm, everything
else identical:

| gate | constant-Q | legacy |
|---|---|---|
| **Q2 PRIMARY** — PSNR, eval mode | **14.39** | 12.70 |
| Q2 — PSNR, batch statistics | **14.38** | 12.70 |
| Q1 — moiré index | **0.0003** | 0.0132 |
| Q3 — beat, overlap-normalised | **0.4217** | 0.6447 |
| Q4 — median Q | **0.49** | 0.42 |

Forcing carrier use bought **+1.7 dB** and cut invented mid-band structure
44-fold. PSNR is reported in both BatchNorm modes because running statistics
are poorly estimated at 3000 steps and can fake a gap alone; the modes agree
to 0.01 dB. **Honest scope:** one dataset, one identity. The supported claim
is "constant-Q beat legacy here", not "constant-Q is better".

### The octave ladder, and the hole it once had

Constant-Q has a consequence: the sigma ceiling sets a **carrier floor**
(`freq >= q/sig_hi`). An earlier build parked carrier-free "gist" Gaussians
at f = 0 while carriers started at 5 cyc/img — so nothing represented
(0, 5), which on a face is the head outline and feature layout. It rendered
as frosted glass plus fine stripes. The fix was one continuous constant-Q
family from f = 1 to Nyquist/2. At 128px / 512 packets:

```
   1 -  2 cyc/img   103 packets   sigma 0.300-0.600   head outline / lighting
   2 -  4 cyc/img   102 packets   sigma 0.150-0.300
   4 -  8 cyc/img   103 packets   sigma 0.075-0.150   feature geometry
   8 - 16 cyc/img   102 packets   sigma 0.037-0.075
  16 - 32 cyc/img   102 packets   sigma 0.019-0.037   fine texture
```

The band → packet-index map is a **recorded checkpoint key** (`band_mode`),
because changing it changes every packet's frequency range while the
state_dict shape stays identical: a striped-trained model rendered by
interleaved code produces 0.98 max absolute error on a 0–1 image, silently.
Old checkpoints load as `striped` and render bit-identically. New runs
default to `permute`. For the same reason, **every consumer must load
through `load_splatvae()`** — constructing the model from
(image_size, num_packets) alone renders legacy checkpoints with wrong
formulas at up to 0.57 max error, with no exception raised.

### The dyadic tree that did not work

The natural next idea was a bifurcating quadtree — parents spawn four
children at double frequency and half sigma (Q preserved), children pinned
inside the parent envelope. Because that geometry is fixed, coefficient
fitting is linear least squares — an upper bound on what any trainer could
reach with it, computable in seconds, *before* spending GPU time:

| basis (matched N = 341, 4 seeds) | lsq PSNR | beat, normalised |
|---|---|---|
| tree, uniform subdivision | 23.44 ± 0.67 | 0.724 |
| tree, residual-driven splitting | 21.68 ± 0.57 | 0.732 |
| **flat basis, same octave histogram** | **22.55 ± 0.24** | 0.725 |

The adaptive tree is 0.87 dB **worse** than a flat basis that merely copies
its histogram, and beat did not drop — pinning children inside parents
relocates fringes among siblings rather than killing them. Everything the
tree bought was the histogram. Not built. (Untested residue: a tree might
still help the *amortized* problem — a decoder predicting structured output —
and the rigid-inheritance motion claim. The ragdoll gets the "grab a
feature" behaviour without it, by soft selection.)

---

## Files

| file | what it is |
|---|---|
| `splat_ragdoll.py` | **the ragdoll.** Latent IK + SlapStack-style direct editing on any checkpoint. `--selftest`, `--gates`, `--record` |
| `splat_trainer5.py` | the trainer. Constant-Q renderer, Q1–Q4 gates, `--audit`, `--compare`, `--resume`, CPU `--smoke` |
| `tiny_avatar4.py` | the studio. Dataset prep, training, webcam avatar driver |
| `splat_trainer3v2.py` | legacy trainer; supplies `Encoder`/`Decoder` and the data cache. Required |
| `spectrum_audit.py` | diagnostic: finds your dataset's sensor-noise knee to pick `--f_max`. Optional |
| `model2.pt` | legacy 96px/256 checkpoint, included so you can play immediately |

## Run order

```bash
python splat_ragdoll.py --model model2.pt              # play first
python splat_trainer5.py --smoke                       # 21 CPU checks, no data
python splat_trainer5.py --compare --data_dir faces1 --steps 3000
python splat_trainer5.py --data_dir faces1 --out runs/hq \
       --image_size 128 --num_packets 512 --detail 1.0 --steps 30000
python splat_trainer5.py --audit runs/hq/model5_constQ.pt
python splat_ragdoll.py --gates --model runs/hq/model5_constQ.pt
python tiny_avatar4.py                                 # webcam driving
```

Budget honestly: 30000 steps × batch 32 ≈ 960k images ≈ **7 hours** at
~38 img/s on a 12 GB 3060, not fifteen minutes. The studio's pulse check
prints the estimate before you commit.

## Driving the avatar (webcam)

Between encoder keyframes, packets glide along the complex-phasor geodesic
rather than crossfading — a linear crossfade drives the field amplitude to
exactly zero at the midpoint (a flat grey frame); phase transport holds
amplitude flat and preserves edges. It passed a registered 8-pair gate.
Modes: **phase** is certified; **lerp** is the baseline it beat; **direct**
re-encodes every frame; **screw** and **dispersion** are uncertified demos
and labelled that way in the driver. Two practical things that matter more
than any of it: frame your live face the way Dataset Prep framed the
training frames (Haar crop, 0.35 margin — the "face-align input" toggle does
it live), and keep beta low.

---

## Honest revisions log

Things that were believed and then measured otherwise. Kept because the
retraction is part of the result.

1. **"High Q causes the moiré."** Falsified by `--audit`: median Q was 0.22 —
   the opposite regime. The trainer's purpose inverted from ceiling to floor.
2. **"Q1 (moiré) is the headline gate."** Demoted: moiré also goes to ~0 for
   maximal blur, since it counts *invented* energy. Only meaningful with the
   PSNR gate holding; Q2 became primary.
3. **"Raw beat_index compares arms."** No — it is confounded by envelope
   size. Arms compare on the overlap-normalised version only.
4. **"A carrier-free gist band gives us the low frequencies."** It opened the
   spectral hole instead. Dropped.
5. **"Equal-per-octave allocation is worth +6.8 dB."** Retracted — the
   trainer already allocated equally; the sweep was against a straw man that
   does not correspond to this code. A smoke test now asserts the equal
   split so nobody "fixes" it back.
6. **"The dyadic tree is the next architecture."** Pre-tested by least
   squares; lost to a flat basis with its own histogram. Not built.
7. **"Lateral inhibition is the missing anti-moiré fix."** There is no moiré
   left to fix (0.0003 vs 0.0132), and a render-time divisive gain is
   cancelled by the optimizer rescaling coefficients. This is a synthesis
   system, not an analysis system; there is no population to normalize.
8. **"~7 MB model."** Imprecise. The *decoder* — the generator — is 7.1 MB
   on the 96px line (12.9 MB at 128px); the renderer has zero learned
   parameters; the shipped training checkpoint is ~96 MB because it also
   carries the encoder and two Adam moment copies of every weight. See the
   size table. Per-frame state remains 128 floats.
9. **"Pull one eye and the head turns with it."** Was a hope; is now a
   measured result — RG1 passed at 9.7× / 6.2× / 4.4× across three models,
   against a control and a negative control. This one survived contact with
   the gate. They don't all die.

## Not certified

Named plainly so nothing here gets read as a result: `screw` and
`dispersion` pursuit modes; the cross-model phase-binding matrix; webcam
mode on any machine but the one it was written on; RG2 on the two models
where it did not converge; ragdoll frame rates on GPU (the app is CPU-only
as shipped); and any claim about datasets other than the ones in the tables.

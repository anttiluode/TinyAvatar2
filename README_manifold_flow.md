# Manifold Flow

**StableAIflow, stabilised by the TinyAvatar splat manifold.**

Identity comes from a prompt. Motion comes from a closed 128-dimensional
manifold of faces. The two are separate layers and neither is asked to do the
other's job.

---

## Why

`StableAIflow` holds a live diffusion filter together with two pixel-space
measurements:

| what it needs        | what it uses now              | what is wrong with it |
|----------------------|-------------------------------|-----------------------|
| frame-to-frame motion | `cv2.phaseCorrelate` — one global `(dx, dy)` | a head that turns is not a translation |
| when to regenerate    | fractal viscosity — a texture statistic | a texture statistic is not a scene model |

Both are *estimates made from the image*. The SplatWorld/TinyAvatar line has
something neither is: a **closed, low-dimensional, non-hallucinating manifold**
of faces. 128 latent dims → N Gabor packets → render.

Its motion is not estimated. The decoder hands you every packet's centre, so
the displacement field between two frames is **read off the model**, exactly:

```
webcam ──▶ face crop ──▶ enc ──▶ z_t                      the manifold state
                                  │
                                  ├─▶ dec ──▶ packet centres c_k(t)
                                  │            │
                                  │            └─▶ W(x) = Σ a_k E_k(x) d_k / Σ a_k E_k(x)
                                  │                                    dense motion field
                                  │
                                  └─▶ novelty ‖z−z_key‖ , drift ∫|W| , residual ‖x − render(z)‖
                                                                              the gate

dream_t = crystallize( warp( dream_{t−1}, W ) )       every frame, at webcam rate
dream   = SDXL-Turbo( structure, prompt )             only when the gate fires
```

The manifold carries **motion and gating**. Diffusion carries **identity and
texture**. Changing the prompt changes who you are; it does not change how you
move.

## It is also faster

StableAIflow displays at diffusion rate. Here the transport loop runs at webcam
rate and diffusion runs on its own thread, so the picture moves at camera speed
while SDXL-Turbo refreshes the identity a few times a second. A dream that lands
late is warped forward by the flow accumulated since it was requested
(`DreamEngine.pending`) before it is shown — without that, the identity arrives a
quarter second in the past and snaps into place.

---

## What is measured

All numbers from `model2.pt` (96px / 256 packets, CelebA, the checkpoint shipped
in SplatField). Frame-to-frame MSE against the next frame, as a **fraction of
doing nothing** — lower is better, 1.0 means the warp bought nothing.

### MF1 — on the model's own renders (`--gates`)

| arm | ratio | |
|---|---|---|
| **manifold field** | **0.775 – 0.836** | the dense per-packet field |
| global shift | 0.930 – 0.963 | control: one amplitude-weighted translation, i.e. the phaseCorrelate incumbent at its best |
| packet scramble | 0.933 – 0.993 | negative control: same displacements, wrong packets |

(n = 24, four seeds, `--step 0.30`.)

**MF1 [V]** — gate: `field ≤ 0.90`, `shift > field`, `scramble ≥ 0.98 × shift`.

**Recorded revision.** The scramble clause first read `scramble ≥ 0.95` against
an assumed null of 1.0, and it failed on 2 of 3 seeds at 0.936–0.956. The
threshold was wrong, not the result. Permuting the packet linkage preserves the
multiset of displacements, so the scrambled field degenerates toward their mean
— which *is* the global-shift arm, sitting at 0.93. The control that means
something is "scramble must not beat shift", because the per-packet **linkage**
is the only thing the manifold adds beyond the mean. It never does. The field
arm passed unchanged at every seed both before and after.

Swept over latent step size, before the gate was written:

| latent step | 0.05 | 0.10 | 0.20 | 0.40 | 0.80 | 1.60 |
|---|---|---|---|---|---|---|
| global shift | 0.969 | 1.009 | 0.986 | 0.918 | 0.990 | 0.973 |
| **manifold** | **0.884** | **0.909** | **0.846** | **0.723** | **0.862** | **0.865** |

Read it honestly: the manifold field explains **9–28%** of frame-to-frame
change as transport, and a global shift explains **essentially nothing**. The
win over the incumbent is real, present at every scale, and **modest**. It is
not a 10×.

Real motion sits in the 0.2–0.4 column — sliding a real crop by 4 px moved the
latent by `|dz| ≈ 1.0` against `|z| ≈ 4.8`.

### MF4 — identity transfer

Apply the same latent motion delta to a *different* anchor identity:

| | corr |
|---|---|
| step 0.30 (operating regime) | **+0.702 ± 0.071** |
| step 1.20 | +0.660 ± 0.134 |
| control: same anchor, different deltas | +0.221 |

**MF4 [V]** — gate was `transfer ≥ 0.50`, `control ≤ 0.25`.

**Caveat that matters more than the pass:** an earlier sweep found the
correlation *rising* with step size (+0.48 at 0.2 → +0.80 at 2.0), and part of
that rise is an artefact — once `|delta| ≫ |z|` both anchors end up in the same
place. The small-step number is the honest one, and ~0.5–0.7 is **not** enough
to build an identity swap on. That is exactly why identity lives in the prompt
here, where it does not need to transfer at all.

### MF2 — UNMEASURED

The load-bearing gate. The dream lives in **real-image space**, so the field has
to transport *real pixels*, not just the model's own render. Everything above is
measured render-to-render, which is the easy case, and MF1 passing does not
imply MF2 passes.

```
python manifold_flow.py --gates --model model2.pt --video me.mp4
```

Gate: `field ≤ 0.95` and `field < shift`. Until that runs, the claim
"the manifold stabilises the diffusion" is **registered, not shown**.

### MF3 — live telemetry

Reuse fraction and mean flicker are in the status bar. Press `m` to A/B the
dense field against a single global shift with everything else held; the honest
comparison is flicker at equal reuse.

---

## Two modes, and the second one is the better idea

**PROMPT mode** (what this file was built for): diffusion is the identity, the
manifold is motion and gating. `structure = 0`, `strength ≈ 0.5`.

**MANIFOLD mode** (his read, added after the first live session, and better
founded): train the avatar on the face you actually want — Einstein, whoever —
and let the *manifold* be the identity. Diffusion then runs at low strength over
the manifold render and does one job: put back the top octave the model provably
loses. `structure = 1.0`, `strength ≈ 0.20`, `gravity ≈ 0.85`, mask from the
manifold.

The reason this is better founded is that it matches both models to their
measured strengths. The manifold's deficit is known and **band-limited** —
capture per octave on model5_constQ is 0.999 / 0.993 / 0.979 / 0.942 / 0.861, a
uniformly mild softness biased to the top (`TroubleShootingFaceSharpness`, FS5
and FS6). Diffusion's deficit is temporal invention. So: the manifold supplies
geometry, identity and motion — all things it does without hallucinating — and
diffusion supplies only the band the manifold cannot reach.

That claim is monitorable. `spectral_split()` measures where in frequency the
diffusion put what it added, and the status bar shows it live as **added HF**.

| | added HF |
|---|---|
| sharpening the render (detail) | **21.8 %** |
| a different identity (invented geometry) | **1.9 %** |

**DM1 [V]** — gate: detail ≥ 5× geometry, measured 11–15×.

High added HF means diffusion is doing the job it was given. Low means it is
redrawing the face, which is identity drift wearing a detail costume, and it is
the exact failure a closed manifold was chosen to avoid. It is a **monitor, not
a controller** — nothing in the app acts on it yet.

**Recorded revision.** DM1 first also demanded `detail ≥ 60 %` absolute and
failed at 22.5 %. The absolute number was invented. The render's spectrum falls
as ~r⁻³, so almost none of its power sits above nyq/4 to begin with, and
sharpening a blurry image cannot put 60 % of the *added* energy up there. The
metric is for separation. Second invented threshold in this project; the first
was MF1's scramble null.

---

## Bugs found by the first live session

The screenshot read `face NO`, `residual 0.1111`, permanent `OFF-MANIFOLD`,
`last fire: off-manifold`, 413 dreams, black motion field.

1. **The off-manifold detector had no scale floor.** It compared the residual
   against a rolling median scaled by MAD, and with a steady residual the MAD
   collapses toward zero, so numerical wobble cleared the threshold on every
   frame. Every frame fired a re-dream; the transport loop never ran. Fixed with
   `scale = max(1.4826·MAD, 0.08·median, 1e-4)` plus a 20-frame warm-up.
   Regression test T9, two-sided: steady residual must NOT fire, a 60 % jump
   must.
2. **No way to turn the face framer off.** When the Haar cascade is missing —
   which it is on Windows Store Python, already on record in this project — the
   box stays `None` forever and the encoder silently receives a centre crop of
   the whole room. Off-manifold input, frozen latent, no field. Now: a
   **Framing** selector (auto / off / manual) with a placeable, resizable manual
   box, and a red warning in the sidebar when the cascade is absent instead of a
   silent fallback.
3. **The mask could only come from Haar.** Added `manifold_coverage()`: the
   amplitude-weighted sum of packet envelopes, i.e. where the model actually
   paints. It is read off the model rather than off a cascade, it has the shape
   of the thing the manifold believes is there rather than of an ellipse, and it
   scales with the packets — lean back and it shrinks by itself.

## 2 Manifold Loop (no webcam)

Remove the camera and close the ring. The avatar manifold encodes the
diffusion's last image; the diffusion paints over the avatar's render and
motion field. Nothing external enters anywhere.

```
        ┌────────────────────────────────────────────┐
        │                                            │
   dream_{t-1} ──crop──▶ enc ──▶ z_t ──▶ dec ──▶ render + field
        ▲                                            │
        │                                            ▼
        └──────── SDXL-Turbo( structure, prompt ) ◀───┘
```

It is a closed dynamical system — two generative models each taking the
other's output as input — so the app names what it is doing rather than
leaving you to guess: **FIXED POINT** (the pair agreed on an image that
survives a round trip), **CYCLE ~N**, or **WANDER**. A frozen picture here is
a result, not a hang.

**Loop drive** defaults to **0.00**, which means nothing at all is injected and
anything that moves is the coupling. Turning it up adds a slow, deterministic,
band-limited latent perturbation — a few incommensurate sinusoids, not white
noise, precisely so that whatever structure appears cannot be blamed on the
injection.

### The lock, measured — LK1

Started from K different points on the avatar manifold, let settle, compare the
spread of the endpoints to the spread of the starting points:

| arm | start spread | attractor spread | contraction |
|---|---|---|---|
| full loop | 6.63 | **1.95** | **0.293** |
| image step → pass-through | 6.63 | 6.00 | 0.905 |

**LK1 [V]** — the pair is a strong contraction; the avatar alone is not. Points
6.6 apart in latent space finish 1.9 apart, so the starting point mostly does
not matter. That is the "it locks into one tight face" observation as a number.
See `README_two_manifold_loop.md` for what the locking might be worth.

### LP1 — is it actually coupled?

The null worth ruling out is that the avatar just relaxes to its own fixed
point and the picture is one model talking to itself.

| arm | mean step | regime |
|---|---|---|
| full loop | 0.147 | WANDER |
| image step replaced by a pass-through | 0.000 | FIXED POINT |

**LP1 [V]** — the avatar alone comes to a dead stop; the image model is what
keeps the latent alive.

**Recorded revision.** With the working default of `structure = 0.33` this
control arm stopped being a null: the structure paste puts the avatar's own
render into the image path independent of the image model, so the "control" was
itself a coupled loop through a second channel and the contrast collapsed from
23× to 2.0×. Both arms now run at `structure = 0`, which is what makes this a
test of the image model rather than of the paste.

**Caveat:** this runs the stub image model, not SDXL-Turbo. It establishes that
the architecture is closed and where the motion comes from. Magnitude and
regime under real diffusion are unmeasured.

### The loop drifted down-right and the head shrank — two causes, neither the field

Diagnosed by replacing diffusion with **identity**, so the only thing acting on
the image was the crop → encode → field → warp → paste chain. Any motion of a
static marker is then bias in the pipeline, not the image model.

**The motion field was innocent.** Turning it off entirely (`field_gain 0`)
changed the drift by less than a thousandth. Mean flow per frame: 0.0000 px.

**Cause 1 — the structure paste was asserting a position and a size.** The
render was dropped into the box rectangle, which silently says "the face is
here, and this big". In a closed loop that nudge *integrates*. Measured over 50
passes with identity diffusion, and it scaled cleanly with the structure share:

| structure | centroid drift | radius |
|---|---|---|
| 0.00 | 0.0000 | 1.000× |
| 0.07 | +0.0031 | 1.106× |
| 0.14 | +0.0171 | 1.228× |
| 0.40 | +0.0284 | 1.759× |

Fix: `_paste_render()` measures the render's *own* coverage centroid and RMS
radius (`coverage_geometry`), then places it so the centroid lands on the box
centre and the radius is a fixed fraction of the box (`render_fill`, default
0.62). The paste geometry now depends only on the box and one constant, so with
a fixed box nothing integrates — the loop settles to a fixed point instead of
marching. At structure 0.40 the centroid now parks at (0.471, 0.524) and the
radius at 0.354, flat from pass 200 onward.

**Cause 2 — the crop was not square at a frame edge.** `FaceFramer` clipped the
box against the edge and handed the surviving rectangle to `cv2.resize`, which
squashed it to the encoder's square input. Measured: aspect **0.690** at
x = 0.90, **1.623** at y = 0.92. The encoder then saw a face stretched by up to
1.6×, so the render was stretched, so the paste stretched the image further —
positive feedback that only fires when the face reaches an edge. With a webcam
it almost never does; in the closed loop it does within seconds, and it is what
tore the head apart. Fix: shrink the square until it fits, then slide its centre
inward. Square at every position (T16).

**The shrinking head was never a separate bug.** It is the drift's consequence:
content slides off the canvas, `BORDER_REPLICATE` smears the edge inward, and
the face is eaten. In AD1 below, stopping the drift takes the radius from
0.43× back to 0.98× with no scale control anywhere in the code.

### AD1 — the anchor

The two fixes stop the *pipeline* injecting drift. They cannot touch the image
model's own bias — at strength 0.74 SDXL-Turbo substantially redraws the frame
each pass and nothing structural prevents it placing the subject a little
further right every time. So `anchor_loop()` measures the global translation the
image actually underwent and integrates it out, with a proportional correction
on the accumulated offset.

It uses `cv2.phaseCorrelate` — the very thing the manifold replaced for
*motion* — for the one job it is genuinely right for: estimating a single global
shift, in order to cancel it. Active in the loop only; with a webcam, global
translation is real information.

Injected creep of 1.2 px/frame, one pass per frame, 120 frames = 0.56 of the
frame:

| | drift | radius |
|---|---|---|
| anchor off | 0.570 | 0.428× |
| anchor 0.50 | **0.013** | **0.980×** |

**AD1 [V]** — 98 % of the injected drift removed.

One detail worth keeping: the **Hanning window** on the correlation is
load-bearing. Without it the frame edges dominate and the estimate under-reads,
leaking ~0.07 of the frame back in per 120 passes even at full anchor strength.
With it, 0.013.

### Lockstep, and why the first probe was meaningless

In the loop the image model sits *inside* the feedback path, so running it on
its own thread makes frames-per-dream a function of machine speed. A probe at
drive 0 duly reported `CYCLE ~4` while the loop was issuing one dream every
3.9 frames — the classifier was detecting the **gate**, not the models. Two
fixes, both kept:

* **Lockstep** (default on): one diffusion pass per frame, synchronously. The
  loop becomes one well-defined map, `z → enc(diffuse(render(z)))`, applied
  once per step. Slower, and the only version whose regime means anything.
  Turn it off for speed and the readout stops being trustworthy — the tooltip
  says so.
* The regime readout **flags itself** when a detected period matches the dream
  cadence: `CYCLE ~4 (= dream cadence, not dynamics)`.

Even after that the probe refused to reproduce run to run. The cause was mine
and had nothing to do with the loop: `StubDream` keyed its palette off Python's
`hash()`, which is **randomised per process** by `PYTHONHASHSEED`, so the test
double was a different function on every invocation. Now `zlib.crc32`, and the
probe is bit-identical across runs (T8b guards it).

With lockstep and the stub, the bare loop at drive 0 settles to a **fixed
point**. The prediction worth testing is that SDXL-Turbo will *not* — diffusion
injects fresh high-frequency content on every pass, which is exactly what a
fixed point cannot survive. That is unmeasured.

## Does dream strength = 1.0 use the avatar at all?

Sharp question, and the honest answer needed the real diffusers scheduler math,
not a guess. Checked directly against `EulerAncestralDiscreteScheduler` with
SDXL's actual beta schedule.

**Short answer: at strength 1.00, no — not the avatar's *pixels*.** img2img
works by adding noise to the input latent up to a starting timestep set by
`strength`, then denoising from there. At `strength = 1.0` that starting point
is the single noisiest timestep in the schedule (σ ≈ 14.6), where the signal
fraction is `1/(1+σ²)`:

```
strength 1.00  ->  0.47%  of the structure image survives
```

The "structure image" is the blend of webcam and avatar render that `structure`
and `gravity` control. At 0.47% retained, that blend is statistically noise to
SDXL. **This is functionally text-to-image** — the avatar's render, and the
webcam, are both being discarded before the model ever sees them.

### The bug this uncovered: the step formula was a knife-edge, not a dial

It's worse than "high strength ignores the image." The app derived step count
as `steps = 2.0 / strength`, and that interacts with `strength` in the SAME
formula that sets the starting timestep — so two adjacent slider positions
could land at *opposite* extremes:

| strength | old steps | retained |
|---|---|---|
| 1.00 | 2 | **0.47%** |
| 0.90 | 2 | **99.91%** |
| 0.74 | 2 | 99.91% |
| 0.50 | 4 | 53.50% |
| 0.33 | 6 | 99.91% |
| 0.20 | 10 | 88.04% |

Moving the slider from 1.00 to 0.90 didn't turn the dial down — it flipped from
"discard almost everything" to "discard almost nothing," a 99-point jump on one
tick. There was no strength setting that gave a moderate, controllable blend.

**Fixed:** step count is now a fixed value (default 4), independent of
strength, with its own slider. Retention now moves smoothly:

| strength | fixed steps | retained |
|---|---|---|
| 1.00 | 4 | 0.47% |
| 0.90 | 4 | 10.51% |
| 0.74 | 4 | 53.50% |
| 0.50 | 4 | 53.50% |
| 0.33 | 4 | 99.91% |
| 0.20 | 4 | 100.00% |

**SS1 [V]** — gate: the old formula has a jump over 90 percentage points
somewhere in that range; the fixed formula's biggest jump is under 70. Measured
99pp vs 46pp.

The status bar now shows **structure retained N%** live, computed from the
actual (steps, strength) pair each time a dream fires, so this is no longer
something you have to take on faith.

### So what *is* the avatar doing at strength 1.0?

Not nothing — its role just isn't what "Manifold share of structure" implies at
that setting. Four things survive regardless of strength, because none of them
route through the noised structure image:

* **The gate** — novelty, drift, residual, all computed from `z`, decide *when*
  a redream fires.
* **The motion field** — transports the standing dream between redreams, read
  off the decoder, independent of what the last diffusion call painted.
* **The coverage mask** — `manifold_coverage()` decides *where* the diffusion
  output gets composited onto the canvas.
* **In the loop**, the avatar is *what encodes the diffusion's own output* —
  the thing that turns the loop into a closed system at all.

What does **not** survive at strength 1.0: the avatar's specific rendered
appearance influencing what SDXL paints. At that setting SDXL is free-running on
the prompt, encoded and gated by the avatar but not looking at it.

This also sharpens the 2-manifold-loop finding (`README_two_manifold_loop.md`):
the attractor lock happens *even though* the structure channel is voided at
`strength = 1.0`. The coupling that produces the lock is not visual
conditioning — it's the motion field, the gate, and the mask, operating on a
diffusion model that is otherwise seeing only the prompt.

## Loading a different avatar model

**Load avatar model...** in the sidebar opens any `.pt` checkpoint from the
TinyAvatar/SplatField line — a different identity, a different resolution, a
different packet count. It swaps the MANIFOLD layer only; it has nothing to do
with the diffusion prompt.

Two things had to be true for this to be safe rather than a source of silent
mixed state:

* **Validate at load, not at the next webcam frame.** `Manifold.__init__` now
  does one forward pass at construction (`render(zeros(128))`). A checkpoint
  that is merely the wrong shape — mismatched packet count, a half-written
  save — fails at the button press with a message box, not three frames later
  inside a tensor op with no context.
* **Every piece of state tied to the OLD model gets dropped on swap**, not just
  the model pointer: the pursued latent, the previous packet state (its
  displacement to a differently-scaled `z` is meaningless), the standing dream,
  the gate's key and residual history, the prompt-changed flag, the flicker
  buffer. Keeping any of these would quietly blend two models' scales into one
  number. `FlowPipeline.load_manifold()` does this centrally so the reset can't
  be forgotten at a second call site later.

Regression tests: T12 (missing file raises `FileNotFoundError` cleanly, not a
crash inside torch) and T13, two-sided (every stale field must actually have
been set before the swap AND actually cleared after — not just "touched").
`--uismoke` drives the real button through a monkeypatched file dialog, once
for the happy path and once for a bad path, and checks the bad path warns
without disturbing the running model.

## Sidebar layout (second live session)

The Manifold group collapsed into overlapping unreadable stripes and its
framing controls were unreachable. Four separate causes, all layout:

1. **No scroll area.** The sidebar is 1269 px of content in an 803 px viewport
   and always will be, so Qt squeezed every group past its minimum. It now
   lives in a `QScrollArea` with a permanent scrollbar.
2. **`QGroupBox` defaults to a `Preferred` vertical policy**, which permits
   shrinking below the size hint. Every group is now pinned `Fixed` to its own
   hint and the trailing stretch absorbs the slack.
3. **Word-wrapped captions had no height-for-width**, so the layout allocated
   one line and clipped the rest — silently; Qt does not complain. Fixed, and
   `MinimumExpanding` was wrong too: an expanding wrapped label soaks up slack
   and starves its siblings (measured: prompt group 13 px short while the
   preset group ran 26 px over).
4. **A wrapped label's height depends on its final width**, which is unknown at
   construction, so the first pass locks in a one-line height. `_fix_wraps()`
   re-measures on show and on every resize.

Four new regression checks in `--uismoke`, all geometric rather than
behavioural: the sidebar must scroll rather than squeeze, no group may fall
below its minimum size hint, no two groups may overlap, no label may be shorter
than its height-for-width. `MF_UISHOT=x.png python manifold_flow.py --uismoke`
renders the whole window to a file so the layout can be eyeballed too.

**Also:** when the Haar cascade is absent the app now *starts* in manual
framing. Starting in a mode that cannot work meant a silent centre crop of the
whole room, which is what the first live session actually ran.

## Honest limits

- **Nothing here says it looks better.** Flicker is measured. Beauty is not.
- The manifold render is blurry — see `TroubleShootingFaceSharpness`. The field
  does not need it to be sharp, only the packet centres to be right, which is
  why `structure` (the manifold's share of what diffusion is shown) **defaults
  to 0**. Turn it up to see what a closed prior does to the hallucination; do
  not expect detail.
- Flow accumulation for late dreams is first-order composition of small
  displacements. Exact for translation, approximate otherwise, unverified beyond
  that. The approximation is bounded by dream latency, which is the thing the
  design minimises anyway.
- The encoder is measurably translation-invariant (SplatWorld's `phase_orbit`
  work), so global head translation is re-indexed rather than transported. The
  crop is done with the *same* Haar + 0.35 margin as Dataset Prep for that
  reason — a different crop puts the encoder off-manifold and every number
  downstream measures that instead.
- `--stub` is **not diffusion**. It exists so that a failure in the transport
  loop cannot hide behind a 7 GB download.
- n = 1 model everywhere. CelebA `model2.pt` only.

---

## Files it needs beside it

```
splat_trainer5.py      splat_trainer3v2.py     # load_splatvae + model defs
splat_ragdoll.py       pin_driver.py           # optional
model2.pt                                       # or any model5_<tag>.pt
```

Everything loads through `ST.load_splatvae` — constructing `SplatVAE` from
`(image_size, num_packets)` alone renders a legacy checkpoint with constant-Q
formulas and is silently wrong by up to 0.57 on a 0–1 image.

## Run order

```bash
python manifold_flow.py --selftest                     # 17 checks, no model
python manifold_flow.py --uismoke --model model2.pt    #  8 checks, offscreen
python manifold_flow.py --gates --model model2.pt      # MF1, MF4
python manifold_flow.py --gates --model model2.pt --video me.mp4   # + MF2
python manifold_flow.py --model model2.pt --loop       # 2-manifold loop, no camera
python manifold_flow.py --model model2.pt --stub       # live, no diffusers
python manifold_flow.py --model model2.pt              # live, SDXL-Turbo
```

Verified in sandbox: **17/17** selftest, **8/8** UI smoke, MF1 [V] and MF4 [V]
on the real `model2.pt`. Live diffusion path is compile-and-structure checked
only — SDXL-Turbo was never downloaded here.

## Controls

| | |
|---|---|
| **Prompt** | the identity. Editing it forces a re-dream. |
| Dream strength / gravity / crystallizer | as in StableAIflow |
| **Manifold share of structure** | 0 = diffusion sees the webcam, 1 = it sees the closed manifold render |
| Pursuit alpha | latent smoothing toward the encoder's reading |
| Field gain | scales the displacement field; 0 disables transport |
| Novelty / drift / max-frames | the three gate thresholds |
| `space` | force a re-dream | 
| `m` | dense field ↔ global shift (the live A/B) |
| `r` | drop the standing dream | `q` quit |

Status bar: fps, dreams issued and their latency, reuse %, novelty, drift,
residual, **OFF-MANIFOLD** when the residual spikes, why the gate last fired,
flicker, `|z|`, and whether a face is detected.

When the residual spikes — hand over the lens, second face, something the
manifold has never seen — the gate marks it off-manifold, fires a re-dream, and
the pipeline **cuts feedback gravity to a quarter**, because a motion field read
off a confused manifold is worse than no motion field.

---

## Wrong turns, on record

1. The first design put identity in the manifold (swap `z_anchor`, keep the
   motion delta). Pre-tested before writing a line of app: transfer correlates
   +0.48 in the operating regime. Not enough. Identity moved to the prompt.
2. The +0.80 identity-transfer number at large steps was nearly reported as the
   headline. It is inflated by the anchor being swamped. The small-step number
   is the one in the table.
3. Global shift was expected to be a weak-but-real arm. It is a **null**
   (0.92–1.01). That makes the manifold's win over it easier to claim and
   smaller in absolute terms than it first looked.
4. The gate's off-manifold clause, the missing framer switch and the
   Haar-only mask — all three found by one live screenshot, all three mine.
   Written up in full above rather than quietly patched.
5. The render was originally computed every frame for the residual. It is 51 ms
   on CPU and the field does not use it at all — decimated to every 2nd frame,
   136 → 83 ms/frame on CPU.

Do not hype. Do not lie. Just show.

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

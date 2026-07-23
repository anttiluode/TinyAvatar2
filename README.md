# TinyAvatar 2

A ~7 MB generative face model built out of nothing but wave interference,
plus the studio that trains and drives it.

A VAE maps a 128-dimensional latent to a few hundred **Gabor wave packets**.
The image is their additive interference on the canvas. No pixels are stored
and there are no convolutions in the decoder — the face *is* the interference
pattern.

TinyAvatar 1 shipped that pipeline. TinyAvatar 2 changes the **basis** it is
built from, and fixes three bugs in the app that were quietly corrupting the
results.

> Do not hype. Do not lie. Just show.

---

## What is actually new

### The old models had abandoned their own carrier

A Gabor packet is an oriented sinusoid under a Gaussian envelope. How much of
it is sinusoid is captured by

```
Q = sigma * freq        # cycles of carrier across one envelope sigma
```

`splat_trainer3v2` sampled `sigma` and `freq` independently, so the model was
free to pick any Q. Measuring what it actually picked, on the shipped
`model2.pt`:

```
Q percentiles 0/10/25/50/75/90/99/100:
  0.04  0.10  0.14  0.22  0.31  0.41  0.66  1.10
fraction with Q > 1.5 : 0.000
sigma median 0.067    freq median 3.24    (pixel Nyquist 48)
```

Median Q of **0.22** is under half a cycle across the whole visible envelope.
The trained basis was a mixture of signed Gaussian **blobs**, not a Gabor
frame. That is the blur: one Gabor can represent an edge; one blob cannot.

This falsified the hypothesis the trainer was originally written to test —
that moiré came from *too much* Q. The measurement said the opposite, so the
constant-Q coupling was inverted from a ceiling into a **floor**:

```
sigma = clamp( (q / freq) * exp(q_slack * tanh(raw)), sig_lo, sig_hi )
```

With `q = 0.6` and one octave of slack this confines Q to roughly `[0.3, 1.2]`
— the blob collapse is forbidden and every packet must oscillate.

### The result, in a matched comparison

96px / 256 packets, 1937 own-face images, 3000 steps per arm, everything else
identical:

| gate | | constant-Q | legacy |
|---|---|---|---|
| **Q2 PRIMARY** | PSNR, eval mode | **14.39** | 12.70 |
| | PSNR, batch statistics | **14.38** | 12.70 |
| Q1 | moiré index | **0.0003** | 0.0132 |
| Q3 | beat, overlap-normalised | **0.4217** | 0.6447 |
| Q4 | median Q | **0.49** | 0.42 |

All four passed. Forcing carrier use bought **+1.7 dB** and cut invented
mid-band structure **44-fold**. PSNR is measured in both BatchNorm modes
because the encoder's running statistics are poorly estimated at 3000 steps
and can fake a gap on their own; the two modes agree to 0.01 dB, so this is
a reconstruction result and not a BatchNorm artefact.

**Honest scope:** one dataset, one identity. The claim this supports is
"constant-Q beat legacy here", not "constant-Q is better". A CelebA rerun
costs about six minutes and would make it much harder to argue with.

---

## The octave ladder

Constant-Q has a consequence: `sigma <= sig_hi` forces `freq >= q/sig_hi`. So
the sigma ceiling sets a **carrier floor**, and if that floor lands above the
face-gestalt band you get no face.

That is exactly what happened. An earlier build reserved a fraction of
packets as carrier-free Gaussians at `freq = 0` while carriers started at
5.0 cycles/image — so **nothing at all** represented `(0, 5)`, which on a face
is the head outline (1–2 cyc/img) and the feature layout (3–5). It rendered
as frosted glass plus fine stripes with no gestalt. That is the spectral-hole
bug, and the studio now draws the ladder live and turns red if you recreate it.

The fix was to drop the separate gist band and run one continuous constant-Q
family from `f = 1` to Nyquist/2. At 128px / 512 packets the ladder is:

```
   1.00 -    2.00 cyc/img   103 packets   sigma 0.300-0.600
   2.00 -    4.00 cyc/img   102 packets   sigma 0.150-0.300
   4.00 -    8.00 cyc/img   103 packets   sigma 0.075-0.150
   8.00 -   16.00 cyc/img   102 packets   sigma 0.037-0.075
  16.00 -   32.00 cyc/img   102 packets   sigma 0.019-0.037
```

### A retraction, kept on the record

It was claimed during development — by me, in this project's own notes, and
then amplified elsewhere — that the trainer allocated packets proportional to
`4^level` and that switching to equal-per-octave was worth **+6.8 dB**.

**That was wrong.** The band assignment

```python
band = floor((k - n_gist) * octaves / n_car)
```

already distributes the budget equally: 103/102/103/102/102, as above. The
least-squares sweep that produced the +6.8 dB figure was measured against a
straw-man arm that does not correspond to this code. Equal-per-octave is the
arm that *won* that sweep. So the sweep **validates** the design rather than
fixing it, there is no free 6.8 dB, and the allocation is untouched. A smoke
test now asserts the equal split so nobody "fixes" it back.

### `band_mode` — how octaves map onto the anchor lattice

`anchor_logit` places packet *k* at raster-scan cell *k* of a regular
`ceil(sqrt(N))` lattice, and the octave was assigned by
`floor(k * octaves / n_car)`. Both are functions of the same *k*, so band and
image region were coupled. At 512 packets / 5 octaves / a 23×23 grid, each
band was confined to an 18–23% vertical strip — the f = 1–2 gestalt band to
`x ∈ [0.08, 0.23]`, the f = 16–32 band to `x ∈ [0.73, 0.92]`.

Measured cost: **+0.14 dB, sd 0.35** across four seeds. That is inside noise,
and the reason is instructive — coarse packets have σ = 0.3–0.6, so their
envelopes blanket the frame no matter where the anchor sits. A shuffle control
matched interleaving to 0.03 dB, confirming the mechanism is spatial coverage
rather than ordering. The fix is taken because the prior is wrong on its face
and costs nothing, **not** because it buys anything. It is not a result.

| mode | per-band anchor span | note |
|---|---|---|
| `striped` | 18–23% | pre-2026-07 behaviour; the default for any checkpoint without the key |
| `interleave` | 100% | `k % octaves` — full coverage, but still a *regular* sublattice |
| `permute` | 95–100% | fixed-seed random assignment, same coverage, no periodicity — **default for new runs** |

**`band_mode` is a recorded checkpoint key, and that is the whole point.**
Changing it changes every packet's frequency *range*, while the `state_dict`
shape stays identical — so a striped-trained model rendered by interleaved
code produces **0.98 max absolute error on a 0–1 image**, silently. Worse than
the driver bug above. A checkpoint written before this key existed loads as
`striped` and renders bit-identically (0.000000), so nothing already trained
is disturbed.

---

## The dyadic tree that did not work

The natural next idea is a bifurcating quadtree: parents spawn four children
at `f_child = 2 f_parent`, `sigma_child = sigma_parent / 2`, which preserves Q
exactly, with children pinned inside the parent envelope so high frequencies
cannot beat across the whole face.

Because the geometry is fixed, coefficient fitting is a **linear
least-squares problem**, which is an upper bound on what any trainer could
reach with that geometry — computable in seconds. Matched N = 341, four
seeds, synthetic multiscale face:

| basis | lsq PSNR | beat, normalised |
|---|---|---|
| tree, uniform subdivision | 23.44 ± 0.67 | 0.724 |
| tree, residual-driven splitting | 21.68 ± 0.57 | 0.732 |
| **flat basis, same octave histogram** | **22.55 ± 0.24** | 0.725 |

The adaptive tree is **0.87 dB worse** than a flat basis that merely copies
its histogram, and beat did not drop. Pinning children inside parents does not
kill fringes; it relocates them among siblings. Everything the tree bought was
the histogram.

What this does *not* rule out: the tree might still help the **amortized**
problem — a VAE predicting N packets from z may find structured tree output
easier than a free list. That is untested. The tree's motion claim (rigid
SE(2) inheritance under head turn) is also untested, and sits badly with the
already-measured translation-invariance of the conv encoder.

Naming note: "Feigenbaum tree" conflates two unrelated senses of
*bifurcation*. Feigenbaum's δ = 4.669 is period-doubling in iterated maps; a
quadtree is anatomical branching. And the closed splat field cannot
period-double at all — its Jacobian is similar to a symmetric PSD operator at
every state, so the spectrum is floored at `leak` everywhere and no eigenvalue
can reach −1. The same positivity that forbids autonomous rotation forbids
chaos.

---

## Three bugs this release fixes

**1. Every long run crashed at the finish line.** `evaluate()` returns two
dicts (eval-mode and batch-statistic), but the single-arm path still unpacked
four values. `ValueError` *after* training completed — the checkpoint was
saved, the report was lost. Seven hours to find out.

**2. The Avatar Driver rendered models with the wrong formulas.** It built

```python
model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
```

which ignores the checkpoint's parameterisation and applies whatever the
renderer defaults are. A constant-Q `state_dict` has the *same shape* as a
legacy one, so this raises no error — it just produces wrong pixels. Measured:

| checkpoint | max absolute error, 0–1 image |
|---|---|
| legacy (`model2.pt`, no `qmode` key) | **0.572** |
| constant-Q with non-default q/octaves/f_max | **0.524** |

Over half the dynamic range, silently. Every load now goes through
`load_splatvae()`, and the driver prints which parameterisation it found.

**3. The progress bar never moved and Resume never armed.** The log parser
required `(PSNR x) kl y`, which the trainer never printed; and Resume scanned
for `model2.pt` while the trainer wrote `model4q_<tag>.pt`, and `--resume` was
not a declared flag so the guard dropped it anyway. All three fixed — and any
flag the app sends is now asserted against the trainer's declared set, so a
control that does nothing refuses to launch instead of pretending.

---

## Files

| file | what it is |
|---|---|
| `splat_trainer5.py` | the trainer. Constant-Q renderer, Q1–Q4 gates, `--audit`, `--compare`, `--resume`, CPU `--smoke` |
| `tiny_avatar4.py` | the studio. Dataset prep, training, avatar driver |
| `spectrum_audit.py` | measures your dataset's radial power spectrum and finds the sensor noise floor |
| `splat_trainer3v2.py` | **required, not included here** — lives in the TinyAvatar repo. Supplies `Encoder`, `Decoder`, `build_cache`, `load_resident`, `batch_from` |

## Run order

```bash
python splat_trainer5.py --smoke                     # 21 CPU checks, no data
python spectrum_audit.py --selftest                  # 5 checks
python spectrum_audit.py --data_dir faces1 --image_size 128
python splat_trainer5.py --compare --data_dir faces1 --steps 3000
python splat_trainer5.py --data_dir faces1 --out runs/hq \
       --image_size 128 --num_packets 512 --detail 1.0 --steps 30000
python splat_trainer5.py --audit runs/hq/model5_constQ.pt
python tiny_avatar4.py
```

Budget honestly: 30000 steps × batch 32 = 960k images. At ~38 img/s that is
about **7 hours**, not fifteen minutes. The studio's pulse check now prints
this estimate before you commit.

### About `--f_max`

`f_max` defaults to half the pixel Nyquist and **nobody has measured whether
that is right for your data**. If your frames have a sensor noise floor below
it, the top octave is spending 20% of the packet budget fitting grain — and
packets fitting independent noise are exactly the ones that beat against each
other. `spectrum_audit.py` finds the knee.

A note on how it finds it, because the obvious method fails. Fitting a power
law to a "structured band" and flagging where the measurement rises above the
extrapolation does not work: you do not know where the structured band ends —
that is the thing being measured — so the fit absorbs the flattening and
chases the floor. On a knee planted at r = 18 that detector fired at r = 59,
and iterating the fit did not help. The shipped detector uses the **local
log-log slope** instead, which has no such circularity: a power-law band has
slope ≈ −α, a flat floor has slope ≈ 0. Selftest recovers planted knees to
19% and 0%, and reports no knee on clean data.

The audit tells you which `f_max` to **try**. A matched `--compare` decides it.

---

## Driving the avatar

Between encoder keyframes, packets glide along the complex-phasor geodesic
rather than crossfading. A linear crossfade drives the field amplitude to
exactly zero at the midpoint — a flat grey frame, all structure gone. Phase
transport holds amplitude flat, preserves edges, and travels about twice as
far across image space before returning. It passed a registered 8-pair gate
(amplitude discipline, sharper-than-road mid-frames, scramble control broke
10×).

Modes in the driver: **phase** is the certified one. **lerp** is the baseline
it beat. **direct** re-encodes every frame. **screw** (SE(2) rigid transport)
and **dispersion** (closed-form phase advance) are demos — they run, they are
not certified, and the driver labels them that way.

Two practical things that matter more than any of it:

- **Frame your live face the way Dataset Prep framed the training frames.**
  Prep face-crops with a Haar detector and a 0.35 margin; a plain centre crop
  gives the encoder a smaller, wandering face, which is off-manifold and
  renders as a blurry average head. The "face-align input" toggle applies the
  same crop live. Background differences do not matter; framing does.
- **Keep beta low.** High beta posterior-collapses the model to grey, or to a
  single face.

---

## Honest revisions log

Things that were believed and then measured otherwise. Kept because the
retraction is part of the result.

1. **High-Q causes the moiré.** Falsified by `--audit`: median Q was 0.22, the
   opposite regime. The trainer's purpose inverted from a Q ceiling to a Q floor.
2. **Q1 (moiré) is the headline gate.** Demoted. `moire_index` also goes to
   ~0 for a maximally blurred output, since it counts *invented* energy and
   blur invents none. It is only meaningful with the PSNR gate holding, so
   Q2 became primary.
3. **`beat_index` compares arms.** Not as raw. It is confounded by envelope
   size — large sigma overlaps everything, so raw beat rises whenever sigma
   rises regardless of fringes. Arms are compared on the overlap-normalised
   version; the raw number is retained and explicitly flagged.
4. **A separate carrier-free gist band gives us the low frequencies.** It
   opened the spectral hole instead. Dropped.
5. **Equal-per-octave allocation is worth +6.8 dB.** Retracted — the trainer
   already did it. See above.
6. **The dyadic tree is the next architecture.** Pre-tested by least squares
   before spending GPU time; it lost to a flat basis with the same histogram.
   Not built.
7. **Lateral inhibition is the biological anti-moiré fix we were missing.**
   Two problems. There is no moiré left — the constant-Q arm measures 0.0003
   against legacy's 0.0132 — and a render-time divisive gain is cancelled by
   the optimizer: at γ = 1 the mean gain falls to 0.22 while achievable PSNR
   drops 0.24 dB, because the raw coefficients simply scale up. V1 is an
   *analysis* system where coefficients are responses to a world with huge
   dynamic range; this is a *synthesis* system where they are commands. There
   is no population to normalize. (Biology's actual anti-aliasing is the eye's
   optical low-pass plus the irregular cone mosaic — Yellott 1983 — not
   inhibition.) The version still worth trying is a training *loss*,
   `L = Σ W_kl A_k A_l` with a **log-frequency** window, aimed at the measured
   diversity collapse rather than at moiré.
8. **Vino's skew operator closes the loop's open frequency problem.**
   Refuted: composing a skew operator with the symmetric-PSD render Gram
   pulls the composite spectrum back to the real axis, so skew barely rotates
   in this medium. Chasing that failure found the actual answer — the
   saturated frequency is drive-invariant and set at criticality, not at the
   operating point.

## Not certified

Named plainly so nothing here gets read as a result: `screw` and `dispersion`
pursuit modes; the cross-model phase-binding matrix; webcam mode on any
machine but the one it was written on; the pulse check's VRAM heuristics on
cards other than a 12 GB 3060.

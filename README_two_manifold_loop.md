# The 2 Manifold Loop

*A morning's experiments, and what manifold locking might be worth.*

Remove the camera and close the ring between two generative models:

```
        ┌──────────────────────────────────────────────┐
        │                                              │
   image_{t-1} ──crop──▶ enc ──▶ z_t ──▶ dec ──▶ render + motion field
        ▲                                              │
        │                                              ▼
        └────────── SDXL-Turbo( structure, prompt ) ◀───┘
```

The avatar (a 128-D Gabor-splat VAE trained on CelebA, 512 packets, 128 px)
encodes the diffusion model's last image. The diffusion model paints over the
avatar's render and its motion field. Nothing external enters anywhere. Then
watch.

---

## What happened

It locked. Hard. Not to noise, not to a mush — to a **specific, narrow,
repeatable face**: a chiselled male, roughly 40 to 60, the visual register of a
CEO headshot at a press event. Different seeds, same destination.

The prompt that made it stable was found by hand:

> *high res photo of famous persons face, only one person, in focus at press
> event, greenscreen, face centered*

That prompt is now the default, along with the rest of the working
configuration.

**The prompt is not a style request here — it is a boundary condition on the
attractor.** "only one person" forbids the loop from splitting into two faces.
"face centered" forbids the translational drift. "greenscreen" removes the
background as a degree of freedom, so the loop cannot spend its energy inventing
rooms. Every clause is a constraint on the fixed point rather than a
description of a picture. Once those constraints were in, the loop stabilised
even on the build that had **no** anti-drift machinery in it at all — which is
worth noticing, because a well-chosen prompt did the same job as a controller.

## The lock, as a number instead of an impression

`--gates` runs **LK1**: start the loop from K different random points on the
avatar manifold, let it settle, and measure how far apart the endpoints are
compared with how far apart the starting points were.

| arm | start spread | attractor spread | contraction |
|---|---|---|---|
| full loop | 6.63 | **1.95** | **0.293** |
| image step → pass-through | 6.63 | 6.00 | 0.905 |

**LK1 [V]** — the full loop is a strong contraction; the avatar alone is not.
Starting points 6.6 apart in latent space finish 1.9 apart. **The starting
point mostly does not matter**, which is precisely the "it locks into a very
tight male face loop" observation, measured.

The companion gate **LP1 [V]** rules out the boring explanation. With the image
step replaced by a pass-through the latent goes to a dead stop (mean step
0.0000, FIXED POINT). So this is not the avatar relaxing to its own fixed point
with a picture along for the ride — the image model is what keeps the state
alive, and the pair is what selects where it lands.

Both figures are `n` in the single digits and run against the **stub** image
model, not SDXL-Turbo. The numbers that matter are the same two gates under
real diffusion with the prompt above. The machinery is in place; the run is
his.

## Why that particular face

Two documented skews multiplied together.

CelebA is roughly 10k celebrity identities scraped in the early 2010s, heavily
weighted toward light-skinned, young-to-middle-aged, professionally photographed
faces. And "famous person's face at a press event" pulls SDXL toward exactly the
studio-lit male actor-or-executive headshot that dominates press photography.

The attractor is where those two mass concentrations overlap. **That is what the
loop is computing.** It is not finding a face; it is finding the *intersection
of two priors*, and reporting it as an image.

Which turns out to be the most interesting thing about it.

---

## So what is manifold locking good for?

Four answers, ordered by what they cost.

### 1. Measuring a composed prior — cheap, and real today

Prompt a model, look at the output, form an impression: that is how model bias
is usually assessed, and it is eyeballing. The loop does something different.
It **integrates** — hundreds of passes, each one a small vote, until the state
arrives where the composition's mass actually is. The attractor is a *sample
from the joint mode*, produced by the models rather than by a human choosing
what to look at.

That makes several things measurable that were previously anecdotal:

* **Where is the joint mode?** Run from K seeds, cluster the attractors. One
  cluster or several? LK1 already gives the contraction figure.
* **Whose prior wins?** The **Load avatar model** button swaps the avatar
  without touching anything else. CelebA → his own face → some other checkpoint.
  If the attractor barely moves, SDXL dominates. If it tracks the avatar, the
  small closed model is steering the big open one. This is a two-line experiment
  with an already-built control and it has not been run.
* **Which prompt clauses are load-bearing?** Drop one clause, re-measure the
  contraction and the attractor. "only one person" and "face centered" are
  testable as regularisers, not as vibes.

This is the use worth his tokens. It needs no training, no new architecture,
and the tooling exists.

### 2. Off-manifold distance as a free, label-free signal

`residual = ‖crop − render(z)‖` is already on screen. It measures how far the
diffusion output lies off the closed avatar manifold, and it needs no labels, no
classifier, and no reference image. In the loop it settled around 0.05–0.06.

A cheap, closed model that can say *how far off the face manifold this image
is* is useful well beyond this app — as a filter, a reward term, a drift alarm.
And note it is a **projection**, not a representation: the manifold does not
have to be able to *build* the world, only to measure distance to a slice of it.

### 3. The world-model idea — right shape, wrong scale, and the loop is evidence against

The intuition is sound: a closed, low-dimensional, non-hallucinating state space
that a large generative model must pass through every step is the correct
*shape* for grounding. It is what stops a model from drifting off into
plausible-looking nonsense, because the state space contains no nonsense to
drift into.

Then the honest problems.

**This manifold covers faces, at one scale, in one framing.** A grounding
manifold needs objects, composition, occlusion, and persistence. Getting from
here to there is not more work in the same direction — it is a different
research programme, and correctly sensed as not worth starting on a token
budget.

**But the deeper objection is the one this morning actually produced.** A
grounding manifold must be expressive enough to hold everything true and nothing
false. What the loop demonstrates is the opposite failure: the manifold is
narrow, so the *narrowness ate the content*. Every seed collapsed to one CEO. If
you wired a real system's world state through a manifold with this
characteristic, you would not get grounding — you would get every situation
being reported as the manifold's favourite situation.

**The uniform CEO face is what mode-collapsed grounding looks like.** So this
experiment is evidence about the difficulty of the world-model programme rather
than a step along it, and that is a genuinely useful thing to have learned in a
morning.

The reframe that survives: not *manifold as world model* but **manifold as
projection operator** — a cheap closed thing you project a big model's output
onto, to measure or bound how far off it went. Projection is a far smaller ask
than representation, it does not require the manifold to be complete, and it is
already working.

### 4. As an instrument, not a product

The loop is a slow, closed, deterministic map, `z → enc(diffuse(render(z)))`.
That is a dynamical system, and it is the same object as everything else in this
ecosystem — fixed points, limit cycles, contraction rates, basins. The regime
readout already names FIXED POINT / CYCLE ~N / WANDER. Sweeping the loop drive
and watching where the regime changes is a bifurcation diagram of two neural
networks talking to each other, and it costs one afternoon.

Whether that produces anything beyond a pretty picture is unknown. It is
cheap enough to find out.

---

## What is not established

* Every loop figure here is the **stub** image model, not SDXL-Turbo. The stub
  exists so that a broken transport loop cannot hide behind a model download;
  it is not diffusion and its attractor is not SDXL's.
* `n` is 6–12 seeds, one avatar checkpoint, one prompt. The "it always lands on
  the same face" claim is an impression from a live session plus one contraction
  number, not a survey.
* Nothing here says the loop is *useful*. It says it is measurable.
* The attractor may be an artefact of the gate cadence, the lockstep setting, or
  the structure share rather than of the two models. Lockstep and the
  cadence-flagged regime readout exist to reduce that risk; they do not
  eliminate it.
* All four "what is it good for" answers above are arguments, not results. The
  only measured claims on this page are LK1 and LP1.

## Three measurement confounds caught this morning

All three were in my own instruments, not in the models. Recorded because a
gate that passes for the wrong reason is worse than one that fails.

1. **The regime classifier was reading the gate, not the models.** A drive-0
   probe reported `CYCLE ~4` while the loop was issuing one dream every 3.9
   frames. Threaded diffusion inside a feedback path makes frames-per-dream a
   function of machine speed. Fixed by Lockstep (one synchronous pass per frame,
   so the loop is one well-defined map) and by making the readout flag itself
   when a period matches the cadence.
2. **The stub's `strength` had the wrong sign.** It blended *toward* the input
   as strength rose, opposite to the diffusers convention. Invisible until
   `strength = 1.00` became the default — at which point the stub silently
   became the identity function and nulled three gates without any of them
   failing. A test double with a wrong sign is worse than no test double.
3. **LP1's control arm was not a null.** With the working `structure = 0.33`,
   the pass-through arm still had the avatar's own render entering the image
   path through the structure paste, so the "control" was a coupled loop through
   a second channel. Contrast collapsed from 23× to 2.0×. Closing that channel
   is what makes LP1 a test of the image model: it now reads 0.0000 vs 0.1468.

Do not hype. Do not lie. Just show.

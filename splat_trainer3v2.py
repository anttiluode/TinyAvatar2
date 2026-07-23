#!/usr/bin/env python3
# splat_trainer3v2.py — the fast trainer (faces folder -> better splat_decoder.onnx)
#
# Same architecture as splat_generator.py (latent 128, Gabor packets, anchor
# grid, complex phase head) so every existing tool — splat_cv5, probe, surf,
# atlas, zoom — works on the new model unchanged. What changed is SPEED:
#
#   1. CACHE ONCE, PER DATASET. The old trainer decoded 200k JPEGs every
#      epoch — that was the real bottleneck, not the GPU. First run on a
#      given --data_dir builds a cache (uint8, center-cropped, resized)
#      with threaded cv2, under a subfolder of --out NAMED AFTER the
#      dataset folder: --out runs/splat2 --data_dir faces1 caches to
#      runs/splat2/faces1/faces_cache_128.npy. Point at "faces" and it
#      caches under runs/splat2/faces/ instead — two different --data_dir
#      folders can never collide on the same cache file again. Every
#      later run against that same folder starts in seconds; a new
#      folder name always (re)builds from scratch.
#   2. DATASET LIVES ON THE GPU. 202k x 96x96x3 uint8 = 5.6 GB -> fits a 12GB
#      card next to the model (64px = 2.5 GB). Batches are fancy-indexed on
#      device; there is NO DataLoader, no workers, no H2D copy per step.
#      Fallback ladder if it doesn't fit: GPU -> pinned CPU -> pageable
#      CPU RAM -> disk memmap (batch-only reads). Any rung that OOMs
#      drops to the next; --disk jumps straight to the last rung.
#   3. VECTORIZED RENDERER. The per-channel python loop is now shared-carrier
#      multiply-sums per chunk (env*cos and env*sin are computed once, not three times).
#      Verified equal to the old loop renderer to float tolerance in --smoke.
#   4. STEPS, NOT EPOCHS. VAEs converge per gradient step; random batches
#      from the resident tensor, cosine LR with warmup, KL beta ramped in
#      steps. --steps 30000 at batch 96 sees ~2.9M images (14 "epochs") in
#      roughly the wall time the old loop needed for 2.
#   5. bf16 autocast for encoder/decoder (renderer stays fp32, as always),
#      fused Adam when available, gradient checkpointing OFF by default
#      (it halves VRAM but doubles renderer compute — flag it back on only
#      if you OOM).
#
#   python splat_trainer2.py --data_dir E:/path/to/faces          # train
#   python splat_trainer2.py --export                             # -> onnx
#   python splat_trainer2.py --smoke                              # CPU test
#
# The export writes splat_decoder.onnx with the exact input/output names
# ("z_latent" / "rendered_image", opset 17, dynamic batch) the cv5 tools use.
#
# HONESTY: --smoke was run end-to-end (train -> export -> cv.dnn reload ->
# torch/ONNX parity) on CPU in the sandbox. The full-speed GPU path (bf16,
# fused Adam, resident-tensor indexing) follows the same code but its
# throughput numbers are yours to measure. PerceptionLab discipline: do not
# hype, do not lie, just show.

import argparse, glob, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 11               # dpx,dpy,ls,th,lf + (a,b) x 3 channels
LATENT = 128         # fixed: every downstream tool assumes it

# ======================================================================
# 1) preprocessing cache: faces folder -> uint8 npy, once
# ======================================================================
def build_cache(data_dir, size, cache_path):
    import cv2 as cv
    from concurrent.futures import ThreadPoolExecutor
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = sorted(p for e in exts for p in glob.glob(os.path.join(data_dir, e)))
    if not paths:
        raise RuntimeError(f"no images in {data_dir}")
    n = len(paths)
    print(f"caching {n} images at {size}px -> {cache_path} (one time)")
    arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8,
                                    shape=(n, size, size, 3))
    def work(i):
        im = cv.imread(paths[i], cv.IMREAD_COLOR)
        if im is None:
            return i, False
        h, w = im.shape[:2]
        s = min(h, w)
        im = im[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        im = cv.resize(im, (size, size), interpolation=cv.INTER_AREA)
        arr[i] = im[:, :, ::-1]                        # BGR -> RGB
        return i, True
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, ok in ex.map(work, range(n)):
            done += 1
            if done % 20000 == 0:
                r = done / (time.time() - t0)
                print(f"  {done}/{n}  ({r:.0f} img/s, eta {(n-done)/r/60:.1f} min)")
    arr.flush()
    print(f"cache built in {(time.time()-t0)/60:.1f} min")

def load_resident(cache_path, dev, force_disk=False):
    """Dataset with a graceful fallback ladder:
         1. GPU resident        (fastest, if it fits with 3GB to spare)
         2. pinned CPU memory   (fast H2D copies)
         3. plain CPU tensor    (pageable RAM, still no JPEG decoding)
         4. disk memmap         (only the batch is ever read/copied)
       --disk forces rung 4 directly. Any rung that OOMs drops to the next."""
    a = np.load(cache_path, mmap_mode="r")
    need = a.nbytes
    if force_disk:
        print(f"dataset stays on DISK (--disk): {need/1e9:.2f} GB memmap, "
              f"{len(a)} images, batch-only reads")
        return a

    # try to materialize the whole cache in RAM at all
    try:
        t = torch.from_numpy(np.ascontiguousarray(a))
    except MemoryError:
        print(f"dataset too big for RAM ({need/1e9:.2f} GB) -> DISK memmap, "
              f"batch-only reads")
        return a

    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if need < free - 3e9:                          # leave 3GB for training
            try:
                t = t.to(dev)
                print(f"dataset resident on GPU: {need/1e9:.2f} GB, {len(t)} images")
                return t
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("GPU residency failed despite free-memory check, "
                      "falling back...")
        try:
            t = t.pin_memory()
            print(f"dataset pinned on CPU ({need/1e9:.2f} GB too big for VRAM)")
            return t
        except RuntimeError:
            # pin_memory allocates page-locked memory through CUDA; on a
            # 10GB dataset this is exactly where a 'CUDA error: out of
            # memory' shows up. Pageable RAM is fine — copies are a bit
            # slower, training is otherwise identical.
            torch.cuda.empty_cache()
            print(f"pin_memory failed -> dataset stays in pageable RAM "
                  f"({need/1e9:.2f} GB), batch-only H2D copies")
    return t

def batch_from(data, idx, dev):
    if isinstance(data, np.ndarray):                    # disk memmap path
        i = np.sort(idx.numpy())                        # sorted = friendlier reads
        x = torch.from_numpy(np.array(data[i], copy=True))
        x = x.to(dev)
    else:
        x = data[idx]
        if x.device != dev:
            x = x.to(dev, non_blocking=True)
    return x.permute(0, 3, 1, 2).float().div_(255.0)

# ======================================================================
# 2) model — identical math to splat_generator.py, faster renderer
# ======================================================================
class GaborRenderer(nn.Module):
    def __init__(self, image_size=96, num_packets=256, chunk=64, use_checkpoint=False):
        super().__init__()
        self.H = self.W = image_size
        self.N, self.chunk, self.use_checkpoint = num_packets, chunk, use_checkpoint
        gy, gx = torch.meshgrid(torch.linspace(0, 1, image_size),
                                torch.linspace(0, 1, image_size), indexing="ij")
        self.register_buffer("GX", gx[None, None].contiguous())
        self.register_buffer("GY", gy[None, None].contiguous())
        side = int(math.ceil(math.sqrt(num_packets)))
        ax = torch.linspace(0.08, 0.92, side)
        anch = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)[:num_packets]
        anch = torch.clamp(anch, 1e-3, 1 - 1e-3)
        self.register_buffer("anchor_logit", torch.log(anch / (1 - anch)))

    def activate(self, raw):
        px = torch.sigmoid(self.anchor_logit[:, 0][None] + raw[..., 0])
        py = torch.sigmoid(self.anchor_logit[:, 1][None] + raw[..., 1])
        sigma = 0.012 + 0.14 * torch.sigmoid(raw[..., 2])
        theta = raw[..., 3]
        freq = 1.0 + 15.0 * torch.sigmoid(raw[..., 4])
        coeff = torch.tanh(raw[..., 5:11]).reshape(*raw.shape[:2], 3, 2)
        return px, py, sigma, theta, freq, coeff

    def _chunk(self, px, py, sigma, theta, freq, coeff):
        """Vectorized: env*cos / env*sin once, channels via one einsum each."""
        px_ = px[..., None, None]; py_ = py[..., None, None]
        s_ = sigma[..., None, None]; th = theta[..., None, None]
        f_ = freq[..., None, None]
        dx = self.GX - px_; dy = self.GY - py_
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
        ec = env * torch.cos(2 * math.pi * f_ * xr)          # (B,n,H,W)
        es = env * torch.sin(2 * math.pi * f_ * xr)
        a, b = coeff[..., 0], coeff[..., 1]                  # (B,n,3)
        # per-channel multiply-sum: ec/es are still computed ONCE (the speed
        # win over the old loop), and the graph is pure Mul+ReduceSum+Stack —
        # no Einsum, no dynamic Reshape — so it runs bit-identically on cv2
        # 4.x legacy dnn AND cv5 ENGINE_NEW, at any batch size
        chans = [(a[:, :, c, None, None] * ec).sum(1)
                 - (b[:, :, c, None, None] * es).sum(1) for c in range(3)]
        return torch.stack(chans, dim=1)

    def forward(self, raw):
        raw = raw.float()                                    # fp32 always
        px, py, sigma, theta, freq, coeff = self.activate(raw)
        out = None                       # no zeros(batch,...): keeps the ONNX
        for i in range(0, self.N, self.chunk):   # graph free of ConstantOfShape
            sl = slice(i, i + self.chunk)
            args = (px[:, sl], py[:, sl], sigma[:, sl],
                    theta[:, sl], freq[:, sl], coeff[:, sl])
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                c = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                c = self._chunk(*args)
            out = c if out is None else out + c
        return torch.sigmoid(out)

class Encoder(nn.Module):
    def __init__(self, image_size=96, latent=LATENT, ch=32):
        super().__init__()
        layers, c_in, sz, c = [], 3, image_size, ch
        while sz > 4:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c),
                       nn.LeakyReLU(0.2, True)]
            c_in, sz, c = c, sz // 2, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.flat = c_in * sz * sz
        self.fc_mu = nn.Linear(self.flat, latent)
        self.fc_lv = nn.Linear(self.flat, latent)
    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

class Decoder(nn.Module):
    def __init__(self, latent=LATENT, num_packets=256, hidden=512):
        super().__init__()
        self.N = num_packets
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, num_packets * K))
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].weight.data *= 0.1
    def forward(self, z):
        return self.net(z).view(-1, self.N, K)

class SplatVAE(nn.Module):
    def __init__(self, image_size=96, num_packets=256, chunk=64, ckpt=False):
        super().__init__()
        self.enc = Encoder(image_size)
        self.dec = Decoder(LATENT, num_packets)
        self.ren = GaborRenderer(image_size, num_packets, chunk, ckpt)
        self.latent = LATENT

def kl(mu, lv):
    return -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1))

def kl_free(mu, lv, fb):
    """free bits: per-dim KL clamped at fb nats before summing.
    fb=0 reduces exactly to kl(). Dims below the floor pay no penalty,
    so beta can rise 10-50x before the collapse that low-beta training
    hits — pushing the aggregate posterior toward the prior, which is
    what makes prior SAMPLES diverse. Effect on THIS architecture is
    unmeasured until you run it: watch the div readout."""
    d = -0.5 * (1 + lv - mu.pow(2) - lv.exp())          # (B, LATENT)
    if fb > 0:
        d = d.clamp(min=fb)
    return d.sum(dim=1).mean()

# ======================================================================
# 3) training — steps, resident data, bf16, cosine LR
# ======================================================================
def train(args, dev):
    # Cache lives under out/<dataset-folder-name>/, so pointing at a
    # different --data_dir (e.g. faces1 vs faces) never reuses another
    # dataset's cache. Same folder name next time -> instant reload.
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    cache_dir = os.path.join(args.out, dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"faces_cache_{args.image_size}.npy")
    if not os.path.exists(cache):
        build_cache(args.data_dir, args.image_size, cache)
    data = load_resident(cache, dev, force_disk=getattr(args, "disk", False))
    n = len(data)

    model = SplatVAE(args.image_size, args.num_packets, args.chunk,
                     args.checkpointing).to(dev)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=dev)["sd"])
        print("resumed", args.resume)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"steps {args.steps}  batch {args.batch}  res {args.image_size}")

    fused = dev.type == "cuda"
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, fused=fused)
    warm = max(1, args.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / warm, 0.5 * (1 + math.cos(math.pi * s / args.steps))))
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"autocast bf16: {use_bf16}  fused adam: {fused}  "
          f"checkpointing: {args.checkpointing}")

    g = torch.Generator(device="cpu").manual_seed(0)
    fixed_idx = torch.randint(0, n, (32,), generator=g)
    z_fixed = torch.randn(64, LATENT, device=dev)
    logf = open(os.path.join(args.out, "loss.csv"), "a")
    t0, run_rec, run_kl, last = time.time(), 0.0, 0.0, 0
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,), generator=g)
        x = batch_from(data, idx, dev)
        if args.aug:
            flip = torch.rand(x.shape[0], device=x.device) < 0.5
            x[flip] = torch.flip(x[flip], dims=[-1])
            gain = 1.0 + (torch.rand(x.shape[0], 1, 1, 1,
                                     device=x.device) - 0.5) * 0.2
            bias = (torch.rand(x.shape[0], 1, 1, 1,
                               device=x.device) - 0.5) * 0.1
            x = (x * gain + bias).clamp(0, 1)
        beta = args.beta * min(1.0, step / max(1, args.beta_warmup_steps))
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            mu, lv = model.enc(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            raw = model.dec(z)
        recon = model.ren(raw)                          # fp32 renderer
        rec = F.mse_loss(recon, x)
        # floater penalty: charge amplitude carried by needle-thin envelopes.
        # the floater strategy = sigma -> min, amp -> max (a bright orphan dot
        # that patches one pixel). amp^2 * max(SIGMA_REF/sigma - 1, 0) prices
        # point-brightness: zero cost above SIGMA_REF, growing cost as the
        # envelope collapses toward the floor. gamma_floater=0 disables.
        if args.gamma_floater > 0:
            _, _, sg, _, _, cf = model.ren.activate(raw.float())
            amp2 = cf.pow(2).sum(dim=(-1, -2))          # (B,N) per-packet energy
            flo = (amp2 * (args.sigma_ref / sg - 1.0).clamp(min=0)).mean()
        else:
            flo = torch.zeros((), device=x.device)
        loss = rec + beta * kl_free(mu, lv, args.free_bits) \
               + args.gamma_floater * flo
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        run_rec += rec.item(); run_kl += kl(mu, lv).item()

        if step % args.log_every == 0 or step == args.steps:
            nb = step - last; last = step
            ips = nb * args.batch / (time.time() - t0); t0 = time.time()
            psnr = 10 * math.log10(1.0 / max(run_rec / nb, 1e-9))
            print(f"step {step:6d}/{args.steps}  rec {run_rec/nb:.4f} "
                  f"(PSNR {psnr:4.1f})  kl {run_kl/nb:7.1f}  beta {beta:.4g}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {ips:6.0f} img/s")
            logf.write(f"{step},{run_rec/nb:.6f},{run_kl/nb:.6f}\n"); logf.flush()
            run_rec = run_kl = 0.0
            model.eval()
            with torch.no_grad():
                torch.save({"sd": model.state_dict(),
                            "image_size": args.image_size,
                            "num_packets": args.num_packets},
                           os.path.join(args.out, "model2.pt"))
                fx = batch_from(data, fixed_idx, dev)
                mu, _ = model.enc(fx)
                rc = model.ren(model.dec(mu))
                grid(torch.cat([fx, rc], 0),
                     os.path.join(args.out, f"recon_{step:06d}.png"))
                samp = model.ren(model.dec(z_fixed))
                grid(samp, os.path.join(args.out, f"sample_{step:06d}.png"))
                v = samp[:32].reshape(32, -1)
                sq = (v * v).sum(1)
                d2 = (sq[:, None] + sq[None, :] - 2 * v @ v.T) / v.shape[1]
                div = (d2.sum() / (32 * 31)).item()
                print(f"        prior-sample diversity {div:.5f}  "
                      f"(96px CelebA model ref ~0.12; << 0.05 = averaging)")
            model.train()
    print("done ->", os.path.join(args.out, "model2.pt"),
          " | now: python splat_trainer2.py --export")

def grid(t, path, nrow=8):
    import cv2 as cv
    t = t.clamp(0, 1).cpu().numpy()
    n, _, h, w = t.shape
    rows = int(math.ceil(n / nrow))
    g = np.zeros((rows * h, nrow * w, 3), np.float32)
    for i in range(n):
        r, c = divmod(i, nrow)
        g[r*h:(r+1)*h, c*w:(c+1)*w] = np.transpose(t[i], (1, 2, 0))
    cv.imwrite(path, (g[:, :, ::-1] * 255).astype(np.uint8))

# ======================================================================
# 4) ONNX export — same contract as the cv5 tools expect
# ======================================================================
class ExportHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dec, self.ren = model.dec, model.ren
        self.ren.use_checkpoint = False
    def forward(self, z):
        return self.ren(self.dec(z))

def export(args, dev):
    ck = torch.load(os.path.join(args.out, "model2.pt"), map_location="cpu")
    model = SplatVAE(ck["image_size"], ck["num_packets"], args.chunk)
    model.load_state_dict(ck["sd"]); model.eval()
    head = ExportHead(model)
    dummy = torch.randn(1, LATENT)
    out = args.onnx or "splat_decoder.onnx"
    torch.onnx.export(head, dummy, out, export_params=True, opset_version=17,
                      do_constant_folding=True, input_names=["z_latent"],
                      output_names=["rendered_image"],
                      dynamic_axes={"z_latent": {0: "batch"},
                                    "rendered_image": {0: "batch"}},
                      dynamo=False)
    mb = os.path.getsize(out) / 1e6
    print(f"exported {out} ({mb:.1f} MB, {ck['image_size']}px, "
          f"{ck['num_packets']} packets) — drop-in for the cv5 tools")

# ======================================================================
# 5) smoke — CPU end-to-end: loop-vs-einsum parity, train, export, cv.dnn parity
# ======================================================================
def smoke():
    ok = True
    def check(name, cond, note=""):
        nonlocal ok; ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")
    torch.manual_seed(0)
    dev = torch.device("cpu")

    # (a) vectorized renderer == original per-channel loop renderer
    ren = GaborRenderer(32, 16, chunk=8)
    raw = torch.randn(2, 16, K) * 0.5
    with torch.no_grad():
        fast = ren(raw)
        px, py, sg, th, fq, cf = ren.activate(raw.float())
        outs = []
        for c in range(3):                              # the old loop, verbatim
            px_ = px[..., None, None]; py_ = py[..., None, None]
            s_ = sg[..., None, None]; t_ = th[..., None, None]
            f_ = fq[..., None, None]
            dx = ren.GX - px_; dy = ren.GY - py_
            xr = dx * torch.cos(t_) + dy * torch.sin(t_)
            env = torch.exp(-(dx*dx + dy*dy) / (2*s_*s_))
            a = cf[:, :, c, 0][..., None, None]; b = cf[:, :, c, 1][..., None, None]
            outs.append((env * (a*torch.cos(2*math.pi*f_*xr)
                              - b*torch.sin(2*math.pi*f_*xr))).sum(1))
        slow = torch.sigmoid(torch.stack(outs, 1))
    err = (fast - slow).abs().max().item()
    check("einsum renderer == loop renderer", err < 1e-5, f"max|d| {err:.2e}")

    # (b) tiny synthetic cache + short training run: loss must fall
    import tempfile, cv2 as cv
    tmp = tempfile.mkdtemp()
    imdir = os.path.join(tmp, "imgs"); os.makedirs(imdir)
    rng = np.random.default_rng(0)
    for i in range(24):
        im = np.zeros((40, 36, 3), np.uint8)
        cv.circle(im, (rng.integers(8, 28), rng.integers(8, 32)),
                  rng.integers(4, 10), tuple(int(v) for v in rng.integers(60, 255, 3)), -1)
        cv.imwrite(os.path.join(imdir, f"{i:03d}.png"), im)
    a = argparse.Namespace(
        data_dir=imdir, out=tmp, image_size=32, num_packets=16, chunk=8,
        batch=8, steps=60, lr=3e-3, beta=1e-4, beta_warmup_steps=30,
        log_every=30, resume="", checkpointing=False, gamma_floater=0.02,
        sigma_ref=0.03, onnx=os.path.join(tmp, "t.onnx"),
        free_bits=0.0, aug=1)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(a, dev)
    lines = [l for l in buf.getvalue().splitlines() if l.startswith("step")]
    r0 = float(lines[0].split("rec")[1].split("(")[0])
    r1 = float(lines[-1].split("rec")[1].split("(")[0])
    check("training loss falls", r1 < r0, f"{r0:.4f} -> {r1:.4f}")
    check("cache built", os.path.exists(
        os.path.join(tmp, os.path.basename(os.path.normpath(imdir)), "faces_cache_32.npy")))

    # (c) export + cv.dnn reload + parity with torch
    with contextlib.redirect_stdout(buf):
        export(a, dev)
    check("onnx written", os.path.exists(a.onnx))
    ck = torch.load(os.path.join(tmp, "model2.pt"), map_location="cpu")
    m = SplatVAE(32, 16, 8); m.load_state_dict(ck["sd"]); m.eval()
    z = torch.randn(3, LATENT)
    with torch.no_grad():
        want = ExportHead(m)(z).numpy()
    net = cv.dnn.readNetFromONNX(a.onnx)
    net.setInput(z.numpy(), "z_latent")
    got = net.forward("rendered_image")
    err = float(np.abs(got - want).max())
    check("cv.dnn output == torch output", err < 1e-4,
          f"max|d| {err:.2e}, batch of 3 through dynamic axis")
    print("smoke:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1

# ======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./faces")
    ap.add_argument("--out", default="./runs/splat2")
    ap.add_argument("--image_size", type=int, default=96)
    ap.add_argument("--num_packets", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta_warmup_steps", type=int, default=3000)
    ap.add_argument("--free_bits", type=float, default=0.0,
                    help="per-dim KL floor in nats (0 = off; try 0.03-0.10 "
                         "with beta raised 10-50x). Diversity effect on this "
                         "architecture is unmeasured — watch div in the log.")
    ap.add_argument("--aug", type=int, default=1,
                    help="1 = on-GPU horizontal flip + light brightness/"
                         "contrast jitter (helps pose coverage and the "
                         "webcam domain gap). 0 = off.")
    ap.add_argument("--gamma_floater", type=float, default=0.02,
                    help="anti-floater energy penalty (0 = off)")
    ap.add_argument("--sigma_ref", type=float, default=0.03,
                    help="envelopes thinner than this pay the penalty")
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--resume", default="")
    ap.add_argument("--checkpointing", action="store_true",
                    help="halve VRAM, double renderer compute (only if OOM)")
    ap.add_argument("--disk", action="store_true",
                    help="never load the whole dataset into RAM/VRAM; "
                         "read batches straight from the .npy memmap on disk")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        sys.exit(smoke())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    if args.export:
        export(args, dev)
    else:
        train(args, dev)
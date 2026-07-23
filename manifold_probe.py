#!/usr/bin/env python3
# ============================================================================
# manifold_probe.py — Spatial Manifold & Covariance Diagnostics
#
# Sweeps pin drags across a grid to measure:
#   1. Coupling Vector Field (How unpinned features co-move)
#   2. Stiffness Heatmap (Latent cost ||dz|| per unit drag)
#   3. Elastic Limit (Max drag distance before moiré threshold)
# ============================================================================
import argparse, math, sys
import numpy as np
import torch
import cv2

try:
    import splat_trainer5 as ST
    from splat_ragdoll import RagdollSolver, Pin, load_model, packet_amp, grab_weights
except ImportError:
    sys.exit("Put manifold_probe.py next to splat_trainer5.py and splat_ragdoll.py")

torch.set_grad_enabled(False)


def probe_manifold(model_path, grid_size=16, drag_amount=0.05, seed=0):
    model, ck = load_model(model_path)
    S = model.ren.H
    torch.manual_seed(seed)
    z0 = torch.randn(128) * 0.8
    raw0 = model.dec(z0[None]).float()
    px0, py0, sg0, th0, fr0, cf0 = model.ren.activate(raw0)
    amp0 = packet_amp(cf0)
    bg = torch.ones(model.ren.N)
    solver = RagdollSolver(model, lam=0.08, beta=0.3, step_clip=0.5)

    gx = np.linspace(0.15, 0.85, grid_size)
    gy = np.linspace(0.15, 0.85, grid_size)
    stiffness_map = np.zeros((grid_size, grid_size))
    elastic_map = np.zeros((grid_size, grid_size))

    print(f"Probing {model_path} ({S}px, N={model.ren.N}) on {grid_size}x{grid_size} grid...")

    for i, py_val in enumerate(gy):
        for j, px_val in enumerate(gx):
            w = grab_weights(px0, py0, sg0, amp0, (px_val, py_val), 0.10, bg)
            if float(w.sum()) < 1e-6:
                continue

            # Centroid at current point
            ws = w.sum() + 1e-9
            cx0 = float((w * px0[0]).sum() / ws)
            cy0 = float((w * py0[0]).sum() / ws)

            # --- 1. Measure Stiffness ||dz|| for fixed small drag ---
            pin = Pin(w, (cx0 + drag_amount, cy0))
            z_step, err = solver.step(z0.clone(), z0, [pin], iters=10, posture=True)
            dz_norm = float((z_step - z0).norm())
            stiffness_map[i, j] = dz_norm / drag_amount

            # --- 2. Measure Elastic Limit (drag until moire > 0.15) ---
            max_valid_drag = 0.0
            for test_drag in np.linspace(0.02, 0.30, 10):
                pin_test = Pin(w, (cx0 + test_drag, cy0))
                z_test, _ = solver.step(z0.clone(), z0, [pin_test], iters=15, posture=False)
                r_test = model.ren(model.dec(z_test[None]).float())
                r_orig = model.ren(raw0)
                mo = float(ST.moire_index(r_test, r_orig))
                if mo < 0.15:
                    max_valid_drag = test_drag
                else:
                    break
            elastic_map[i, j] = max_valid_drag

    return model, z0, stiffness_map, elastic_map, gx, gy


def render_probe_visuals(model, z0, stiffness_map, elastic_map, gx, gy):
    S = model.ren.H
    scale = 4
    D = S * scale
    raw0 = model.dec(z0[None]).float()
    recon = model.ren(raw0)
    base_img = (recon[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)[:, :, ::-1]
    base_img = cv2.resize(base_img, (D, D), interpolation=cv2.INTER_CUBIC)

    # Normalize maps for colormap display
    stiff_norm = cv2.normalize(stiffness_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stiff_color = cv2.applyColorMap(cv2.resize(stiff_norm, (D, D)), cv2.COLORMAP_JET)

    elastic_norm = cv2.normalize(elastic_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elastic_color = cv2.applyColorMap(cv2.resize(elastic_norm, (D, D)), cv2.COLORMAP_VIRIDIS)

    # Blend overlays with reconstruction
    stiff_overlay = cv2.addWeighted(base_img, 0.5, stiff_color, 0.5, 0)
    elastic_overlay = cv2.addWeighted(base_img, 0.5, elastic_color, 0.5, 0)

    combined = np.hstack([base_img, stiff_overlay, elastic_overlay])
    cv2.putText(combined, "Base Recon", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Stiffness Map ||dz||", (D + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "Elastic Limit (Max Drag)", (2 * D + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Manifold Probe — Latent Space Geometry", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manifold Probe")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--grid", type=int, default=16, help="Probe grid resolution")
    args = parser.parse_args()

    model, z0, stiff, elastic, gx, gy = probe_manifold(args.model, grid_size=args.grid)
    render_probe_visuals(model, z0, stiff, elastic, gx, gy)
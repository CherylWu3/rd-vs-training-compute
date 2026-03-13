"""
Regression: Image Model Training Compute vs. Parameters x Resolution
=====================================================================

Data comes from Epoch's Notable AI Models dataset (all_ai_models.csv),
filtered to dense (non-MoE) image generation models with both training
compute and parameters. Resolutions manually curated.

Excluded:
  - MoE models (Wu Dao 2.0)
  - Multimodal LLMs (Chameleon-34B, Amazon Titan, Hunyuan, Emu1/3.5, etc.)
  - Non-image-gen (Earth-2, FragLlama)
  - Fine-tunes with suspiciously low FLOP (eDiff-I, YaART)
  - Very low-res research models (iGPT 48x48, DDPM-IP 64x64, etc.)

Regression: log10(training_FLOP) ~ log10(params * pixels)

Uncertainty quantification:
  - Bootstrap: resample data points (captures estimation uncertainty from small N)
  - Image-01 param MC: sample params from U(6B, 18B)
"""
from __future__ import annotations

import csv
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# CONSTANTS
# =============================================================================
N_BOOT = 20_000
RNG = np.random.default_rng(42)


# =============================================================================
# PARSE DATA
# =============================================================================

def parse_resolution(res_str: str) -> int | None:
    """Parse resolution string to total pixels. Returns None if unparseable."""
    res_str = res_str.strip()
    res_str_first = res_str.split("/")[0].strip()

    m = re.match(r"^(\d+)p$", res_str_first)
    if m:
        h = int(m.group(1))
        w = int(h * 16 / 9)
        return w * h

    m = re.search(r"(\d+)\s*x\s*(\d+)", res_str_first)
    if m:
        return int(m.group(1)) * int(m.group(2))

    return None


def load_image_models(path=os.path.join(os.path.dirname(__file__), "..", "data", "image_models.csv")):
    """Load image models from CSV.

    Returns list of dicts with keys:
        name, flop, params, pixels, mfu (float or None)
    Drops models missing training compute, parameters, or resolution.
    """
    models = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Model"].strip()

            flop_str = row["Training compute (FLOP)"].strip()
            if not flop_str:
                print(f"  Skipping {name}: no training compute")
                continue
            flop = float(flop_str)
            if flop == 0:
                print(f"  Skipping {name}: zero training compute")
                continue

            # Subtract finetune compute to get base model compute only
            finetune_str = row.get("Finetune compute (FLOP)", "").strip()
            if finetune_str:
                finetune_flop = float(finetune_str)
                if finetune_flop > 0:
                    base_flop = flop - finetune_flop
                    print(f"  {name}: subtracting finetune {finetune_flop:.2e} from total {flop:.2e} -> base {base_flop:.2e}")
                    flop = base_flop

            params_str = row["Parameters"].strip()
            if not params_str:
                print(f"  Skipping {name}: no parameter count")
                continue
            params = float(params_str)

            res_str = row.get("Resolution", "").strip()
            pixels = parse_resolution(res_str)
            if pixels is None:
                print(f"  Skipping {name}: could not parse resolution '{res_str}'")
                continue

            mfu_str = row.get("Hardware utilization (MFU)", "").strip()
            mfu = float(mfu_str) if mfu_str else None

            models.append({
                "name": name,
                "flop": flop,
                "params": params,
                "pixels": pixels,
                "mfu": mfu,
            })

    return models


# =============================================================================
# REGRESSION
# =============================================================================

def run_ols(log_x, log_y):
    """OLS regression. Returns (slope, intercept, r2, p_value)."""
    slope, intercept, r, p, se = stats.linregress(log_x, log_y)
    return slope, intercept, r**2, p


def bootstrap_regression(log_x, log_y, n_boot=N_BOOT, rng=RNG):
    """Bootstrap: resample (x,y) pairs, fit OLS each time."""
    n = len(log_x)
    slopes = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        s, ic, _, _ = run_ols(log_x[idx], log_y[idx])
        slopes[i] = s
        intercepts[i] = ic
    return slopes, intercepts


# =============================================================================
# PLOTTING & ANALYSIS
# =============================================================================

def plot_regression(models, save_path="../output/image_resolution_regression.png"):
    params = np.array([m["params"] for m in models])
    pixels = np.array([m["pixels"] for m in models], dtype=float)
    flops = np.array([m["flop"] for m in models])
    param_pixels = params * pixels
    log_x = np.log10(param_pixels)
    log_y = np.log10(flops)

    # --- Point estimate OLS ---
    slope, intercept, r2, p_val = run_ols(log_x, log_y)

    print(f"\n{'='*70}")
    print(f"Point estimate OLS:")
    print(f"  log10(FLOP) = {slope:.3f} * log10(params*pixels) + {intercept:.3f}")
    print(f"  R² = {r2:.3f},  p = {p_val:.2e}")
    print(f"  N = {len(models)} models")
    print(f"  Interpretation: doubling params*pixels -> {2**slope:.2f}x FLOP")

    # --- Bootstrap ---
    print(f"\nBootstrap ({N_BOOT} iterations):")
    boot_slopes, boot_intercepts = bootstrap_regression(log_x, log_y)
    print(f"  Slope:     median={np.median(boot_slopes):.3f}  "
          f"90% CI=[{np.percentile(boot_slopes, 5):.3f}, {np.percentile(boot_slopes, 95):.3f}]  "
          f"95% CI=[{np.percentile(boot_slopes, 2.5):.3f}, {np.percentile(boot_slopes, 97.5):.3f}]")
    print(f"  Intercept: median={np.median(boot_intercepts):.3f}  "
          f"90% CI=[{np.percentile(boot_intercepts, 5):.3f}, {np.percentile(boot_intercepts, 95):.3f}]")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(log_x, log_y, s=60, c="#1f77b4", edgecolors="white",
               linewidths=0.5, zorder=5)

    for i, m in enumerate(models):
        ax.annotate(m["name"][:25], (log_x[i], log_y[i]),
                   textcoords="offset points", xytext=(5, 5),
                   fontsize=6, alpha=0.7)

    # Regression line + CI band
    x_range = np.linspace(log_x.min() - 0.3, log_x.max() + 0.3, 200)
    med_slope = np.median(boot_slopes)
    med_int = np.median(boot_intercepts)
    ax.plot(x_range, med_slope * x_range + med_int,
            color="#d62728", linewidth=2, zorder=4,
            label=f"Median fit (slope={med_slope:.2f})")

    y_lines = boot_slopes[:, None] * x_range[None, :] + boot_intercepts[:, None]
    lo90 = np.percentile(y_lines, 5, axis=0)
    hi90 = np.percentile(y_lines, 95, axis=0)
    ax.fill_between(x_range, lo90, hi90, alpha=0.15, color="#d62728",
                    label="90% CI (bootstrap)")
    lo95 = np.percentile(y_lines, 2.5, axis=0)
    hi95 = np.percentile(y_lines, 97.5, axis=0)
    ax.fill_between(x_range, lo95, hi95, alpha=0.07, color="#d62728",
                    label="95% CI")

    ax.set_xlabel(r"log$_{10}$(params $\times$ pixels)", fontsize=12)
    ax.set_ylabel(r"log$_{10}$(training FLOP)", fontsize=12)
    ax.set_title("Image Generation Models: Training FLOP vs. Params x Pixels",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved to {save_path}")
    plt.show()

    return boot_slopes, boot_intercepts


# =============================================================================
# PREDICTION: MiniMax Image-01
# =============================================================================

def predict_image01(models, boot_slopes, boot_intercepts,
                    save_path="../output/image01_prediction.png"):
    """
    Predict MiniMax Image-01 training FLOP (conditional mean):
      - Use bootstrap regression: log10(FLOP) ~ log10(params * pixels)
      - Sample Image-01 params from U(6B, 18B)
      - Resolution: 2048x2048

    Propagates:
      - Bootstrap uncertainty (small N)
      - Uncertainty in Image-01's param count
    """
    image01_pixels = 2048 * 2048  # 4,194,304 pixels
    IMAGE01_PARAMS_LO = 6e9
    IMAGE01_PARAMS_HI = 18e9

    N_ITER = len(boot_slopes)
    rng = np.random.default_rng(456)

    predicted_log_flops = np.empty(N_ITER)

    for i in range(N_ITER):
        slope = boot_slopes[i]
        intercept = boot_intercepts[i]

        # Sample Image-01 params
        img01_params = rng.uniform(IMAGE01_PARAMS_LO, IMAGE01_PARAMS_HI)
        log_x_img01 = np.log10(img01_params * image01_pixels)

        # Predict conditional mean log10(FLOP)
        predicted_log_flops[i] = slope * log_x_img01 + intercept

    predicted_flops = 10 ** predicted_log_flops

    # --- Results ---
    print(f"\n{'='*70}")
    print(f"MiniMax Image-01 Training FLOP Prediction")
    print(f"{'='*70}")
    print(f"  Resolution: 2048x2048 = {image01_pixels:,} pixels")
    print(f"  Params: U({IMAGE01_PARAMS_LO:.0e}, {IMAGE01_PARAMS_HI:.0e})")
    print(f"  Method: {N_ITER} iterations of bootstrap + MC(params)")
    print()

    for pct_label, pcts in [("80% CI", [10, 50, 90]),
                             ("90% CI", [5, 50, 95]),
                             ("95% CI", [2.5, 50, 97.5])]:
        vals = np.percentile(predicted_flops, pcts)
        print(f"  {pct_label}:  {vals[0]:.2e}  --  median {vals[1]:.2e}  --  {vals[2]:.2e}")

    p5, p50, p95 = np.percentile(predicted_flops, [5, 50, 95])
    print(f"\n  Summary:  median = {p50:.2e} FLOP")
    print(f"            90% CI = [{p5:.2e}, {p95:.2e}]")

    # --- Plot 1: histogram of predicted log10(FLOP) ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(predicted_log_flops, bins=80, density=True, color="#1a9988",
             alpha=0.7, edgecolor="white")
    ax1.axvline(np.log10(p50), color="#d62728", linewidth=2,
                label=f"Median: {p50:.2e}")
    ax1.axvline(np.log10(p5), color="#d62728", linewidth=1, linestyle="--",
                label=f"P5: {p5:.2e}")
    ax1.axvline(np.log10(p95), color="#d62728", linewidth=1, linestyle="--",
                label=f"P95: {p95:.2e}")
    ax1.set_xlabel(r"log$_{10}$(Training FLOP)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("MiniMax Image-01: Predicted Training FLOP\n"
                  "(bootstrap + MC over params)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        base = save_path.rsplit(".", 1)
        hist_path = f"{base[0]}_histogram.{base[1]}" if len(base) == 2 else f"{save_path}_histogram"
        fig1.savefig(hist_path, dpi=200, bbox_inches="tight")
        print(f"\n  Saved to {hist_path}")
    plt.show()

    # --- Plot 2: scatter + Image-01 prediction band ---
    params = np.array([m["params"] for m in models])
    pixels = np.array([m["pixels"] for m in models], dtype=float)
    flops = np.array([m["flop"] for m in models])
    param_pixels = params * pixels
    log_x_data = np.log10(param_pixels)
    log_y_data = np.log10(flops)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(log_x_data, log_y_data, s=60, c="#1f77b4", edgecolors="white",
                linewidths=0.5, zorder=5)
    for i, m in enumerate(models):
        ax2.annotate(m["name"][:20], (log_x_data[i], log_y_data[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, alpha=0.7)

    # Regression line
    x_range = np.linspace(log_x_data.min() - 0.5,
                          max(log_x_data.max(),
                              np.log10(IMAGE01_PARAMS_HI * image01_pixels)) + 0.5,
                          200)
    slope_pt, int_pt, _, _ = run_ols(log_x_data, log_y_data)
    ax2.plot(x_range, slope_pt * x_range + int_pt,
             color="#d62728", linewidth=1.5, alpha=0.7)

    # Image-01 prediction region
    img01_pp_lo = IMAGE01_PARAMS_LO * image01_pixels
    img01_pp_hi = IMAGE01_PARAMS_HI * image01_pixels
    img01_flop_p5 = np.percentile(predicted_flops, 5)
    img01_flop_p95 = np.percentile(predicted_flops, 95)

    ax2.fill_between(
        [np.log10(img01_pp_lo), np.log10(img01_pp_hi)],
        [np.log10(img01_flop_p5)] * 2,
        [np.log10(img01_flop_p95)] * 2,
        alpha=0.25, color="#e85d75", label="Image-01 90% CI",
    )
    ax2.scatter(
        [np.log10(np.sqrt(img01_pp_lo * img01_pp_hi))],
        [np.log10(p50)],
        s=120, c="#e85d75", edgecolors="black", linewidths=1, marker="*",
        zorder=6, label="Image-01 median",
    )

    ax2.set_xlabel(r"log$_{10}$(params $\times$ pixels)", fontsize=11)
    ax2.set_ylabel(r"log$_{10}$(training FLOP)", fontsize=11)
    ax2.set_title("Regression with MiniMax Image-01 Prediction", fontsize=11,
                  fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig2.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved to {save_path}")
    plt.show()

    return {"p5": p5, "p50": p50, "p95": p95}


def get_image_percentiles():
    """Run full image regression pipeline and return p5/p50/p95 for Image-01.

    Returns dict with key "image01" mapping to {"p5": float, "p50": float, "p95": float}.
    """
    models = load_image_models()
    params = np.array([m["params"] for m in models])
    pixels = np.array([m["pixels"] for m in models], dtype=float)
    log_x = np.log10(params * pixels)
    log_y = np.log10(np.array([m["flop"] for m in models]))
    boot_slopes, boot_intercepts = bootstrap_regression(log_x, log_y)
    return {
        "image01": predict_image01(models, boot_slopes, boot_intercepts, save_path=None),
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    models = load_image_models()
    print(f"\nLoaded {len(models)} image models with FLOP + params + resolution:")
    print(f"  {'Model':<40s}  {'FLOP':>12s}  {'Params':>12s}  {'Pixels':>10s}  {'Params*Pix':>14s}")
    print(f"  {'-'*40}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*14}")
    for m in models:
        print(f"  {m['name']:40s}  {m['flop']:12.2e}  {m['params']:12.2e}  "
              f"{m['pixels']:10,}  {m['params']*m['pixels']:14.2e}")

    boot_slopes, boot_intercepts = plot_regression(models)
    predict_image01(models, boot_slopes, boot_intercepts)

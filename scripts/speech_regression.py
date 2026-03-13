"""
Regression: Speech/Audio Model Training Compute vs. Parameters
==============================================================

Data comes from Epoch's Notable AI Models dataset (all_ai_models.csv),
filtered to waveform-generating speech/audio models with both training
compute and parameters.

Regression: log10(training_FLOP) ~ log10(params * sample_rate_Hz)

Uncertainty quantification:
  - Bootstrap: resample data points (captures estimation uncertainty from small N)
  - Speech-02 param MC: sample params from a prior range
"""

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
# SPEECH / AUDIO WAVEFORM GENERATOR MODELS
# =============================================================================
# Curated from Epoch's Notable AI Models dataset.
# Only waveform generators (TTS, speech synthesis, audio generation).
# ASR-only, classification-only, and symbolic (MIDI) models are excluded.
#
# sample_rate: output sample rate in Hz
# max_duration_s: max output duration in seconds (None if no hard cap published)

SPEECH_MODELS = [
    {
        "name": "Step-Audio-Chat 130B",
        "params": 1.30e+11,
        "flop": 1.30e+24,       # audio training only; excludes base LLM pretrain (6.24e+23)
        "sample_rate": 22050,   # 41.6 Hz audio-token rate -> ~22.05 kHz waveform
        "max_duration_s": None,
    },
    {
        "name": "Movie Gen Audio",
        "params": 1.30e+10,
        "flop": 1.40e+23,
        "sample_rate": 48000,
        "max_duration_s": None,  # "several minutes"
    },
    {
        "name": "OpenAudio-S1",
        "params": 4.00e+09,
        "flop": 3.36e+22,
        "sample_rate": 44100,
        "max_duration_s": None,
    },
    {
        "name": "AudioGen",
        "params": 1.00e+09,
        "flop": 7.20e+21,       # AudioGen training only; excludes T5-Large text encoder pretrain (2.3e+21)
        "sample_rate": 16000,
        "max_duration_s": 10,
    },
    {
        "name": "VALL-E X",
        "params": 7.00e+08,
        "flop": 1.20e+21,
        "sample_rate": 24000,
        "max_duration_s": 22,
    },
    {
        "name": "E2 TTS",
        "params": 3.35e+08,
        "flop": 4.94e+20,
        "sample_rate": 24000,
        "max_duration_s": 22,
    },
    {
        "name": "F5-TTS",
        "params": 3.36e+08,
        "flop": 4.53e+20,
        "sample_rate": 24000,
        "max_duration_s": 22,
    },
    # MuseNet: symbolic MIDI, not waveform — SKIP
    {
        "name": "Kokoro v1.0",
        "params": 8.20e+07,
        "flop": 1.68e+20,
        "sample_rate": 24000,
        "max_duration_s": None,
    },
    {
        "name": "Kokoro v0.19",
        "params": 8.20e+07,
        "flop": 1.68e+20,
        "sample_rate": 24000,
        "max_duration_s": None,
    },
    {
        "name": "OuteTTS-0.1-350M",
        "params": 3.50e+08,
        "flop": 6.30e+19,
        "sample_rate": 24000,
        "max_duration_s": 55,  # ~54.6 s theoretical from 4096 max_length / 75 tok/s
    },
    {
        "name": "VALL-E",
        "params": 3.53e+08,
        "flop": 1.01e+19,
        "sample_rate": 24000,
        "max_duration_s": None,
    },
    # FastSpeech: outputs mel spectrogram, not waveform — SKIP
    {
        "name": "AudioLM",
        "params": 1.50e+09,
        "flop": 3.90e+18,
        "sample_rate": 16000,
        "max_duration_s": 10,  # 7 s continuation from 3 s prompt
    },
    # FastSpeech 2: outputs mel spectrogram, not waveform — SKIP
    # Pre-2010 models (Weight Decay, NetTalk variants): too architecturally different — SKIP
]


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

def plot_regression(models, save_path="../output/speech_regression.png"):
    params = np.array([m["params"] for m in models])
    hz = np.array([m["sample_rate"] for m in models], dtype=float)
    flops = np.array([m["flop"] for m in models])
    param_hz = params * hz
    log_x = np.log10(param_hz)
    log_y = np.log10(flops)

    # --- Point estimate OLS ---
    slope, intercept, r2, p_val = run_ols(log_x, log_y)

    print(f"\n{'='*70}")
    print(f"Point estimate OLS:")
    print(f"  log10(FLOP) = {slope:.3f} * log10(params*Hz) + {intercept:.3f}")
    print(f"  R² = {r2:.3f},  p = {p_val:.2e}")
    print(f"  N = {len(models)} models")
    print(f"  Interpretation: doubling params*Hz -> {2**slope:.2f}x FLOP")

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

    ax.scatter(log_x, log_y, s=60, c="#2ca02c", edgecolors="white",
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

    ax.set_xlabel(r"log$_{10}$(params $\times$ Hz)", fontsize=12)
    ax.set_ylabel(r"log$_{10}$(training FLOP)", fontsize=12)
    ax.set_title("Speech/Audio Generation Models: Training FLOP vs. Params x Hz",
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
# PREDICTION: MiniMax Speech-01 & Speech-02
# =============================================================================

def predict_speech(models, boot_slopes, boot_intercepts,
                   name, hz, params_lo, params_hi,
                   save_path=None, color="#e85d75", seed=123):
    """
    Predict training FLOP for a speech model via bootstrap + MC:
      - Bootstrap regression: log10(FLOP) ~ log10(params * Hz)
      - Sample params from U(params_lo, params_hi)
      - Hz is fixed (known)

    Propagates bootstrap uncertainty (small N) and param uncertainty.
    """
    N_ITER = len(boot_slopes)
    rng = np.random.default_rng(seed)

    predicted_log_flops = np.empty(N_ITER)

    for i in range(N_ITER):
        slope = boot_slopes[i]
        intercept = boot_intercepts[i]

        sampled_params = rng.uniform(params_lo, params_hi)
        log_x = np.log10(sampled_params * hz)
        predicted_log_flops[i] = slope * log_x + intercept

    predicted_flops = 10 ** predicted_log_flops

    # --- Results ---
    print(f"\n{'='*70}")
    print(f"{name} Training FLOP Prediction")
    print(f"{'='*70}")
    print(f"  Sample rate: {hz:,} Hz")
    print(f"  Params: U({params_lo:.0e}, {params_hi:.0e})")
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
    ax1.set_title(f"{name}: Predicted Training FLOP\n"
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

    # --- Plot 2: scatter + prediction band ---
    params_arr = np.array([m["params"] for m in models])
    hz_arr = np.array([m["sample_rate"] for m in models], dtype=float)
    flops_arr = np.array([m["flop"] for m in models])
    param_hz = params_arr * hz_arr
    log_x_data = np.log10(param_hz)
    log_y_data = np.log10(flops_arr)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(log_x_data, log_y_data, s=60, c="#2ca02c", edgecolors="white",
                linewidths=0.5, zorder=5)
    for i, m in enumerate(models):
        ax2.annotate(m["name"][:20], (log_x_data[i], log_y_data[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, alpha=0.7)

    # Regression line
    x_range = np.linspace(log_x_data.min() - 0.5,
                          max(log_x_data.max(),
                              np.log10(params_hi * hz)) + 0.5,
                          200)
    slope_pt, int_pt, _, _ = run_ols(log_x_data, log_y_data)
    ax2.plot(x_range, slope_pt * x_range + int_pt,
             color="#d62728", linewidth=1.5, alpha=0.7)

    # Prediction region
    phz_lo = params_lo * hz
    phz_hi = params_hi * hz
    flop_p5 = np.percentile(predicted_flops, 5)
    flop_p95 = np.percentile(predicted_flops, 95)

    ax2.fill_between(
        [np.log10(phz_lo), np.log10(phz_hi)],
        [np.log10(flop_p5)] * 2,
        [np.log10(flop_p95)] * 2,
        alpha=0.25, color=color, label=f"{name} 90% CI",
    )
    ax2.scatter(
        [np.log10(np.sqrt(phz_lo * phz_hi))],
        [np.log10(p50)],
        s=120, c=color, edgecolors="black", linewidths=1, marker="*",
        zorder=6, label=f"{name} median",
    )

    ax2.set_xlabel(r"log$_{10}$(params $\times$ Hz)", fontsize=11)
    ax2.set_ylabel(r"log$_{10}$(training FLOP)", fontsize=11)
    ax2.set_title(f"Regression with {name} Prediction", fontsize=11,
                  fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig2.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved to {save_path}")
    plt.show()

    return {"p5": p5, "p50": p50, "p95": p95}


def get_speech_percentiles():
    """Run full speech regression pipeline and return p5/p50/p95 for all models.

    Returns dict with keys: "speech01_hd", "speech01_turbo",
    "speech02_hd", "speech02_turbo", each mapping to
    {"p5": float, "p50": float, "p95": float}.
    """
    models = SPEECH_MODELS
    params = np.array([m["params"] for m in models])
    hz = np.array([m["sample_rate"] for m in models], dtype=float)
    log_x = np.log10(params * hz)
    log_y = np.log10(np.array([m["flop"] for m in models]))
    boot_slopes, boot_intercepts = bootstrap_regression(log_x, log_y)
    return {
        "speech01_hd": predict_speech(models, boot_slopes, boot_intercepts,
            name="MiniMax Speech-01", hz=48000, params_lo=0.3e9, params_hi=2.5e9,
            save_path=None, seed=123),
        "speech01_turbo": predict_speech(models, boot_slopes, boot_intercepts,
            name="MiniMax Speech-01-turbo", hz=44100, params_lo=0.4e9, params_hi=1.5e9,
            save_path=None, seed=789),
        "speech02_hd": predict_speech(models, boot_slopes, boot_intercepts,
            name="MiniMax Speech-02", hz=44100, params_lo=0.5e9, params_hi=3e9,
            save_path=None, seed=456),
        "speech02_turbo": predict_speech(models, boot_slopes, boot_intercepts,
            name="MiniMax Speech-02-turbo", hz=44100, params_lo=0.6e9, params_hi=2e9,
            save_path=None, seed=101),
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    models = SPEECH_MODELS
    print(f"\n{len(models)} waveform-generating speech/audio models:")
    print(f"  {'Model':<30s}  {'Params':>12s}  {'FLOP':>12s}  {'Hz':>8s}  {'Params*Hz':>14s}  {'Max dur':>8s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*14}  {'-'*8}")
    for m in models:
        hz = f"{m['sample_rate']:,}" if m["sample_rate"] else "N/A"
        dur = f"{m['max_duration_s']}s" if m["max_duration_s"] else "N/A"
        phz = m["params"] * m["sample_rate"]
        print(f"  {m['name']:30s}  {m['params']:12.2e}  {m['flop']:12.2e}  {hz:>8s}  {phz:14.2e}  {dur:>8s}")

    boot_slopes, boot_intercepts = plot_regression(models)

    predict_speech(models, boot_slopes, boot_intercepts,
                   name="MiniMax Speech-01",
                   hz=48000, params_lo=0.3e9, params_hi=2.5e9,
                   save_path="../output/speech01_prediction.png",
                   color="#e85d75", seed=123)

    predict_speech(models, boot_slopes, boot_intercepts,
                   name="MiniMax Speech-02",
                   hz=44100, params_lo=0.5e9, params_hi=3e9,
                   save_path="../output/speech02_prediction.png",
                   color="#5d85e8", seed=456)

    predict_speech(models, boot_slopes, boot_intercepts,
                   name="MiniMax Speech-01-turbo",
                   hz=44100, params_lo=0.4e9, params_hi=1.5e9,
                   save_path="../output/speech01_turbo_prediction.png",
                   color="#e8a85d", seed=789)

    predict_speech(models, boot_slopes, boot_intercepts,
                   name="MiniMax Speech-02-turbo",
                   hz=44100, params_lo=0.6e9, params_hi=2e9,
                   save_path="../output/speech02_turbo_prediction.png",
                   color="#5de8a8", seed=101)

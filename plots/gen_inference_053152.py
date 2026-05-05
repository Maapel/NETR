"""
5 inference plots — latest calibration recording with complete data.
Uses 20260505_040942 (88 samples, best LOO: ~25px median).
Model targets are X,Y (homography gaze estimate). sx,sy are the stimulus targets.

LOO error = ||predicted(X,Y) - held-out(X,Y)||   ← model fitting quality
Homography error = ||X,Y - sx,sy||               ← end-to-end accuracy

Run from iot-project root:
  /home/maadhav/pio-venv/bin/python plots/gen_inference_053152.py
"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

REC = pathlib.Path("recordings/20260505_040942")
OUT = pathlib.Path("plots/20260505_040942")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from compute.gaze_model import SplitGlintModel

meta    = json.loads((REC / "meta.json").read_text())
samples = json.loads((REC / "samples.json").read_text())
SW, SH  = meta["screen_w"], meta["screen_h"]

# Ground-truth targets (stimulus dots)
sx = np.array([s["sx"] for s in samples])
sy = np.array([s["sy"] for s in samples])
# Homography-estimated gaze (model prediction target)
hx = np.array([s["X"] for s in samples])
hy = np.array([s["Y"] for s in samples])
# Homography error (end-to-end noise floor)
homo_err = np.hypot(hx - sx, hy - sy)

# Two-glint subset (model only trained on these)
both_idx = [i for i, s in enumerate(samples) if "dx1" in s]
both     = [samples[i] for i in both_idx]
bsx      = sx[both_idx];  bsy = sy[both_idx]
bhx      = hx[both_idx];  bhy = hy[both_idx]
bdx1     = np.array([s["dx1"] for s in both])
bdy1     = np.array([s["dy1"] for s in both])
bdx2     = np.array([s["dx2"] for s in both])
bdy2     = np.array([s["dy2"] for s in both])

# ── LOO on both-glint samples ────────────────────────────────────────────────
print(f"Running LOO on {len(both)} two-glint samples…")
loo_px, loo_py = [], []
for i in range(len(both)):
    leave = [t for j, t in enumerate(samples) if j != both_idx[i]]
    m = SplitGlintModel()
    m.fit(leave)
    px, py = m.predict(both[i]["dx1"], both[i]["dy1"],
                       both[i]["dx2"], both[i]["dy2"])
    loo_px.append(px); loo_py.append(py)
loo_px   = np.array(loo_px)
loo_py   = np.array(loo_py)
loo_err  = np.hypot(loo_px - bhx, loo_py - bhy)   # vs homography target
loo_e2e  = np.hypot(loo_px - bsx, loo_py - bsy)   # vs stimulus target

print(f"LOO model err:  median={np.median(loo_err):.1f}  mean={np.mean(loo_err):.1f}  p90={np.percentile(loo_err,90):.1f}")
print(f"LOO end-to-end: median={np.median(loo_e2e):.1f}  mean={np.mean(loo_e2e):.1f}  p90={np.percentile(loo_e2e,90):.1f}")
print(f"Homography err: median={np.median(homo_err):.1f}  mean={np.mean(homo_err):.1f}  p90={np.percentile(homo_err,90):.1f}")

# ── Plot 1: LOO predicted vs homography target (model quality) ───────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_xlim(0, SW); ax.set_ylim(SH, 0)
ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
cmap = plt.cm.plasma
norm = Normalize(vmin=0, vmax=np.percentile(loo_err, 95))
for i in range(len(both)):
    c = cmap(norm(loo_err[i]))
    ax.plot([bhx[i], loo_px[i]], [bhy[i], loo_py[i]], color=c, alpha=0.5, lw=0.9)
sc = ax.scatter(bhx, bhy, s=55, c=loo_err, cmap=cmap, norm=norm,
                edgecolors="white", linewidths=0.4, zorder=5, label="homog. target")
ax.scatter(loo_px, loo_py, s=18, color="cyan", alpha=0.55, zorder=4, label="LOO predicted")
plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="LOO model error (px)")
ax.set_title(f"LOO Predicted vs Homography Target\n"
             f"median {np.median(loo_err):.1f}px  mean {np.mean(loo_err):.1f}px  "
             f"p90 {np.percentile(loo_err,90):.1f}px  N={len(both)}",
             color="white", fontsize=10)
ax.set_xlabel("screen X (px)", color="white"); ax.set_ylabel("screen Y (px)", color="white")
ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
ax.legend(facecolor="#222", labelcolor="white", loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "01_loo_predicted_vs_target.png", dpi=140)
plt.close(fig)
print("01 saved")

# ── Plot 2: Error heatmap (spatial LOO error on screen) ──────────────────────
from scipy.interpolate import griddata
xi = np.linspace(0, SW, 90); yi = np.linspace(0, SH, 54)
XX, YY = np.meshgrid(xi, yi)
ZZ = griddata((bsx, bsy), loo_err, (XX, YY), method="linear")

fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(ZZ, origin="upper", extent=[0, SW, SH, 0],
               cmap="YlOrRd", aspect="auto", alpha=0.80, vmin=0)
sc = ax.scatter(bsx, bsy, s=35, c=loo_err, cmap="YlOrRd",
                edgecolors="black", linewidths=0.4, zorder=5)
plt.colorbar(im, ax=ax, label="LOO model error (px)")
ax.set_title(f"Spatial Error Heatmap  |  median {np.median(loo_err):.1f}px")
ax.set_xlabel("stimulus X (px)"); ax.set_ylabel("stimulus Y (px)")
fig.tight_layout()
fig.savefig(OUT / "02_error_heatmap.png", dpi=140)
plt.close(fig)
print("02 saved")

# ── Plot 3: Error CDF — model error vs end-to-end vs homography ──────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
def _cdf(arr, color, lbl):
    s = np.sort(arr)
    c = np.arange(1, len(s)+1) / len(s) * 100
    ax.plot(s, c, lw=2, color=color, label=lbl)
_cdf(loo_err,   "#4fc3f7", f"LOO model  (N={len(loo_err)}, med={np.median(loo_err):.0f}px)")
_cdf(loo_e2e,   "#ef9a9a", f"End-to-end (med={np.median(loo_e2e):.0f}px)")
_cdf(homo_err,  "#a5d6a7", f"Homography (all N={len(homo_err)}, med={np.median(homo_err):.0f}px)")
for pct, col in [(50, "#ffb300"), (75, "#ef5350"), (90, "#ce93d8")]:
    v = np.percentile(loo_err, pct)
    ax.axvline(v, color=col, lw=1.2, linestyle="--", alpha=0.7)
    ax.text(v+2, pct-6, f"p{pct}={v:.0f}", color=col, fontsize=8)
ax.set_xlabel("Error (px)"); ax.set_ylabel("Cumulative %")
ax.set_title("Error CDF: model vs end-to-end vs homography baseline")
ax.set_facecolor("#f9f9f9"); ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "03_error_cdf.png", dpi=140)
plt.close(fig)
print("03 saved")

# ── Plot 4: PCCR feature space — right vs left LED, coloured by target X ─────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (dx_, dy_, lbl) in zip(axes,
        [(bdx1, bdy1, "Right LED  (dx1, dy1)"),
         (bdx2, bdy2, "Left LED   (dx2, dy2)")]):
    sc = ax.scatter(dx_, dy_, c=bsx, cmap="plasma", s=35, alpha=0.85,
                    edgecolors="none", vmin=0, vmax=SW)
    ax.set_xlabel("dx (px)", fontsize=11); ax.set_ylabel("dy (px)", fontsize=11)
    ax.set_title(lbl, fontsize=11); ax.grid(alpha=0.18)
    plt.colorbar(sc, ax=ax, label="stimulus X (px)")
fig.suptitle("PCCR Feature Space — colour encodes horizontal gaze target\n"
             "dx = main axis (camera 90° rotated → vertical displacement)",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "04_pccr_feature_space.png", dpi=140)
plt.close(fig)
print("04 saved")

# ── Plot 5: Per-fixation LOO error bar sorted by screen X ────────────────────
order  = np.argsort(bsx)
colors = ["#ef5350" if e > np.percentile(loo_err, 75) else
          "#ffb300" if e > np.median(loo_err) else "#66bb6a"
          for e in loo_err[order]]
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(range(len(order)), loo_err[order], color=colors, width=0.85)
med = np.median(loo_err); p75 = np.percentile(loo_err, 75); p90 = np.percentile(loo_err, 90)
ax.axhline(med, color="white",   lw=1.5, linestyle="--")
ax.axhline(p75, color="#ffb300", lw=1.2, linestyle=":")
ax.axhline(p90, color="#ef5350", lw=1.0, linestyle=":")
ax.text(len(order)+0.3, med,  f" med={med:.0f}px",  color="white",   va="center", fontsize=8)
ax.text(len(order)+0.3, p75,  f" p75={p75:.0f}px",  color="#ffb300", va="center", fontsize=8)
ax.text(len(order)+0.3, p90,  f" p90={p90:.0f}px",  color="#ef5350", va="center", fontsize=8)
ax.set_xlabel("fixation index (sorted left→right by stimulus X)", color="white")
ax.set_ylabel("LOO model error (px)", color="white")
ax.set_title("Per-Fixation LOO Error", color="white")
ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
ax.yaxis.label.set_color("white"); ax.xaxis.label.set_color("white")
green_p = mpatches.Patch(color="#66bb6a", label="≤ median")
yel_p   = mpatches.Patch(color="#ffb300", label="median → p75")
red_p   = mpatches.Patch(color="#ef5350", label="> p75")
ax.legend(handles=[green_p, yel_p, red_p], facecolor="#222", labelcolor="white", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "05_per_fixation_error.png", dpi=140)
plt.close(fig)
print("05 saved")

print(f"\nAll plots → {OUT}/")

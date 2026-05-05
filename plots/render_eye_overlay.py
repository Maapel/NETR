"""
Apply current glint+pupil detection to eye_cam_raw.avi and save annotated video.
Uses the same EyePipeline / GlintDetector settings as the live receiver.

Run from iot-project root:
  /home/maadhav/pio-venv/bin/python plots/render_eye_overlay.py
"""
import sys, json, pathlib
import cv2
import numpy as np

REC  = pathlib.Path("recordings/20260505_040942")
OUT  = pathlib.Path("plots/20260505_040942/eye_overlay.avi")

sys.path.insert(0, ".")
sys.path.insert(0, "compute")
from compute.eye_pipeline import EyePipeline

# Load eye settings to mirror what the receiver uses
eye_cfg_path = pathlib.Path("eye_settings.json")
eye_cfg = json.loads(eye_cfg_path.read_text()) if eye_cfg_path.exists() else {}
print("Eye config keys:", list(eye_cfg.keys()))

# Build pipeline matching receiver defaults
glint_kwargs = {
    "glint_margin":      eye_cfg.get("g_glint_margin",      30),
    "brightness_thresh": eye_cfg.get("g_brightness_thresh", 180),
    "min_area":          eye_cfg.get("g_min_area",          5),
    "max_area":          eye_cfg.get("g_max_area",          800),
    "circularity_min":   eye_cfg.get("g_circularity_min",   0.60),
    "min_dist_factor":   eye_cfg.get("g_min_dist_factor",   0.70),
    "limbus_n_rays":     eye_cfg.get("g_limbus_n_rays",     16),
    "iris_radius_factor":eye_cfg.get("g_iris_radius_factor",1.8),
    "search_radius_factor":eye_cfg.get("g_search_radius_factor", 2.5),
    "ellipse_slack":     eye_cfg.get("g_ellipse_slack",     0.12),
}
pupil_kwargs = {
    "algorithm":      eye_cfg.get("p_algorithm",      "threshold"),
    "blur_ksize":     eye_cfg.get("p_blur_ksize",     7),
    "morph_ksize":    eye_cfg.get("p_morph_ksize",    5),
    "min_radius":     eye_cfg.get("p_min_radius",     15),
    "max_radius":     eye_cfg.get("p_max_radius",     150),
    "dark_percentile":eye_cfg.get("p_dark_percentile",1.8),
    "thresh_offset":  eye_cfg.get("p_thresh_offset",  8),
    "circularity_min":eye_cfg.get("p_circularity_min",0.4),
}
pipeline = EyePipeline(
    pupil_kwargs=pupil_kwargs,
    glint_kwargs=glint_kwargs,
    swap_pccr=True,
    preferred_side=1.0,
)

cap = cv2.VideoCapture(str(REC / "eye_cam_raw.avi"))
fps   = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Input: {W}×{H} @ {fps:.1f}fps  {total} frames")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out    = cv2.VideoWriter(str(OUT), fourcc, fps, (W, H))

# Colours (BGR)
COL_PUPIL    = (0, 255, 100)   # green circle
COL_PUPIL_C  = (255, 255, 255) # pupil centre dot
COL_GLINT_R  = (0, 200, 255)   # right LED glint — yellow-ish
COL_GLINT_L  = (255, 150, 0)   # left LED glint  — blue-ish
COL_PCCR     = (100, 100, 255) # PCCR vector line
COL_LIMBUS   = (80, 80, 200)   # limbus dashed circle
COL_IRIS     = (60, 180, 60)   # iris annulus tint

n_pupil = 0; n_glint = 0; n_dual = 0

for fi in range(total):
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res  = pipeline.process(gray)

    ann = frame.copy()

    # ── Iris mask tint ────────────────────────────────────────────────────────
    if res.glint.iris_mask is not None:
        tint = np.zeros_like(ann)
        tint[res.glint.iris_mask > 0] = COL_IRIS
        ann = cv2.addWeighted(ann, 0.85, tint, 0.15, 0)

    # ── Pupil ─────────────────────────────────────────────────────────────────
    if res.pupil_center and res.pupil_radius:
        cx, cy = int(res.pupil_center[0]), int(res.pupil_center[1])
        r = int(res.pupil_radius)
        cv2.circle(ann, (cx, cy), r, COL_PUPIL, 2)
        cv2.circle(ann, (cx, cy), 3, COL_PUPIL_C, -1)
        n_pupil += 1

        # Limbus circle
        if res.glint.limbus_radius:
            lr = int(res.glint.limbus_radius)
            # dashed circle via arc segments
            for deg in range(0, 360, 20):
                a1 = deg * np.pi / 180; a2 = (deg + 10) * np.pi / 180
                p1 = (int(cx + lr * np.cos(a1)), int(cy + lr * np.sin(a1)))
                p2 = (int(cx + lr * np.cos(a2)), int(cy + lr * np.sin(a2)))
                cv2.line(ann, p1, p2, COL_LIMBUS, 1)

    # ── Glints + PCCR ─────────────────────────────────────────────────────────
    dual = False
    for gx, gy, side in res.labeled_glints:
        gxi, gyi = int(gx), int(gy)
        col = COL_GLINT_R if side > 0 else COL_GLINT_L
        label = "R" if side > 0 else "L"
        cv2.circle(ann, (gxi, gyi), 5, col, -1)
        cv2.circle(ann, (gxi, gyi), 7, col, 1)
        cv2.putText(ann, label, (gxi + 8, gyi - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if len(res.labeled_glints) >= 2:
        dual = True; n_dual += 1
    if res.labeled_glints:
        n_glint += 1

    # PCCR vector from primary glint
    if res.pccr_vector and res.pupil_center:
        cx, cy = int(res.pupil_center[0]), int(res.pupil_center[1])
        dx, dy = res.pccr_vector
        gx2 = int(cx - dx); gy2 = int(cy - dy)
        cv2.arrowedLine(ann, (gx2, gy2), (cx, cy), COL_PCCR, 1,
                        tipLength=0.2, line_type=cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────────────────
    hud_lines = []
    if res.pupil_center:
        cx, cy = res.pupil_center
        hud_lines.append(f"pupil ({cx:.0f},{cy:.0f}) r={res.pupil_radius}")
    else:
        hud_lines.append("pupil: NOT FOUND")
    if res.pccr_vector:
        dx, dy = res.pccr_vector
        side_str = "+1" if res.pccr_side > 0 else "-1"
        hud_lines.append(f"PCCR dx={dx:.1f} dy={dy:.1f} side={side_str}")
    glint_n = len(res.labeled_glints)
    hud_lines.append(f"glints: {glint_n}  {'DUAL' if dual else 'single'}")
    if res.glint.limbus_radius:
        hud_lines.append(f"limbus r={res.glint.limbus_radius:.0f}px")
    hud_lines.append(f"frame {fi+1}/{total}")

    for i, txt in enumerate(hud_lines):
        y = 18 + i * 18
        cv2.putText(ann, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ann, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1, cv2.LINE_AA)

    out.write(ann)
    if fi % 100 == 0:
        print(f"  {fi}/{total}  pupil={n_pupil} glint={n_glint} dual={n_dual}", flush=True)

cap.release(); out.release()
print(f"\nSaved → {OUT}")
print(f"pupil detected: {n_pupil}/{total} ({100*n_pupil//total}%)")
print(f"glint detected: {n_glint}/{total} ({100*n_glint//total}%)")
print(f"dual glint:     {n_dual}/{total} ({100*n_dual//total}%)")

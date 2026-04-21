"""Offline evaluation of TextROIDetector on PDFs / images / videos.

Usage:
    python tools/test_text_detection.py pdf <pdf_path>   [--dpi 150] [--out text_debug/]
    python tools/test_text_detection.py image <img_path> [--out text_debug/]
    python tools/test_text_detection.py video <avi_path> [--step 30] [--out text_debug/]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compute.text_detector import TextROIDetector


def _save(out_dir: Path, name: str, img: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    cv2.imwrite(str(p), img)
    return p


def _run_on_image(det: TextROIDetector, bgr: np.ndarray, stem: str, out_dir: Path) -> None:
    regions = det.detect_regions(bgr)
    quads, theta = det.detect_lines(bgr)
    img_r = det.annotate(bgr, level="region", color=(255, 0, 0), thickness=1)
    img_l = det.annotate(bgr, level="line", color=(0, 255, 0), thickness=2)
    img_d = det.annotate_debug(bgr)
    _save(out_dir, f"{stem}_regions.png", img_r)
    _save(out_dir, f"{stem}_lines.png", img_l)
    _save(out_dir, f"{stem}_debug.png", img_d)
    print(f"  {stem}: {len(regions)} regions, {len(quads)} lines, skew={theta:+.1f}deg")


def cmd_pdf(args: argparse.Namespace) -> None:
    import fitz  # PyMuPDF

    pdf_path = Path(args.pdf_path).expanduser()
    out_dir = Path(args.out)
    det = TextROIDetector()
    doc = fitz.open(str(pdf_path))
    print(f"PDF: {pdf_path} — {len(doc)} page(s) @ {args.dpi} DPI")
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=args.dpi, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n == 3 else img
        stem = f"{pdf_path.stem}_page{i + 1:02d}"
        _run_on_image(det, bgr, stem, out_dir)


def cmd_image(args: argparse.Namespace) -> None:
    img_path = Path(args.img_path).expanduser()
    out_dir = Path(args.out)
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        sys.exit(f"could not read {img_path}")
    det = TextROIDetector()
    _run_on_image(det, bgr, img_path.stem, out_dir)


def cmd_video(args: argparse.Namespace) -> None:
    vid_path = Path(args.avi_path).expanduser()
    out_dir = Path(args.out)
    det = TextROIDetector()
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        sys.exit(f"could not open {vid_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {vid_path} — {total} frames, sampling every {args.step}")
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.step == 0:
            stem = f"{vid_path.stem}_f{idx:06d}"
            _run_on_image(det, frame, stem, out_dir)
            saved += 1
        idx += 1
    cap.release()
    print(f"Saved {saved} sampled frames to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Test TextROIDetector")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pdf")
    pp.add_argument("pdf_path")
    pp.add_argument("--dpi", type=int, default=150)
    pp.add_argument("--out", default="text_debug")
    pp.set_defaults(func=cmd_pdf)

    pi = sub.add_parser("image")
    pi.add_argument("img_path")
    pi.add_argument("--out", default="text_debug")
    pi.set_defaults(func=cmd_image)

    pv = sub.add_parser("video")
    pv.add_argument("avi_path")
    pv.add_argument("--step", type=int, default=30)
    pv.add_argument("--out", default="text_debug")
    pv.set_defaults(func=cmd_video)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""
ArUco + homography helpers shared by calibration_server and offline replay.

Screen marker IDs 0,1,2,3 = TL, TR, BL, BR (same as calibration UI).
"""

from __future__ import annotations

import cv2
import numpy as np

ARUCO_IDS = [0, 1, 2, 3]
ARUCO_DICTS = {
    "4x4": cv2.aruco.DICT_4X4_50,
    "5x5": cv2.aruco.DICT_5X5_50,
    "6x6": cv2.aruco.DICT_6X6_50,
    "7x7": cv2.aruco.DICT_7X7_50,
}

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def make_detector_params() -> cv2.aruco.DetectorParameters:
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.01
    p.polygonalApproxAccuracyRate = 0.08
    return p


def create_aruco_detector(dict_name: str) -> cv2.aruco.ArucoDetector:
    if dict_name not in ARUCO_DICTS:
        raise ValueError(f"Unknown aruco dict {dict_name!r}, expected one of {list(ARUCO_DICTS)}")
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    return cv2.aruco.ArucoDetector(d, make_detector_params())


def detect_aruco_corners(
    frame_bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> tuple[dict[int, np.ndarray] | None, list[int]]:
    """Returns (id->center in scene pixels, all detected ids). dict None if <4 required markers."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    corners, ids, _ = detector.detectMarkers(gray)
    all_ids = ids.flatten().tolist() if ids is not None else []
    if ids is None or len(ids) < 4:
        return None, all_ids
    result: dict[int, np.ndarray] = {}
    for i, mid in enumerate(ids.flatten()):
        if mid in ARUCO_IDS:
            c = corners[i][0]
            result[int(mid)] = c.mean(axis=0)
    if len(result) < 4:
        return None, all_ids
    return result, all_ids


def compute_homography_screen_to_scene(
    scene_corners: dict[int, np.ndarray],
    screen_markers: dict,
) -> np.ndarray | None:
    """
    Homography from calibration browser pixels → world camera pixels.
    screen_markers: id -> [x, y] or ndarray (4 corners required).
    """
    sm: dict[int, np.ndarray] = {}
    for k, v in screen_markers.items():
        kid = int(k)
        sm[kid] = np.asarray(v, dtype=np.float32)
    if len(sm) < 4 or not all(i in sm for i in ARUCO_IDS):
        return None
    if not all(i in scene_corners for i in ARUCO_IDS):
        return None
    screen_pts = np.array([sm[i] for i in ARUCO_IDS], dtype=np.float32)
    scene_pts = np.array([scene_corners[i] for i in ARUCO_IDS], dtype=np.float32)
    H, _ = cv2.findHomography(screen_pts, scene_pts)
    return H


def screen_to_scene(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pt = np.array([[[x, y]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H)
    return float(res[0][0][0]), float(res[0][0][1])

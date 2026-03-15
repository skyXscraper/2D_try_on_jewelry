# ---------------------------------------------------------------------------
# landmarks.py — MediaPipe Face Mesh landmark extraction (Tasks API)
#
# Uses the Tasks API (FaceLandmarker) which is stable across mediapipe 0.10.x.
# The mp.solutions.face_mesh API is broken on some 0.10.x builds; the Tasks
# API is the officially supported replacement.
# ---------------------------------------------------------------------------

import math
import os
import urllib.request

import cv2
import numpy as np

from mediapipe.tasks.python import vision as _mp_vision
from mediapipe.tasks import python as _mp_tasks
import mediapipe as mp

import config

# ---------------------------------------------------------------------------
# Model file — auto-downloaded on first run
# ---------------------------------------------------------------------------
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "face_landmarker.task")


def _ensure_model():
    if os.path.exists(_MODEL_PATH):
        return
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    print(f"[landmarks] Downloading face landmark model (~7 MB) …")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print(f"[landmarks] Model saved to {_MODEL_PATH}")


# ---------------------------------------------------------------------------
# Build the FaceLandmarker (VIDEO mode enables cross-frame tracking)
# ---------------------------------------------------------------------------
_ensure_model()

_options = _mp_vision.FaceLandmarkerOptions(
    base_options=_mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
    running_mode=_mp_vision.RunningMode.VIDEO,
    num_faces=config.MAX_NUM_FACES,
    min_face_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_face_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
_detector = _mp_vision.FaceLandmarker.create_from_options(_options)

# ---------------------------------------------------------------------------
# Key landmark indices for the 478-point face mesh
# ---------------------------------------------------------------------------
# Cheek extremes — used only for face-width measurement
LEFT_CHEEK_IDX  = 234
RIGHT_CHEEK_IDX = 454

# Earlobe approximations — sit lower on the jaw/ear contour than the cheeks
# 132 = lower-left ear region, 361 = lower-right ear region
LEFT_EAR_IDX   = 132
RIGHT_EAR_IDX  = 361

CHIN_IDX       = 152
JAW_LEFT_IDX   = 172
JAW_RIGHT_IDX  = 397
LEFT_EYE_IDX   = 33
RIGHT_EYE_IDX  = 263

# Mutable container so the timestamp can be incremented inside the function
_state = {"frame_ts_ms": 0}


def _lm_px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def get_face_landmarks(frame: np.ndarray) -> dict | None:
    """
    Process *frame* and return anchor dict, or None if no face detected.

    Keys: left_ear, right_ear, jaw_mid, face_width, tilt_angle
    """
    _state["frame_ts_ms"] += 33   # ~30 fps; close enough for VIDEO-mode tracking

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = _detector.detect_for_video(mp_img, _state["frame_ts_ms"])

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]   # list of NormalizedLandmark

    left_ear   = _lm_px(lms[LEFT_EAR_IDX],   w, h)   # earlobe anchor
    right_ear  = _lm_px(lms[RIGHT_EAR_IDX],  w, h)
    left_cheek = _lm_px(lms[LEFT_CHEEK_IDX], w, h)   # cheek extreme (face width)
    right_cheek= _lm_px(lms[RIGHT_CHEEK_IDX],w, h)
    chin       = _lm_px(lms[CHIN_IDX],        w, h)
    jaw_left   = _lm_px(lms[JAW_LEFT_IDX],   w, h)
    jaw_right  = _lm_px(lms[JAW_RIGHT_IDX],  w, h)
    left_eye   = _lm_px(lms[LEFT_EYE_IDX],   w, h)
    right_eye  = _lm_px(lms[RIGHT_EYE_IDX],  w, h)

    # Face width from cheek extremes (wider span; stable for scaling)
    face_width = math.dist(left_cheek, right_cheek)

    # Necklace anchor: horizontally centred between jaw ends, vertically at chin tip
    jaw_mid_x = (jaw_left[0] + jaw_right[0]) // 2
    jaw_mid_y = chin[1]   # chin tip is the lowest face point — best neck anchor

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    tilt_angle = math.degrees(math.atan2(dy, dx))

    return {
        "left_ear":        left_ear,       # earlobe-level anchor
        "right_ear":       right_ear,
        "jaw_mid":         (jaw_mid_x, jaw_mid_y),
        "face_width":      face_width,     # cheek-to-cheek for scale
        "tilt_angle":      tilt_angle,
        "left_eye_outer":  left_eye,       # outer eye corners for spectacles
        "right_eye_outer": right_eye,
        "eye_span":        math.dist(left_eye, right_eye),
    }

"""Real-time face mesh using MediaPipe Tasks API + OpenCV.

Press ESC to quit.
"""
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

CAMERA_INDEX = 0          # change to 1, 2... if the window is blank
COLOR = (230, 216, 173)   # BGR (light sky blue)
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def get_connections():
    """Mesh connections; falls back to dots only if unavailable."""
    try:
        return vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    except AttributeError:
        return []


def main():
    ensure_model()
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    connections = get_connections()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) if os.name == "nt" \
        else cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        timestamp_ms = 0
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame - check camera index")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            for face in result.face_landmarks:
                pts = [(int(p.x * w), int(p.y * h)) for p in face]
                for c in connections:
                    cv2.line(frame, pts[c.start], pts[c.end], COLOR, 1)
                if not connections:
                    for x, y in pts:
                        cv2.circle(frame, (x, y), 1, COLOR, -1)

            cv2.imshow("MediaPipe Face Mesh", cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

CAMERA_INDEX = 0  # change if blank
COLOR = (230, 216, 173)  # BGR

options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Mesh connections (falls back to dots only if unavailable in your version)
try:
    CONNECTIONS = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
except AttributeError:
    CONNECTIONS = []

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    timestamp_ms = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame.shape[:2]
        for face in result.face_landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in face]
            for c in CONNECTIONS:
                cv2.line(frame, pts[c.start], pts[c.end], COLOR, 1)
            if not CONNECTIONS:
                for x, y in pts:
                    cv2.circle(frame, (x, y), 1, COLOR, -1)

        cv2.imshow("MediaPipe Face Mesh", cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
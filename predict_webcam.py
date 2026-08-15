import cv2
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_extractor import extract_features


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "yoga_pose_model.pkl"
LABEL_MAPPING_PATH = "label_mapping.pkl"

model = joblib.load(MODEL_PATH)

label_mapping = joblib.load(
    LABEL_MAPPING_PATH
)

# Convert:
# plank -> 0
# tree -> 1
# warrior_ii -> 2
#
# into:
# 0 -> plank
# 1 -> tree
# 2 -> warrior_ii

reverse_mapping = {
    value: key
    for key, value in label_mapping.items()
}


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(
    options
)


# ============================================================
# MEDIAPIPE POSE CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [
    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Shoulders
    (11, 12),

    # Left side body
    (11, 23),
    (23, 25),
    (25, 27),

    # Right side body
    (12, 24),
    (24, 26),
    (26, 28),

    # Hips
    (23, 24),

    # Left leg
    (27, 29),
    (29, 31),

    # Right leg
    (28, 30),
    (30, 32),

    # Feet
    (27, 31),
    (28, 32)
]


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


# ============================================================
# TIMESTAMP
# ============================================================

frame_timestamp = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print("Could not read webcam frame.")

        break


    # --------------------------------------------------------
    # Mirror webcam
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # Get frame dimensions

    height, width, _ = frame.shape


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Create MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    frame_timestamp += 33


    # --------------------------------------------------------
    # Detect pose
    # --------------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )


    # ========================================================
    # IF POSE FOUND
    # ========================================================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]


        # ====================================================
        # DRAW LANDMARK DOTS
        # ====================================================

        for landmark in landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )


            # Make sure point is inside frame

            if (
                0 <= x < width
                and
                0 <= y < height
            ):

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


        # ====================================================
        # DRAW SKELETON
        # ====================================================

        for start, end in POSE_CONNECTIONS:

            start_landmark = landmarks[start]
            end_landmark = landmarks[end]


            start_x = int(
                start_landmark.x * width
            )

            start_y = int(
                start_landmark.y * height
            )


            end_x = int(
                end_landmark.x * width
            )

            end_y = int(
                end_landmark.y * height
            )


            # Make sure both points are visible

            if (
                0 <= start_x < width
                and
                0 <= start_y < height
                and
                0 <= end_x < width
                and
                0 <= end_y < height
            ):

                cv2.line(
                    frame,
                    (start_x, start_y),
                    (end_x, end_y),
                    (255, 255, 255),
                    2
                )


        # ====================================================
        # EXTRACT FEATURES
        # ====================================================

        features = extract_features(
            landmarks
        )


        # ====================================================
        # PREPARE MODEL INPUT
        # ====================================================

        X = np.array(
            features
        ).reshape(
            1,
            -1
        )


        # ====================================================
        # PREDICT POSE
        # ====================================================

        prediction = model.predict(X)[0]


        pose_name = reverse_mapping[
            int(prediction)
        ]


        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                X
            )[0]

            confidence = float(
                np.max(probabilities)
            )


        # ====================================================
        # DISPLAY POSE
        # ====================================================

        display_name = pose_name.replace(
            "_",
            " "
        ).upper()


        cv2.putText(
            frame,
            f"Pose: {display_name}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY CONFIDENCE
        # ====================================================

        if confidence is not None:

            confidence_text = (
                f"Confidence: "
                f"{confidence * 100:.1f}%"
            )


            cv2.putText(
                frame,
                confidence_text,
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


    # ========================================================
    # NO POSE
    # ========================================================

    else:

        cv2.putText(
            frame,
            "No pose detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AI Yoga Pose Analyzer",
        frame
    )


    # ========================================================
    # PRESS Q TO EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_smoother import PoseSmoother
from pose_smoother import landmarks_are_visible

from features import extract_features


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "pose_landmarker.task"

CSV_FILE = "dataset.csv"


# ============================================================
# LABELS
# ============================================================

LABELS = {

    "w": "Warrior II",

    "t": "Tree Pose",

    "p": "Plank",

    "s": "Standing"
}


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
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
# SMOOTHER
# ============================================================

smoother = PoseSmoother(
    alpha=0.5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Cannot open webcam.")

    exit()


# ============================================================
# CSV
# ============================================================

feature_names = [

    "left_elbow_angle",

    "right_elbow_angle",

    "left_knee_angle",

    "right_knee_angle",

    "left_body_angle",

    "right_body_angle",

    "wrist_ratio",

    "ankle_ratio",

    "hip_ratio",

    "body_ratio"
]


# Create CSV if it doesn't exist

file_exists = os.path.exists(CSV_FILE)


csv_file = open(
    CSV_FILE,
    "a",
    newline=""
)


writer = csv.writer(
    csv_file
)


if not file_exists:

    writer.writerow(
        feature_names + ["label"]
    )


# ============================================================
# STATE
# ============================================================

current_label = None

timestamp_ms = 0

sample_count = 0


print()
print("========================================")
print("       AI YOGA DATA COLLECTOR")
print("========================================")
print()
print("W = Warrior II")
print("T = Tree Pose")
print("P = Plank")
print("S = Standing")
print("Q = Quit")
print()
print("Press a key to select a label.")
print("Then hold that pose.")
print()


# ============================================================
# LOOP
# ============================================================

while cap.isOpened():

    success, frame = cap.read()


    if not success:

        print("Could not read webcam.")

        break


    # ========================================================
    # RGB
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp_ms += 33


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # PERSON DETECTED
    # ========================================================

    if len(result.pose_landmarks) > 0:

        landmarks = result.pose_landmarks[0]


        # Smooth

        landmarks = smoother.smooth(
            landmarks
        )


        # ====================================================
        # VISIBILITY
        # ====================================================

        visible = landmarks_are_visible(
            landmarks,
            threshold=0.5
        )


        if visible:

            # =================================================
            # EXTRACT FEATURES
            # =================================================

            features = extract_features(
                landmarks
            )


            # =================================================
            # DRAW SKELETON
            # =================================================

            connections = [

                (11, 12),

                (11, 13),
                (13, 15),

                (12, 14),
                (14, 16),

                (11, 23),
                (12, 24),
                (23, 24),

                (23, 25),
                (25, 27),

                (24, 26),
                (26, 28)
            ]


            height, width, _ = frame.shape


            for start, end in connections:

                p1 = landmarks[start]

                p2 = landmarks[end]


                x1 = int(p1.x * width)
                y1 = int(p1.y * height)

                x2 = int(p2.x * width)
                y2 = int(p2.y * height)


                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            for landmark in landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )


                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            # =================================================
            # RECORD DATA
            # =================================================

            if current_label is not None:

                row = [

                    features["left_elbow_angle"],

                    features["right_elbow_angle"],

                    features["left_knee_angle"],

                    features["right_knee_angle"],

                    features["left_body_angle"],

                    features["right_body_angle"],

                    features["wrist_ratio"],

                    features["ankle_ratio"],

                    features["hip_ratio"],

                    features["body_ratio"],

                    current_label
                ]


                writer.writerow(row)

                csv_file.flush()

                sample_count += 1


            # =================================================
            # DISPLAY LABEL
            # =================================================

            if current_label is None:

                label_text = (
                    "No label selected"
                )

                label_color = (
                    0,
                    0,
                    255
                )

            else:

                label_text = (
                    f"Recording: {current_label}"
                )

                label_color = (
                    0,
                    255,
                    0
                )


            cv2.putText(
                frame,
                label_text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                label_color,
                2
            )


            # =================================================
            # SAMPLE COUNT
            # =================================================

            cv2.putText(
                frame,
                f"Samples: {sample_count}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        else:

            cv2.putText(
                frame,
                "Move fully into camera view",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


    else:

        cv2.putText(
            frame,
            "No person detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Yoga Dataset Collector",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("w"):

        current_label = "Warrior II"

        print(
            "Recording Warrior II..."
        )


    elif key == ord("t"):

        current_label = "Tree Pose"

        print(
            "Recording Tree Pose..."
        )


    elif key == ord("p"):

        current_label = "Plank"

        print(
            "Recording Plank..."
        )


    elif key == ord("s"):

        current_label = "Standing"

        print(
            "Recording Standing..."
        )


    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

csv_file.close()

cap.release()

cv2.destroyAllWindows()


print()
print("========================================")
print("DATA COLLECTION COMPLETE")
print("========================================")
print(
    f"Total samples: {sample_count}"
)
print(
    f"Dataset saved to: {CSV_FILE}"
)
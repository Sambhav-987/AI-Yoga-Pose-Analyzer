import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_analyzer import detect_pose

from pose_smoother import (
    PoseSmoother,
    landmarks_are_visible
)

from features import extract_features


# ============================================================
# 1. LOAD MEDIAPIPE MODEL
# ============================================================

MODEL_PATH = "pose_landmarker.task"

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
# 2. CREATE SMOOTHER
# ============================================================

smoother = PoseSmoother(
    alpha=0.5
)


# ============================================================
# 3. SKELETON CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [

    # Shoulders
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Right leg
    (24, 26),
    (26, 28)
]


# ============================================================
# 4. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


# ============================================================
# 5. TIMESTAMP
# ============================================================

timestamp_ms = 0


# ============================================================
# 6. MAIN LOOP
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read webcam.")

        break


    # ========================================================
    # BGR → RGB
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # MEDIA PIPE IMAGE
    # ========================================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp_ms += 33


    # ========================================================
    # DETECT POSE
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


        # ====================================================
        # SMOOTH LANDMARKS
        # ====================================================

        landmarks = smoother.smooth(
            landmarks
        )


        height, width, _ = frame.shape


        # ====================================================
        # DRAW LANDMARKS
        # ====================================================

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


            cv2.line(
                frame,
                (start_x, start_y),
                (end_x, end_y),
                (255, 255, 255),
                2
            )


        # ====================================================
        # CHECK LANDMARK VISIBILITY
        # ====================================================

        if not landmarks_are_visible(
            landmarks,
            threshold=0.5
        ):

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

            # =================================================
            # POSE ANALYSIS
            # =================================================

            pose_result = detect_pose(
                landmarks
            )


            pose_name = pose_result["pose"]

            score = pose_result["score"]

            feedback = pose_result["feedback"]


            # =================================================
            # FEATURE EXTRACTION
            # =================================================

            features = extract_features(
                landmarks
            )


            # =================================================
            # DISPLAY POSE
            # =================================================

            if score >= 80:

                color = (0, 255, 0)

            elif score >= 60:

                color = (0, 255, 255)

            else:

                color = (0, 0, 255)


            cv2.putText(
                frame,
                pose_name,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )


            # =================================================
            # SCORE
            # =================================================

            cv2.putText(
                frame,
                f"Score: {score}/100",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


            # =================================================
            # DISPLAY FEATURES
            # =================================================

            y = 115


            feature_display = [

                (
                    "L Knee",
                    features["left_knee_angle"]
                ),

                (
                    "R Knee",
                    features["right_knee_angle"]
                ),

                (
                    "L Elbow",
                    features["left_elbow_angle"]
                ),

                (
                    "R Elbow",
                    features["right_elbow_angle"]
                ),

                (
                    "L Body",
                    features["left_body_angle"]
                ),

                (
                    "R Body",
                    features["right_body_angle"]
                ),

                (
                    "Body Ratio",
                    features["body_ratio"]
                )
            ]


            for name, value in feature_display:

                text = f"{name}: {value:.2f}"


                cv2.putText(
                    frame,
                    text,
                    (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )


                y += 27


            # =================================================
            # FEEDBACK
            # =================================================

            if len(feedback) > 0:

                feedback_text = feedback[0]

                feedback_color = (0, 0, 255)

            else:

                feedback_text = "Excellent Form!"

                feedback_color = (0, 255, 0)


            cv2.putText(
                frame,
                feedback_text,
                (30, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                feedback_color,
                2
            )


    # ========================================================
    # NO PERSON
    # ========================================================

    else:

        cv2.putText(
            frame,
            "No Human Detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )


    # ========================================================
    # SHOW WINDOW
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
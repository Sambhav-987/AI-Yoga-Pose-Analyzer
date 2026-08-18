import cv2
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_extractor import extract_features
from form_analyzer import analyze_form


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "yoga_pose_model_v3.pkl"
LABEL_MAPPING_PATH = "label_mapping_v3.pkl"


model = joblib.load(MODEL_PATH)

label_mapping = joblib.load(
    LABEL_MAPPING_PATH
)

reverse_mapping = {
    value: key
    for key, value in label_mapping.items()
}


# ============================================================
# MEDIAPIPE
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
# POSE CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [
    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 12),

    (11, 23),
    (23, 25),
    (25, 27),

    (12, 24),
    (24, 26),
    (26, 28),

    (23, 24)
]


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# PREDICTION SMOOTHING
# ============================================================

prediction_history = []

HISTORY_SIZE = 10


# ============================================================
# TIMESTAMP
# ============================================================

timestamp = 0


# ============================================================
# FORM SCORE SMOOTHING
# ============================================================

score_history = []

SCORE_HISTORY_SIZE = 8


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

    height, width, _ = frame.shape


    # --------------------------------------------------------
    # Convert image
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp += 33


    # --------------------------------------------------------
    # Detect pose
    # --------------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # POSE DETECTED
    # ========================================================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]


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

            if (
                0 <= x < width
                and
                0 <= y < height
            ):

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


        # ====================================================
        # DRAW SKELETON
        # ====================================================

        for start, end in POSE_CONNECTIONS:

            p1 = landmarks[start]
            p2 = landmarks[end]

            x1 = int(p1.x * width)
            y1 = int(p1.y * height)

            x2 = int(p2.x * width)
            y2 = int(p2.y * height)

            if (
                0 <= x1 < width
                and
                0 <= y1 < height
                and
                0 <= x2 < width
                and
                0 <= y2 < height
            ):

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )


        # ====================================================
        # EXTRACT FEATURES
        # ====================================================

        features = extract_features(
            landmarks
        )


        X = np.array(
            features,
            dtype=float
        ).reshape(
            1,
            -1
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        confidence = float(
            np.max(probabilities)
        )


        # ====================================================
        # STORE PREDICTION
        # ====================================================

        prediction_history.append(
            int(prediction)
        )


        if len(prediction_history) > HISTORY_SIZE:

            prediction_history.pop(0)


        # ====================================================
        # MAJORITY VOTE
        # ====================================================

        counts = np.bincount(
            prediction_history,
            minlength=len(label_mapping)
        )

        stable_prediction = int(
            np.argmax(counts)
        )


        pose_name = reverse_mapping[
            stable_prediction
        ]


        # ====================================================
        # UNKNOWN THRESHOLD
        # ====================================================

        if confidence < 0.60:

            pose_name = "unknown"


        display_name = pose_name.replace(
            "_",
            " "
        ).upper()


        # ====================================================
        # POSE DISPLAY
        # ====================================================

        cv2.putText(
            frame,
            f"POSE: {display_name}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        cv2.putText(
            frame,
            f"CONFIDENCE: {confidence * 100:.1f}%",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ====================================================
        # FORM ANALYSIS
        # ====================================================

        if pose_name in [
            "plank",
            "tree",
            "warrior_ii"
        ]:

            form_result = analyze_form(
                pose_name,
                landmarks
            )

            current_score = form_result[
                "score"
            ]

            feedback = form_result[
                "feedback"
            ]


            # =================================================
            # SCORE SMOOTHING
            # =================================================

            score_history.append(
                current_score
            )


            if len(score_history) > SCORE_HISTORY_SIZE:

                score_history.pop(0)


            form_score = int(
                round(
                    sum(score_history)
                    /
                    len(score_history)
                )
            )


            # =================================================
            # FORM SCORE
            # =================================================

            cv2.putText(
                frame,
                f"FORM SCORE: {form_score}/100",
                (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


            # =================================================
            # FEEDBACK
            # =================================================

            y_position = 165


            for message in feedback[:2]:

                cv2.putText(
                    frame,
                    message,
                    (25, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                y_position += 30


        else:

            score_history.clear()


            cv2.putText(
                frame,
                "FORM: No yoga pose",
                (25, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


    # ========================================================
    # NO POSE
    # ========================================================

    else:

        prediction_history.clear()

        score_history.clear()


        cv2.putText(
            frame,
            "NO POSE DETECTED",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2
        )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "Press Q to quit",
        (25, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "AI Yoga Pose Analyzer",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()
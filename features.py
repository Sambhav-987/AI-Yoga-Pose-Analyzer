import numpy as np


# ============================================================
# DISTANCE BETWEEN TWO LANDMARKS
# ============================================================

def calculate_distance(a, b):

    return np.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


# ============================================================
# ANGLE BETWEEN THREE LANDMARKS
# ============================================================

def calculate_angle(a, b, c):

    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return 0

    cosine = np.dot(ba, bc) / denominator

    cosine = np.clip(
        cosine,
        -1,
        1
    )

    return np.degrees(
        np.arccos(cosine)
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(landmarks):

    # --------------------------------------------------------
    # Important landmarks
    # --------------------------------------------------------

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_elbow = landmarks[13]
    right_elbow = landmarks[14]

    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    left_hip = landmarks[23]
    right_hip = landmarks[24]

    left_knee = landmarks[25]
    right_knee = landmarks[26]

    left_ankle = landmarks[27]
    right_ankle = landmarks[28]


    # ========================================================
    # 1. JOINT ANGLES
    # ========================================================

    left_elbow_angle = calculate_angle(
        left_shoulder,
        left_elbow,
        left_wrist
    )

    right_elbow_angle = calculate_angle(
        right_shoulder,
        right_elbow,
        right_wrist
    )


    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )


    # ========================================================
    # 2. BODY ANGLES
    # ========================================================

    left_body_angle = calculate_angle(
        left_shoulder,
        left_hip,
        left_knee
    )

    right_body_angle = calculate_angle(
        right_shoulder,
        right_hip,
        right_knee
    )


    # ========================================================
    # 3. DISTANCES
    # ========================================================

    shoulder_width = calculate_distance(
        left_shoulder,
        right_shoulder
    )

    hip_width = calculate_distance(
        left_hip,
        right_hip
    )

    wrist_distance = calculate_distance(
        left_wrist,
        right_wrist
    )

    ankle_distance = calculate_distance(
        left_ankle,
        right_ankle
    )


    # ========================================================
    # 4. NORMALIZED DISTANCES
    # ========================================================

    # We divide by shoulder width so the feature doesn't
    # depend heavily on how close the person is to the camera.

    if shoulder_width > 0:

        wrist_ratio = (
            wrist_distance /
            shoulder_width
        )

        ankle_ratio = (
            ankle_distance /
            shoulder_width
        )

        hip_ratio = (
            hip_width /
            shoulder_width
        )

    else:

        wrist_ratio = 0
        ankle_ratio = 0
        hip_ratio = 0


    # ========================================================
    # 5. BODY ASPECT RATIO
    # ========================================================

    points = [
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle
    ]


    xs = [point.x for point in points]
    ys = [point.y for point in points]


    body_width = max(xs) - min(xs)

    body_height = max(ys) - min(ys)


    if body_height > 0:

        body_ratio = (
            body_width /
            body_height
        )

    else:

        body_ratio = 0


    # ========================================================
    # 6. RETURN FEATURES
    # ========================================================

    features = {

        "left_elbow_angle":
            left_elbow_angle,

        "right_elbow_angle":
            right_elbow_angle,

        "left_knee_angle":
            left_knee_angle,

        "right_knee_angle":
            right_knee_angle,

        "left_body_angle":
            left_body_angle,

        "right_body_angle":
            right_body_angle,

        "wrist_ratio":
            wrist_ratio,

        "ankle_ratio":
            ankle_ratio,

        "hip_ratio":
            hip_ratio,

        "body_ratio":
            body_ratio
    }


    return features
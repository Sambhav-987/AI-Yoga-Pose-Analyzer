import numpy as np


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ============================================================
# CALCULATE ANGLE
# ============================================================

def calculate_angle(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    radians = (
        np.arctan2(
            c[1] - b[1],
            c[0] - b[0]
        )
        -
        np.arctan2(
            a[1] - b[1],
            a[0] - b[0]
        )
    )

    angle = abs(
        radians * 180.0 / np.pi
    )

    if angle > 180:
        angle = 360 - angle

    return angle


# ============================================================
# DISTANCE
# ============================================================

def calculate_distance(a, b):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    return np.linalg.norm(a - b)


# ============================================================
# POINT
# ============================================================

def point(landmark):

    return [
        landmark.x,
        landmark.y
    ]


# ============================================================
# DIRECTION ANGLE
#
# Calculates the angle of a line relative to the
# horizontal x-axis.
#
# Example:
#
# horizontal line → approximately 0°
# vertical line   → approximately 90°
# ============================================================

def calculate_direction_angle(a, b):

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    return abs(angle)


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(landmarks):

    # --------------------------------------------------------
    # Get landmarks
    # --------------------------------------------------------

    left_shoulder = point(
        landmarks[LEFT_SHOULDER]
    )

    right_shoulder = point(
        landmarks[RIGHT_SHOULDER]
    )

    left_elbow = point(
        landmarks[LEFT_ELBOW]
    )

    right_elbow = point(
        landmarks[RIGHT_ELBOW]
    )

    left_wrist = point(
        landmarks[LEFT_WRIST]
    )

    right_wrist = point(
        landmarks[RIGHT_WRIST]
    )

    left_hip = point(
        landmarks[LEFT_HIP]
    )

    right_hip = point(
        landmarks[RIGHT_HIP]
    )

    left_knee = point(
        landmarks[LEFT_KNEE]
    )

    right_knee = point(
        landmarks[RIGHT_KNEE]
    )

    left_ankle = point(
        landmarks[LEFT_ANKLE]
    )

    right_ankle = point(
        landmarks[RIGHT_ANKLE]
    )


    # ========================================================
    # 1–8 : JOINT ANGLES
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

    left_shoulder_angle = calculate_angle(
        left_elbow,
        left_shoulder,
        left_hip
    )

    right_shoulder_angle = calculate_angle(
        right_elbow,
        right_shoulder,
        right_hip
    )

    left_hip_angle = calculate_angle(
        left_shoulder,
        left_hip,
        left_knee
    )

    right_hip_angle = calculate_angle(
        right_shoulder,
        right_hip,
        right_knee
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
    # 9–12 : EXISTING DISTANCE FEATURES
    # ========================================================

    shoulder_distance = calculate_distance(
        left_shoulder,
        right_shoulder
    )

    hip_distance = calculate_distance(
        left_hip,
        right_hip
    )

    left_hand_to_hip = calculate_distance(
        left_wrist,
        left_hip
    )

    right_hand_to_hip = calculate_distance(
        right_wrist,
        right_hip
    )


    # ========================================================
    # 13 : TORSO ORIENTATION
    # ========================================================

    # Calculate the midpoint of both shoulders.

    shoulder_center = [
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    ]


    # Calculate the midpoint of both hips.

    hip_center = [
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    ]


    # Angle of torso relative to horizontal.

    torso_angle = calculate_direction_angle(
        shoulder_center,
        hip_center
    )


    # ========================================================
    # 14 : LEFT LEG ORIENTATION
    # ========================================================

    left_leg_angle = calculate_direction_angle(
        left_hip,
        left_ankle
    )


    # ========================================================
    # 15 : RIGHT LEG ORIENTATION
    # ========================================================

    right_leg_angle = calculate_direction_angle(
        right_hip,
        right_ankle
    )


    # ========================================================
    # 16 : BODY VERTICAL SPAN
    # ========================================================

    all_x = [
        left_shoulder[0],
        right_shoulder[0],
        left_hip[0],
        right_hip[0],
        left_knee[0],
        right_knee[0],
        left_ankle[0],
        right_ankle[0]
    ]

    all_y = [
        left_shoulder[1],
        right_shoulder[1],
        left_hip[1],
        right_hip[1],
        left_knee[1],
        right_knee[1],
        left_ankle[1],
        right_ankle[1]
    ]

    vertical_span = max(all_y) - min(all_y)


    # ========================================================
    # 17 : BODY HORIZONTAL SPAN
    # ========================================================

    horizontal_span = max(all_x) - min(all_x)


    # ========================================================
    # 18 : BODY ASPECT RATIO
    # ========================================================

    if horizontal_span > 0:

        body_aspect_ratio = (
            vertical_span /
            horizontal_span
        )

    else:

        body_aspect_ratio = 0.0


    # ========================================================
    # RETURN ALL 18 FEATURES
    # ========================================================

    return [

        # 1–8
        left_elbow_angle,
        right_elbow_angle,

        left_shoulder_angle,
        right_shoulder_angle,

        left_hip_angle,
        right_hip_angle,

        left_knee_angle,
        right_knee_angle,

        # 9–12
        shoulder_distance,
        hip_distance,

        left_hand_to_hip,
        right_hand_to_hip,

        # 13–18
        torso_angle,
        left_leg_angle,
        right_leg_angle,

        vertical_span,
        horizontal_span,
        body_aspect_ratio
    ]
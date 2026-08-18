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
# HELPER FUNCTIONS
# ============================================================

def get_point(landmarks, index):
    """
    Convert a MediaPipe landmark into an (x, y) point.
    """

    landmark = landmarks[index]

    return np.array([
        landmark.x,
        landmark.y
    ], dtype=float)


def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees.
    """

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return 0.0

    cosine_angle = np.dot(
        ba,
        bc
    ) / denominator

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle


def calculate_distance(a, b):
    """
    Calculate Euclidean distance.
    """

    return np.linalg.norm(
        np.array(a) - np.array(b)
    )


# ============================================================
# PLANK ANALYSIS
# ============================================================

def analyze_plank(landmarks):

    feedback = []

    scores = []


    # --------------------------------------------------------
    # Get points
    # --------------------------------------------------------

    left_shoulder = get_point(
        landmarks,
        LEFT_SHOULDER
    )

    right_shoulder = get_point(
        landmarks,
        RIGHT_SHOULDER
    )

    left_hip = get_point(
        landmarks,
        LEFT_HIP
    )

    right_hip = get_point(
        landmarks,
        RIGHT_HIP
    )

    left_knee = get_point(
        landmarks,
        LEFT_KNEE
    )

    right_knee = get_point(
        landmarks,
        RIGHT_KNEE
    )

    left_ankle = get_point(
        landmarks,
        LEFT_ANKLE
    )

    right_ankle = get_point(
        landmarks,
        RIGHT_ANKLE
    )


    # --------------------------------------------------------
    # Hip alignment
    # --------------------------------------------------------

    left_body_angle = calculate_angle(
        left_shoulder,
        left_hip,
        left_ankle
    )

    right_body_angle = calculate_angle(
        right_shoulder,
        right_hip,
        right_ankle
    )

    average_body_angle = (
        left_body_angle
        +
        right_body_angle
    ) / 2


    # Good plank should have a relatively
    # straight shoulder-hip-ankle line.

    if average_body_angle >= 160:

        scores.append(100)

    elif average_body_angle >= 150:

        scores.append(90)

        feedback.append(
            "Keep your body straighter"
        )

    elif average_body_angle >= 135:

        scores.append(70)

        feedback.append(
            "Avoid dropping or raising your hips"
        )

    else:

        scores.append(45)

        feedback.append(
            "Keep your hips aligned with your shoulders"
        )


    # --------------------------------------------------------
    # Knee alignment
    # --------------------------------------------------------

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

    average_knee_angle = (
        left_knee_angle
        +
        right_knee_angle
    ) / 2


    if average_knee_angle >= 165:

        scores.append(100)

    elif average_knee_angle >= 150:

        scores.append(85)

        feedback.append(
            "Keep your legs more extended"
        )

    else:

        scores.append(60)

        feedback.append(
            "Straighten your legs"
        )


    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    form_score = int(
        round(
            sum(scores) / len(scores)
        )
    )


    # --------------------------------------------------------
    # Default feedback
    # --------------------------------------------------------

    if not feedback:

        feedback.append(
            "Great plank form!"
        )


    return {
        "score": form_score,
        "feedback": feedback
    }


# ============================================================
# TREE ANALYSIS
# ============================================================

def analyze_tree(landmarks):

    feedback = []

    scores = []


    # --------------------------------------------------------
    # Get points
    # --------------------------------------------------------

    left_shoulder = get_point(
        landmarks,
        LEFT_SHOULDER
    )

    right_shoulder = get_point(
        landmarks,
        RIGHT_SHOULDER
    )

    left_hip = get_point(
        landmarks,
        LEFT_HIP
    )

    right_hip = get_point(
        landmarks,
        RIGHT_HIP
    )

    left_knee = get_point(
        landmarks,
        LEFT_KNEE
    )

    right_knee = get_point(
        landmarks,
        RIGHT_KNEE
    )

    left_ankle = get_point(
        landmarks,
        LEFT_ANKLE
    )

    right_ankle = get_point(
        landmarks,
        RIGHT_ANKLE
    )


    # --------------------------------------------------------
    # Torso alignment
    # --------------------------------------------------------

    shoulder_center = (
        left_shoulder
        +
        right_shoulder
    ) / 2

    hip_center = (
        left_hip
        +
        right_hip
    ) / 2


    torso_dx = abs(
        shoulder_center[0]
        -
        hip_center[0]
    )


    # Smaller horizontal displacement
    # means a more upright torso.

    if torso_dx < 0.05:

        scores.append(100)

    elif torso_dx < 0.10:

        scores.append(85)

        feedback.append(
            "Keep your torso more upright"
        )

    else:

        scores.append(60)

        feedback.append(
            "Straighten your upper body"
        )


    # --------------------------------------------------------
    # Supporting leg
    # --------------------------------------------------------

    left_leg_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_leg_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )


    # One leg should generally remain extended.

    best_leg_angle = max(
        left_leg_angle,
        right_leg_angle
    )


    if best_leg_angle >= 165:

        scores.append(100)

    elif best_leg_angle >= 150:

        scores.append(85)

        feedback.append(
            "Keep your supporting leg straighter"
        )

    else:

        scores.append(60)

        feedback.append(
            "Straighten your supporting leg"
        )


    # --------------------------------------------------------
    # Raised knee
    # --------------------------------------------------------

    left_knee_distance = calculate_distance(
        left_knee,
        left_hip
    )

    right_knee_distance = calculate_distance(
        right_knee,
        right_hip
    )


    # We mainly check that the two legs
    # are not completely identical.

    difference = abs(
        left_knee_distance
        -
        right_knee_distance
    )


    if difference > 0.05:

        scores.append(100)

    else:

        scores.append(80)

        feedback.append(
            "Lift one leg into the tree position"
        )


    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    form_score = int(
        round(
            sum(scores) / len(scores)
        )
    )


    if not feedback:

        feedback.append(
            "Great tree pose form!"
        )


    return {
        "score": form_score,
        "feedback": feedback
    }


# ============================================================
# WARRIOR II ANALYSIS
# ============================================================

def analyze_warrior_ii(landmarks):

    feedback = []

    scores = []


    # --------------------------------------------------------
    # Get points
    # --------------------------------------------------------

    left_shoulder = get_point(
        landmarks,
        LEFT_SHOULDER
    )

    right_shoulder = get_point(
        landmarks,
        RIGHT_SHOULDER
    )

    left_elbow = get_point(
        landmarks,
        LEFT_ELBOW
    )

    right_elbow = get_point(
        landmarks,
        RIGHT_ELBOW
    )

    left_hip = get_point(
        landmarks,
        LEFT_HIP
    )

    right_hip = get_point(
        landmarks,
        RIGHT_HIP
    )

    left_knee = get_point(
        landmarks,
        LEFT_KNEE
    )

    right_knee = get_point(
        landmarks,
        RIGHT_KNEE
    )

    left_ankle = get_point(
        landmarks,
        LEFT_ANKLE
    )

    right_ankle = get_point(
        landmarks,
        RIGHT_ANKLE
    )


    # --------------------------------------------------------
    # Front knee
    # --------------------------------------------------------

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


    # One knee should generally be bent
    # while the other leg remains straighter.

    bent_knee = min(
        left_knee_angle,
        right_knee_angle
    )

    straight_knee = max(
        left_knee_angle,
        right_knee_angle
    )


    if bent_knee <= 115:

        scores.append(100)

    elif bent_knee <= 135:

        scores.append(85)

        feedback.append(
            "Bend your front knee more"
        )

    else:

        scores.append(60)

        feedback.append(
            "Bend your front knee"
        )


    # --------------------------------------------------------
    # Back leg
    # --------------------------------------------------------

    if straight_knee >= 155:

        scores.append(100)

    elif straight_knee >= 140:

        scores.append(85)

        feedback.append(
            "Straighten your back leg"
        )

    else:

        scores.append(60)

        feedback.append(
            "Keep your back leg straight"
        )


    # --------------------------------------------------------
    # Arms
    # --------------------------------------------------------

    left_arm_angle = calculate_angle(
        left_shoulder,
        left_elbow,
        [
            left_elbow[0] + 0.1,
            left_elbow[1]
        ]
    )

    right_arm_angle = calculate_angle(
        right_shoulder,
        right_elbow,
        [
            right_elbow[0] - 0.1,
            right_elbow[1]
        ]
    )


    if (
        left_arm_angle > 150
        and
        right_arm_angle > 150
    ):

        scores.append(100)

    else:

        scores.append(75)

        feedback.append(
            "Extend both arms fully"
        )


    # --------------------------------------------------------
    # Shoulder alignment
    # --------------------------------------------------------

    shoulder_difference = abs(
        left_shoulder[1]
        -
        right_shoulder[1]
    )


    if shoulder_difference < 0.05:

        scores.append(100)

    elif shoulder_difference < 0.10:

        scores.append(85)

        feedback.append(
            "Keep your shoulders level"
        )

    else:

        scores.append(65)

        feedback.append(
            "Level your shoulders"
        )


    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    form_score = int(
        round(
            sum(scores) / len(scores)
        )
    )


    if not feedback:

        feedback.append(
            "Great Warrior II form!"
        )


    return {
        "score": form_score,
        "feedback": feedback
    }


# ============================================================
# MAIN FORM ANALYZER
# ============================================================

def analyze_form(
    pose_name,
    landmarks
):

    if pose_name == "plank":

        return analyze_plank(
            landmarks
        )


    elif pose_name == "tree":

        return analyze_tree(
            landmarks
        )


    elif pose_name == "warrior_ii":

        return analyze_warrior_ii(
            landmarks
        )


    else:

        return {
            "score": 0,
            "feedback": [
                "No yoga pose detected"
            ]
        }
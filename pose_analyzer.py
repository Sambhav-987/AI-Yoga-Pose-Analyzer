import numpy as np


# ============================================================
# ANGLE
# ============================================================

def calculate_angle(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return 0

    cosine = np.dot(ba, bc) / denominator

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# ============================================================
# BODY BOUNDING BOX
# ============================================================

def get_body_dimensions(landmarks):

    important_points = [
        11, 12,       # shoulders
        13, 14,       # elbows
        15, 16,       # wrists
        23, 24,       # hips
        25, 26,       # knees
        27, 28        # ankles
    ]

    xs = [
        landmarks[i].x
        for i in important_points
    ]

    ys = [
        landmarks[i].y
        for i in important_points
    ]

    width = max(xs) - min(xs)

    height = max(ys) - min(ys)

    return width, height


# ============================================================
# WARRIOR II
# ============================================================

def analyze_warrior_ii(landmarks):

    ls = landmarks[11]
    rs = landmarks[12]

    le = landmarks[13]
    re = landmarks[14]

    lw = landmarks[15]
    rw = landmarks[16]

    lh = landmarks[23]
    rh = landmarks[24]

    lk = landmarks[25]
    rk = landmarks[26]

    la = landmarks[27]
    ra = landmarks[28]


    left_knee = calculate_angle(
        [lh.x, lh.y],
        [lk.x, lk.y],
        [la.x, la.y]
    )

    right_knee = calculate_angle(
        [rh.x, rh.y],
        [rk.x, rk.y],
        [ra.x, ra.y]
    )

    left_elbow = calculate_angle(
        [ls.x, ls.y],
        [le.x, le.y],
        [lw.x, lw.y]
    )

    right_elbow = calculate_angle(
        [rs.x, rs.y],
        [re.x, re.y],
        [rw.x, rw.y]
    )


    score = 0
    feedback = []


    # One knee should be bent
    if (
        70 <= left_knee <= 115
        or
        70 <= right_knee <= 115
    ):

        score += 30

    else:

        feedback.append(
            "Bend one knee"
        )


    # One leg should be straight
    if (
        left_knee >= 155
        or
        right_knee >= 155
    ):

        score += 25

    else:

        feedback.append(
            "Keep one leg straight"
        )


    # Arms straight
    if (
        left_elbow >= 155
        and
        right_elbow >= 155
    ):

        score += 25

    else:

        feedback.append(
            "Straighten your arms"
        )


    # Arms should be spread
    arm_width = abs(
        lw.x - rw.x
    )

    shoulder_width = abs(
        ls.x - rs.x
    )

    if arm_width > shoulder_width * 1.3:

        score += 20

    else:

        feedback.append(
            "Extend your arms sideways"
        )


    return {
        "pose": "Warrior II",
        "score": score,
        "feedback": feedback,
        "angles": {
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow
        }
    }


# ============================================================
# TREE POSE
# ============================================================

def analyze_tree_pose(landmarks):

    lh = landmarks[23]
    rh = landmarks[24]

    lk = landmarks[25]
    rk = landmarks[26]

    la = landmarks[27]
    ra = landmarks[28]


    left_knee = calculate_angle(
        [lh.x, lh.y],
        [lk.x, lk.y],
        [la.x, la.y]
    )

    right_knee = calculate_angle(
        [rh.x, rh.y],
        [rk.x, rk.y],
        [ra.x, ra.y]
    )


    left_straight = left_knee > 160
    right_straight = right_knee > 160

    left_bent = left_knee < 130
    right_bent = right_knee < 130


    valid = (
        left_straight and right_bent
    ) or (
        right_straight and left_bent
    )


    if valid:

        score = 90

        feedback = []

    else:

        score = 0

        feedback = [
            "Lift one leg and bend the other"
        ]


    return {
        "pose": "Tree Pose",
        "score": score,
        "feedback": feedback,
        "angles": {
            "left_knee": left_knee,
            "right_knee": right_knee
        }
    }


# ============================================================
# PLANK
# ============================================================

def analyze_plank(landmarks):

    # --------------------------------------------------------
    # Calculate body bounding box
    # --------------------------------------------------------

    body_width, body_height = get_body_dimensions(
        landmarks
    )


    # Prevent division by zero

    if body_height == 0:

        return {
            "pose": "Plank",
            "score": 0,
            "feedback": [
                "Unable to determine body shape"
            ],
            "angles": {}
        }


    # ========================================================
    # MOST IMPORTANT TEST
    # ========================================================

    aspect_ratio = body_width / body_height


    # A standing person should be tall.
    #
    # A plank should be wide.
    #
    # Therefore we require a strongly horizontal
    # body bounding box.

    horizontal_body = aspect_ratio > 1.25


    # --------------------------------------------------------
    # If the body is not horizontal,
    # PLANK IS IMMEDIATELY INVALID.
    # --------------------------------------------------------

    if not horizontal_body:

        return {
            "pose": "Plank",
            "score": 0,
            "feedback": [
                "Body is not horizontal"
            ],
            "angles": {
                "body_ratio": aspect_ratio
            }
        }


    # ========================================================
    # Get landmarks
    # ========================================================

    ls = landmarks[11]
    rs = landmarks[12]

    le = landmarks[13]
    re = landmarks[14]

    lw = landmarks[15]
    rw = landmarks[16]

    lh = landmarks[23]
    rh = landmarks[24]

    lk = landmarks[25]
    rk = landmarks[26]

    la = landmarks[27]
    ra = landmarks[28]


    # ========================================================
    # KNEES
    # ========================================================

    left_knee = calculate_angle(
        [lh.x, lh.y],
        [lk.x, lk.y],
        [la.x, la.y]
    )

    right_knee = calculate_angle(
        [rh.x, rh.y],
        [rk.x, rk.y],
        [ra.x, ra.y]
    )


    # ========================================================
    # ELBOWS
    # ========================================================

    left_elbow = calculate_angle(
        [ls.x, ls.y],
        [le.x, le.y],
        [lw.x, lw.y]
    )

    right_elbow = calculate_angle(
        [rs.x, rs.y],
        [re.x, re.y],
        [rw.x, rw.y]
    )


    # ========================================================
    # BODY ALIGNMENT
    # ========================================================

    left_body = calculate_angle(
        [ls.x, ls.y],
        [lh.x, lh.y],
        [lk.x, lk.y]
    )

    right_body = calculate_angle(
        [rs.x, rs.y],
        [rh.x, rh.y],
        [rk.x, rk.y]
    )


    score = 0
    feedback = []


    # ========================================================
    # BODY SHAPE
    # ========================================================

    score += 30


    # ========================================================
    # BODY STRAIGHT
    # ========================================================

    if (
        left_body >= 150
        and
        right_body >= 150
    ):

        score += 30

    else:

        feedback.append(
            "Keep your body straight"
        )


    # ========================================================
    # LEGS STRAIGHT
    # ========================================================

    if (
        left_knee >= 150
        and
        right_knee >= 150
    ):

        score += 20

    else:

        feedback.append(
            "Straighten both legs"
        )


    # ========================================================
    # ARMS
    # ========================================================

    if (
        left_elbow >= 140
        and
        right_elbow >= 140
    ):

        score += 20

    else:

        feedback.append(
            "Position your arms correctly"
        )


    return {
        "pose": "Plank",
        "score": score,
        "feedback": feedback,
        "angles": {
            "body_ratio": aspect_ratio,
            "left_body": left_body,
            "right_body": right_body,
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow
        }
    }


# ============================================================
# FINAL POSE DETECTOR
# ============================================================

def detect_pose(landmarks):

    warrior = analyze_warrior_ii(
        landmarks
    )

    tree = analyze_tree_pose(
        landmarks
    )

    plank = analyze_plank(
        landmarks
    )


    # ========================================================
    # VALID POSES
    # ========================================================

    valid = []


    if warrior["score"] >= 70:

        valid.append(warrior)


    if tree["score"] >= 80:

        valid.append(tree)


    if plank["score"] >= 70:

        valid.append(plank)


    # ========================================================
    # NO VALID POSE
    # ========================================================

    if len(valid) == 0:

        return {
            "pose": "Unknown Pose",
            "score": 0,
            "feedback": [
                "Pose not recognized"
            ],
            "angles": {}
        }


    # ========================================================
    # BEST VALID POSE
    # ========================================================

    return max(
        valid,
        key=lambda result: result["score"]
    )
import os
import cv2
import csv
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_extractor import extract_features


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset/images"
MODEL_PATH = "pose_landmarker.task"
OUTPUT_FILE = "dataset.csv"


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(
    options
)


# ============================================================
# CSV HEADER
# ============================================================

header = [

    # Joint angles
    "left_elbow_angle",
    "right_elbow_angle",

    "left_shoulder_angle",
    "right_shoulder_angle",

    "left_hip_angle",
    "right_hip_angle",

    "left_knee_angle",
    "right_knee_angle",

    # Distances
    "shoulder_distance",
    "hip_distance",

    "left_hand_to_hip",
    "right_hand_to_hip",

    # Body orientation
    "torso_angle",
    "left_leg_angle",
    "right_leg_angle",

    "vertical_span",
    "horizontal_span",
    "body_aspect_ratio",

    # Target
    "label"
]


# ============================================================
# PROCESS DATASET
# ============================================================

rows = []

total_images = 0
successful_images = 0
no_pose_images = 0
failed_images = 0


for pose_name in os.listdir(DATASET_DIR):

    pose_folder = os.path.join(
        DATASET_DIR,
        pose_name
    )

    if not os.path.isdir(pose_folder):
        continue


    print("\n" + "=" * 60)
    print(f"Processing class: {pose_name}")
    print("=" * 60)


    for image_name in os.listdir(pose_folder):

        image_path = os.path.join(
            pose_folder,
            image_name
        )


        # Only process images

        if not image_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue


        total_images += 1


        try:

            # =================================================
            # READ IMAGE
            # =================================================

            image = cv2.imread(image_path)


            if image is None:

                print(
                    f"Skipped {image_name}: "
                    f"could not read image"
                )

                failed_images += 1
                continue


            # =================================================
            # BGR → RGB
            # =================================================

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # CREATE MEDIAPIPE IMAGE
            # =================================================

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )


            # =================================================
            # DETECT POSE
            # =================================================

            result = detector.detect(
                mp_image
            )


            # =================================================
            # CHECK POSE
            # =================================================

            if not result.pose_landmarks:

                print(
                    f"No pose detected: "
                    f"{image_name}"
                )

                no_pose_images += 1
                continue


            # =================================================
            # GET LANDMARKS
            # =================================================

            landmarks = result.pose_landmarks[0]


            # =================================================
            # EXTRACT 18 FEATURES
            # =================================================

            features = extract_features(
                landmarks
            )


            # =================================================
            # ADD LABEL
            # =================================================

            features.append(
                pose_name
            )


            # =================================================
            # ADD ROW
            # =================================================

            rows.append(
                features
            )

            successful_images += 1


            print(
                f"[{successful_images}] "
                f"Processed: {image_name} "
                f"→ {pose_name}"
            )


        except Exception as error:

            failed_images += 1

            print(
                f"Error processing "
                f"{image_name}: {error}"
            )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(header)

    writer.writerows(rows)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print(
    f"Total images found: "
    f"{total_images}"
)

print(
    f"Successful images: "
    f"{successful_images}"
)

print(
    f"No pose detected: "
    f"{no_pose_images}"
)

print(
    f"Failed images: "
    f"{failed_images}"
)

print(
    f"\nDataset saved to: "
    f"{OUTPUT_FILE}"
)

print(
    f"Rows written to CSV: "
    f"{len(rows)}"
)

print(
    f"Features per sample: "
    f"{len(header) - 1}"
)
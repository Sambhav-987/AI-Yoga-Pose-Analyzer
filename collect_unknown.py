import cv2
import os


# ============================================================
# SETTINGS
# ============================================================

SAVE_DIR = "dataset/images/unknown"

TARGET_IMAGES = 80

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# IMAGE COUNTER
# ============================================================

existing_images = [
    file
    for file in os.listdir(SAVE_DIR)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

count = len(existing_images)


print("=" * 60)
print("UNKNOWN POSE DATA COLLECTION")
print("=" * 60)

print()
print("We will collect normal/non-target poses.")
print()
print("DO NOT perform:")
print("1. Plank")
print("2. Tree")
print("3. Warrior II")
print()
print("Press SPACE to capture an image.")
print("Press Q to quit.")
print()
print(f"Target images: {TARGET_IMAGES}")


# ============================================================
# MAIN LOOP
# ============================================================

while count < TARGET_IMAGES:

    success, frame = cap.read()

    if not success:

        print("Could not read webcam.")
        break


    # Mirror webcam

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Unknown images: {count}/{TARGET_IMAGES}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Capture | Q = Quit",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Show webcam
    # --------------------------------------------------------

    cv2.imshow(
        "Collect Unknown Poses",
        frame
    )


    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # SPACE → save image

    if key == ord(" "):

        filename = os.path.join(
            SAVE_DIR,
            f"unknown_{count + 1}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        count += 1

        print(
            f"Captured {count}/{TARGET_IMAGES}"
        )


    # Q → quit

    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETE")
print("=" * 60)

print(
    f"Unknown images collected: {count}"
)

print(
    f"Saved in: {SAVE_DIR}"
)
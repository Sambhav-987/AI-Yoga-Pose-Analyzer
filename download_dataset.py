import os
import requests
from urllib.parse import urlparse

# ==========================================
# SETTINGS
# ==========================================

LINKS_DIR = "dataset/links"
DATASET_DIR = "dataset/images"

# Start small for testing
IMAGES_PER_CLASS = 100

POSE_FILES = {
    "plank": "Plank_Pose_or_Kumbhakasana_.txt",
    "tree": "Tree_Pose_or_Vrksasana_.txt",
    "warrior_ii": "Warrior_II_Pose_or_Virabhadrasana_II_.txt"
}


# ==========================================
# CREATE DATASET DIRECTORY
# ==========================================

os.makedirs(DATASET_DIR, exist_ok=True)


# ==========================================
# DOWNLOAD ONE POSE
# ==========================================

def download_images(pose_name, filename):

    link_file = os.path.join(LINKS_DIR, filename)
    pose_folder = os.path.join(DATASET_DIR, pose_name)

    os.makedirs(pose_folder, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Processing: {pose_name}")
    print("=" * 50)

    # Check whether the link file exists
    if not os.path.exists(link_file):
        print(f"ERROR: File not found:")
        print(link_file)
        return

    # Read all links
    with open(link_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    downloaded = 0

    for line in lines:

        # Stop after reaching our target
        if downloaded >= IMAGES_PER_CLASS:
            break

        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        # --------------------------------------
        # Split:
        # image path + URL
        # --------------------------------------

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        image_path = parts[0].strip()
        image_url = parts[1].strip()

        # --------------------------------------
        # Get image filename
        # --------------------------------------

        image_name = os.path.basename(image_path)

        if not image_name:
            continue

        # --------------------------------------
        # Check URL
        # --------------------------------------

        parsed_url = urlparse(image_url)

        if parsed_url.scheme not in ["http", "https"]:
            print(f"Skipped: {image_name} (invalid URL)")
            continue

        # --------------------------------------
        # Where to save image
        # --------------------------------------

        save_path = os.path.join(
            pose_folder,
            image_name
        )

        # --------------------------------------
        # Don't download existing images again
        # --------------------------------------

        if os.path.exists(save_path):

            downloaded += 1

            print(
                f"[{downloaded}/{IMAGES_PER_CLASS}] "
                f"Already exists: {image_name}"
            )

            continue

        # --------------------------------------
        # DOWNLOAD
        # --------------------------------------

        try:

            response = requests.get(
                image_url,
                timeout=(5, 10),
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
                allow_redirects=True
            )

            # ----------------------------------
            # Check HTTP response
            # ----------------------------------

            if response.status_code != 200:

                print(
                    f"Skipped: {image_name} "
                    f"(HTTP {response.status_code})"
                )

                continue

            # ----------------------------------
            # Check whether file has content
            # ----------------------------------

            if len(response.content) < 1000:

                print(
                    f"Skipped: {image_name} "
                    f"(file too small)"
                )

                continue

            # ----------------------------------
            # Save image
            # ----------------------------------

            with open(save_path, "wb") as image_file:

                image_file.write(response.content)

            downloaded += 1

            print(
                f"[{downloaded}/{IMAGES_PER_CLASS}] "
                f"Downloaded: {image_name}"
            )

        # --------------------------------------
        # Handle network errors
        # --------------------------------------

        except requests.exceptions.Timeout:

            print(
                f"Skipped: {image_name} "
                f"(timeout)"
            )

        except requests.exceptions.RequestException as error:

            print(
                f"Failed: {image_name} "
                f"({error})"
            )

        except Exception as error:

            print(
                f"Unexpected error for {image_name}: "
                f"{error}"
            )

    # ------------------------------------------
    # SUMMARY
    # ------------------------------------------

    print("\nFinished:", pose_name)
    print(
        f"Successfully available: "
        f"{downloaded}/{IMAGES_PER_CLASS}"
    )


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("        AI YOGA DATASET DOWNLOADER")
    print("=" * 50)

    print(f"\nTarget images per class: {IMAGES_PER_CLASS}")

    # Process every pose
    for pose_name, filename in POSE_FILES.items():

        download_images(
            pose_name,
            filename
        )

    print("\n" + "=" * 50)
    print("Dataset download complete!")
    print("=" * 50)
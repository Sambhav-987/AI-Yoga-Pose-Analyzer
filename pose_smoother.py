import numpy as np


class PoseSmoother:

    def __init__(self, alpha=0.5):

        self.alpha = alpha

        # Stores landmarks from the previous frame
        self.previous_landmarks = None


    def smooth(self, landmarks):

        # First frame
        if self.previous_landmarks is None:

            self.previous_landmarks = [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
                for landmark in landmarks
            ]

            return landmarks


        # Smooth every landmark
        for i, landmark in enumerate(landmarks):

            current = np.array([
                landmark.x,
                landmark.y,
                landmark.z
            ])

            previous = np.array(
                self.previous_landmarks[i]
            )


            # Exponential moving average
            smoothed = (
                self.alpha * current
                +
                (1 - self.alpha) * previous
            )


            # Save smoothed coordinates
            self.previous_landmarks[i] = smoothed.tolist()


            # Update MediaPipe landmark
            landmark.x = float(smoothed[0])
            landmark.y = float(smoothed[1])
            landmark.z = float(smoothed[2])


        return landmarks


def landmarks_are_visible(landmarks, threshold=0.5):

    # Important landmarks required by our analyzers
    important_points = [
        11, 12,       # shoulders
        13, 14,       # elbows
        15, 16,       # wrists
        23, 24,       # hips
        25, 26,       # knees
        27, 28        # ankles
    ]


    for index in important_points:

        if landmarks[index].visibility < threshold:

            return False


    return True
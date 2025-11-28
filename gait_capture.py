import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
drawing_utils = mp.solutions.drawing_utils

# Global stop flag for external API stopping
_stop_flag = False

def request_stop():
    """Allows an external API call to stop the capture loop."""
    global _stop_flag
    _stop_flag = True

def _angle(a, b, c):
    """Calculate 2D angle at joint B (in degrees)."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def capture_gait(max_duration_s: float | None = 20.0):
    """
    Captures gait from webcam and returns summary stats in dict/JSON form.
    Stops when:
      - duration reached
      - Q is pressed
      - request_stop() is called
    """
    global _stop_flag
    _stop_flag = False  # reset flag

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam")

    start_time = time.time()
    left_knee_angles = []
    right_knee_angles = []
    step_widths_norm = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            # stop conditions
            if _stop_flag:
                break
            if max_duration_s is not None and (time.time() - start_time) >= max_duration_s:
                break

            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed, stopping...")
                break

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # ---- Draw landmarks here so you KNOW it's tracking the right spots ----
            if results.pose_landmarks:
                # Add landmark overlay BEFORE showing the frame
                if results.pose_landmarks:
                    drawing_utils.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                # Extract pose landmarks
                lm = results.pose_landmarks.landmark

                # Compute joint angles
                left_knee = _angle(
                    lm[mp_pose.PoseLandmark.LEFT_HIP],
                    lm[mp_pose.PoseLandmark.LEFT_KNEE],
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                )
                right_knee = _angle(
                    lm[mp_pose.PoseLandmark.RIGHT_HIP],
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                )

                # Compute spatial step width normalization (relative body spacing)
                la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                step_width_norm = abs(la.x - ra.x)

                # Append results for stats
                left_knee_angles.append(left_knee)
                right_knee_angles.append(right_knee)
                step_widths_norm.append(step_width_norm)

            # Show feed with landmarks overlay
            cv2.imshow("Gait capture (press Q to stop)", frame)

            # Manual stop from keypress
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    duration_real_s = time.time() - start_time

    if not left_knee_angles:
        raise RuntimeError("No person detected in gait recording")

    # Create summary stats
    summary = {
        "duration_s": duration_real_s,
        "mean_left_knee_deg": float(np.mean(left_knee_angles)),
        "mean_right_knee_deg": float(np.mean(right_knee_angles)),
        "knee_symmetry_deg": abs(np.mean(left_knee_angles) - np.mean(right_knee_angles)),
        "mean_step_width_norm": float(np.mean(step_widths_norm)),
        "std_step_width_norm": float(np.std(step_widths_norm)),
        "n_frames": int(len(left_knee_angles)),
    }

    return summary

if __name__ == "__main__":
    # run for 20 seconds if executing directly
    result = capture_gait(max_duration_s=20.0)
    print(result)

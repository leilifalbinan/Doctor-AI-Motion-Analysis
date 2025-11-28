import cv2
import mediapipe as mp
import numpy as np
import time
import json
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def angle(a, b, c):
    """Compute angle ABC with vectors BA and BC (2D projected)."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    # List to store per-frame gait data
    gait_frames = []

    start_time = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Current timestamp (seconds since start)
            t = time.time() - start_time

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Knee angles
                left_knee_angle = angle(
                    lm[mp_pose.PoseLandmark.LEFT_HIP],
                    lm[mp_pose.PoseLandmark.LEFT_KNEE],
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE],
                )
                right_knee_angle = angle(
                    lm[mp_pose.PoseLandmark.RIGHT_HIP],
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
                )

                # Step width (distance between ankles in normalized x coord)
                left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                step_width = abs(float(left_ankle.x) - float(right_ankle.x))

                # Save data for this frame
                gait_frames.append({
                    "time_s": t,
                    "left_knee_deg": left_knee_angle,
                    "right_knee_deg": right_knee_angle,
                    "step_width_norm": step_width,
                })

                # On-screen debug text
                cv2.putText(image_bgr, f"Left Knee: {left_knee_angle:.1f} deg",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.putText(image_bgr, f"Right Knee: {right_knee_angle:.1f} deg",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.putText(image_bgr, f"Step Width: {step_width:.3f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            cv2.imshow("DoctorAI - Gait Analysis (press Q to save & quit)", image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Save to JSON when done
    if gait_frames:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gait_data_{ts}.json"
        data = {
            "created_at": datetime.now().isoformat(),
            "num_frames": len(gait_frames),
            "frames": gait_frames,
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved {len(gait_frames)} frames to {filename}")
    else:
        print("No gait data captured (no pose detected).")

if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose

# Global flag so another endpoint can request the capture to stop
_stop_flag = False


def request_stop():
    """Ask the current capture loop to stop on the next frame."""
    global _stop_flag
    _stop_flag = True


def _angle(a, b, c):
    """Compute angle ABC with vectors BA and BC (2D projected)."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _summarize_gait(left_knees, right_knees, step_widths, step_count, duration_s):
    """Compute all stats + summary text from time-series."""
    if not left_knees:
        raise RuntimeError("No gait frames captured")

    left_knees_arr = np.array(left_knees, dtype=float)
    right_knees_arr = np.array(right_knees, dtype=float)
    step_widths_arr = np.array(step_widths, dtype=float)

    n_frames = len(left_knees_arr)

    mean_left = float(np.mean(left_knees_arr))
    mean_right = float(np.mean(right_knees_arr))

    min_left = float(np.min(left_knees_arr))
    min_right = float(np.min(right_knees_arr))

    max_left = float(np.max(left_knees_arr))
    max_right = float(np.max(right_knees_arr))

    std_left = float(np.std(left_knees_arr))
    std_right = float(np.std(right_knees_arr))

    mean_step_width = float(np.mean(step_widths_arr))
    std_step_width = float(np.std(step_widths_arr))

    num_steps = int(step_count)
    cadence_spm = (
        float(num_steps / (duration_s / 60.0))
        if duration_s > 0 and num_steps > 0
        else 0.0
    )

    # ---- summary text ----
    knee_diff = abs(mean_left - mean_right)
    if knee_diff < 5:
        asym_text = "symmetric knee angles"
    elif knee_diff < 10:
        asym_text = f"mild knee asymmetry (~{knee_diff:.1f}° difference)"
    else:
        asym_text = f"marked knee asymmetry (~{knee_diff:.1f}° difference)"

    left_reduced = min_left > 140
    right_reduced = min_right > 140
    if left_reduced and right_reduced:
        flex_text = "bilaterally reduced knee flexion (stiff-knee pattern)"
    elif left_reduced:
        flex_text = "reduced left knee flexion"
    elif right_reduced:
        flex_text = "reduced right knee flexion"
    else:
        flex_text = "normal knee flexion range"

    if mean_step_width < 0.15:
        base_text = "very narrow base of support"
    elif mean_step_width > 0.45:
        base_text = "wide base of support"
    else:
        base_text = "normal base of support"

    summary_text = (
        f"Gait shows {asym_text}, {flex_text}, and {base_text}. "
        f"Approx. {num_steps} steps captured at ~{cadence_spm:.1f} steps/min."
    )

    return {
        "duration_s": float(duration_s),
        "n_frames": int(n_frames),
        "mean_left_knee_deg": mean_left,
        "mean_right_knee_deg": mean_right,
        "min_left_knee_deg": min_left,
        "min_right_knee_deg": min_right,
        "max_left_knee_deg": max_left,
        "max_right_knee_deg": max_right,
        "std_left_knee_deg": std_left,
        "std_right_knee_deg": std_right,
        "mean_step_width_norm": mean_step_width,
        "std_step_width_norm": std_step_width,
        "num_steps": num_steps,
        "cadence_spm": cadence_spm,
        "summary_text": summary_text,
    }


def capture_gait(max_duration_s: float | None = 20.0):
    """
    Record gait from the webcam until:
      - max_duration_s is reached (if not None), OR
      - user presses Q in the camera window, OR
      - request_stop() is called (sets _stop_flag).

    Returns summary metrics as a dict.
    """
    global _stop_flag
    _stop_flag = False  # reset stop flag

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    start = time.time()
    left_knees = []
    right_knees = []
    step_widths = []

    # rough step counting based on left knee flexion
    left_flexing = False
    step_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            if _stop_flag:
                break
            if max_duration_s is not None and (time.time() - start) >= max_duration_s:
                break

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                lk = _angle(
                    lm[mp_pose.PoseLandmark.LEFT_HIP],
                    lm[mp_pose.PoseLandmark.LEFT_KNEE],
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE],
                )
                rk = _angle(
                    lm[mp_pose.PoseLandmark.RIGHT_HIP],
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
                )

                la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                sw = abs(float(la.x) - float(ra.x))

                left_knees.append(lk)
                right_knees.append(rk)
                step_widths.append(sw)

                # knee flexion-based step count
                if lk < 160 and not left_flexing:
                    left_flexing = True
                if lk > 170 and left_flexing:
                    step_count += 1
                    left_flexing = False

            cv2.imshow("Gait capture (press Q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    duration_real = time.time() - start
    return _summarize_gait(left_knees, right_knees, step_widths, step_count, duration_real)


def capture_gait_from_file(video_path: str, max_duration_s: float | None = 30.0):
    """
    Analyze gait from a pre-recorded video file.
    - video_path: path to a local video (mp4, mov, etc.)
    - max_duration_s: optional cap on how much of the clip to process
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback guess

    left_knees = []
    right_knees = []
    step_widths = []
    left_flexing = False
    step_count = 0
    frame_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            frame_count += 1
            current_time_s = frame_count / fps
            if max_duration_s is not None and current_time_s >= max_duration_s:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                lk = _angle(
                    lm[mp_pose.PoseLandmark.LEFT_HIP],
                    lm[mp_pose.PoseLandmark.LEFT_KNEE],
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE],
                )
                rk = _angle(
                    lm[mp_pose.PoseLandmark.RIGHT_HIP],
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
                )

                la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                sw = abs(float(la.x) - float(ra.x))

                left_knees.append(lk)
                right_knees.append(rk)
                step_widths.append(sw)

                if lk < 160 and not left_flexing:
                    left_flexing = True
                if lk > 170 and left_flexing:
                    step_count += 1
                    left_flexing = False

    cap.release()

    duration_est = frame_count / fps if fps > 0 else max_duration_s or 0.0
    return _summarize_gait(left_knees, right_knees, step_widths, step_count, duration_est)


if __name__ == "__main__":
    # quick manual test
    summary = capture_gait(max_duration_s=10.0)
    print(summary)

import argparse
import json
from gait_capture import capture_gait, capture_gait_from_file

def main():
    parser = argparse.ArgumentParser(
        description="Run MediaPipe gait analysis and print JSON summary."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--webcam",
        action="store_true",
        help="Use default webcam as the video source",
    )
    group.add_argument(
        "--video",
        type=str,
        help="Path to a video file (mp4, mov, etc.)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Max duration to analyze in seconds (default: 20)",
    )

    args = parser.parse_args()

    if args.webcam:
        summary = capture_gait(max_duration_s=args.duration)
    else:
        summary = capture_gait_from_file(args.video, max_duration_s=args.duration)

    # Pretty JSON output
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

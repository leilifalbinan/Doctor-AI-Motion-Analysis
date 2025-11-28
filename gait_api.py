from flask import Flask, jsonify, request
from flask_cors import CORS
from gait_capture import capture_gait, request_stop, capture_gait_from_file
import tempfile
import os

app = Flask(__name__)
CORS(app)  # allow requests from Vite (localhost:5173)


@app.route("/api/gait", methods=["GET"])
def api_gait():
    """
    Start gait capture from webcam.
    Optional query parameter:
      /api/gait?duration=30   -> max 30 seconds
    """
    try:
      duration = request.args.get("duration", default=20.0, type=float)
      summary = capture_gait(max_duration_s=duration)
      return jsonify({"ok": True, "summary": summary})
    except Exception as e:
      return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/gait/stop", methods=["POST"])
def api_gait_stop():
    """
    Request the currently running gait capture to stop early.
    """
    request_stop()
    return jsonify({"ok": True})


@app.route("/api/gait/upload", methods=["POST"])
def api_gait_upload():
    """
    Analyze gait from an uploaded video file.

    Frontend should send:
      POST /api/gait/upload?duration=30
      Content-Type: multipart/form-data
      field: "file" = video file
    """
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file part in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"ok": False, "error": "No selected file"}), 400

        duration = request.args.get("duration", default=30.0, type=float)

        # Save to a temporary file
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            file.save(tmp_path)
            summary = capture_gait_from_file(tmp_path, max_duration_s=duration)
        finally:
            # Clean up temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # threaded=True so /api/gait and /api/gait/stop can run concurrently
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=True)

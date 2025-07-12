import streamlit as st
import cv2
import numpy as np
import tempfile
from utils import detect_people

# Streamlit setup
st.set_page_config(page_title="Crowd Density Estimator", layout="wide")
st.title("📊 Crowd Density Estimator with Heatmap")

# Threshold slider
crowd_threshold = st.slider("🚨 Crowd Alert Threshold", 5, 50, 10)

# Upload video file
uploaded_file = st.file_uploader("📁 Upload a video (.mp4 only)", type=["mp4"])

if uploaded_file:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()
    count_placeholder = st.empty()

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output/processed_output.mp4", fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and fix BGRA if needed
        frame = cv2.resize(frame, (640, 480))
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Detect people using YOLOv8
        people = detect_people(frame)

        # Create heatmap
        heatmap = np.zeros((480, 640), dtype=np.uint8)
        for box in people:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(heatmap, (cx, cy), 30, 255, -1)

        # Apply heatmap and overlay
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.7, colored_heatmap, 0.5, 0)

        # Add text
        cv2.putText(blended, f"People Detected: {len(people)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if len(people) > crowd_threshold:
            cv2.putText(blended, "⚠️ Crowd Limit Exceeded!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        out.write(blended)

        # Convert to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        count_placeholder.info(f"👥 People detected: {len(people)}")

    cap.release()
    out.release()

    st.success("✅ Video processing completed.")
    st.video("output/processed_output.mp4")

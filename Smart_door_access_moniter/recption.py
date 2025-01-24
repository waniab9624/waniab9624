
import cv2
import streamlit as st
from datetime import datetime
import threading
import time
import os
import numpy as np
import tempfile
from playsound import playsound

# Function to play alarm sound
def play_alarm():
    playsound("E:\\Agent\\ring-phone-190265.mp3")  # Replace with your alarm sound path

# Function to check if current time is within specified hours
def is_within_hours(start_hour, end_hour):
    current_hour = datetime.now().hour
    return start_hour <= current_hour < end_hour

# Streamlit UI setup
st.title("Door Activity Monitoring System")

st.sidebar.header("Settings")
start_hour = st.sidebar.slider("Startup Hour", 0, 23, 8)
end_hour = st.sidebar.slider("End Hour", 0, 23, 20)
detection_threshold = st.sidebar.slider("Detection Threshold (seconds)", 1, 60, 10)  # Default 10 seconds
alarm_rearm_time = st.sidebar.slider("Alarm Rearm Time (seconds)", 10, 300, 60)

detection_status = st.empty()
frame_placeholder = st.empty()

# Load YOLO model (with configuration and weights)
yolo_net = cv2.dnn.readNet("yolov4 (1).weights", "yolov4 (1).cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Upload video file through Streamlit
video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_video_path = temp_file.name

    video_capture = cv2.VideoCapture(temp_video_path)

    if not video_capture.isOpened():
        st.error("Could not open video file.")
    else:
        st.info("Video loaded successfully.")

    # Variables for detection timing, person count, and door entry tracking
    detection_start_time = None
    alarm_triggered = False
    alarm_last_triggered = None
    persons_entered = 0
    persons_exited = 0

    # Frame processing variables
    frame_skip = 5
    frame_count = 0

    # Process video feed
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            st.info("Video processing complete.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames to optimize performance
            continue

        height, width, _ = frame.shape
        # Define the door area dynamically based on frame size
        door_area = (
            int(width * 0.4), int(height * 0.7),
            int(width * 0.2), int(height * 0.2)
        )

        # Prepare the frame for YOLO object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(output_layers)

        # Parse YOLO outputs
        class_ids = []
        confidences = []
        boxes = []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Adjust confidence threshold if necessary
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression to avoid multiple boxes on the same object
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        person_detected_in_door = False

        if len(indices) > 0:
            for i in indices.flatten():
                if classes[class_ids[i]] == 'person':  # Check if it's a person
                    x, y, w, h = boxes[i]
                    # Check if any part of the person’s bounding box is within the door area
                    if (x + w > door_area[0] and x < door_area[0] + door_area[2] and
                        y + h > door_area[1] and y < door_area[1] + door_area[3]):
                        person_detected_in_door = True

        # Set alarm after 10 seconds if no person is detected
        if not person_detected_in_door:
            if detection_start_time is None:
                detection_start_time = time.time()

            elapsed_time = time.time() - detection_start_time

            if elapsed_time >= detection_threshold and not alarm_triggered:
                if is_within_hours(start_hour, end_hour):
                    threading.Thread(target=play_alarm).start()
                    alarm_triggered = True
                    alarm_last_triggered = time.time()
        else:
            detection_start_time = None  # Reset timer when a person is detected

        # Rearm the alarm after a set time
        if alarm_last_triggered and time.time() - alarm_last_triggered > alarm_rearm_time:
            alarm_triggered = False

        # Display video frame in Streamlit
        # Draw door area rectangle in red for visualization
        cv2.rectangle(frame, (door_area[0], door_area[1]),
                      (door_area[0] + door_area[2], door_area[1] + door_area[3]), (0, 0, 255), 2)
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Show status
        detection_status.info(
            f"Alarm Triggered: {'Yes' if alarm_triggered else 'No'} | No Detection Time: {int(elapsed_time) if not person_detected_in_door else 0}s"
        )

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
else:
    st.info("Please upload a video to start the monitoring.")

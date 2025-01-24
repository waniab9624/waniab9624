
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import tempfile
from playsound import playsound
from collections import deque
from io import BytesIO

# FastAPI instance
app = FastAPI()

# Function to play alarm sound
def play_alarm():
    playsound("E:\\Agent\\ring-phone-190265.mp3")  # Replace with your alarm sound path

# Function to check if current time is within specified hours
def is_within_hours(start_hour: int, end_hour: int):
    current_hour = datetime.now().hour
    return start_hour <= current_hour < end_hour

# Load YOLO model
yolo_net = cv2.dnn.readNet("yolov4 (1).weights", "yolov4 (1).cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Global variables
persons_entered = 0
tracked_persons = deque(maxlen=50)  # To track persons
next_person_id = 1

@app.post("/process_video/")
async def process_video(
    video_file: UploadFile,
    start_hour: int = Form(...),
    end_hour: int = Form(...),
    detection_threshold: int = Form(...),
    alarm_rearm_time: int = Form(...),
):
    global persons_entered, next_person_id, tracked_persons

    # Save uploaded video to a temporary file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_video_path, "wb") as temp_file:
        temp_file.write(await video_file.read())

    # Open the video file
    video_capture = cv2.VideoCapture(temp_video_path)
    if not video_capture.isOpened():
        return JSONResponse(content={"error": "Could not open video file."}, status_code=400)

    detection_start_time = None
    alarm_triggered = False
    alarm_last_triggered = None

    def video_stream():
        nonlocal detection_start_time, alarm_triggered, alarm_last_triggered
        global persons_entered, next_person_id

        frame_skip = 5
        frame_count = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            height, width, _ = frame.shape
            door_area = (
                int(width * 0.4), int(height * 0.7),
                int(width * 0.2), int(height * 0.2)
            )

            # Prepare the frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
            yolo_net.setInput(blob)
            outputs = yolo_net.forward(output_layers)

            class_ids, confidences, boxes = [], [], []
            for out in outputs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            detected_persons = []

            if len(indices) > 0:
                for i in indices.flatten():
                    if classes[class_ids[i]] == 'person':
                        x, y, w, h = boxes[i]
                        if (x + w > door_area[0] and x < door_area[0] + door_area[2] and
                            y + h > door_area[1] and y < door_area[1] + door_area[3]):
                            detected_persons.append((x, y, w, h))

            new_tracked_persons = []
            for person in detected_persons:
                px, py, pw, ph = person
                matched = False

                for tid, tx, ty, tw, th in tracked_persons:
                    if abs(px - tx) < 50 and abs(py - ty) < 50:
                        new_tracked_persons.append((tid, px, py, pw, ph))
                        matched = True
                        break

                if not matched:
                    new_tracked_persons.append((next_person_id, px, py, pw, ph))
                    next_person_id += 1
                    persons_entered += 1

            tracked_persons.clear()
            tracked_persons.extend(new_tracked_persons)

            if len(detected_persons) == 0:
                if detection_start_time is None:
                    detection_start_time = time.time()

                elapsed_time = time.time() - detection_start_time

                if elapsed_time >= detection_threshold and not alarm_triggered:
                    if is_within_hours(start_hour, end_hour):
                        threading.Thread(target=play_alarm).start()
                        alarm_triggered = True
                        alarm_last_triggered = time.time()
            else:
                detection_start_time = None

            if alarm_last_triggered and time.time() - alarm_last_triggered > alarm_rearm_time:
                alarm_triggered = False

            cv2.rectangle(frame, (door_area[0], door_area[1]),
                          (door_area[0] + door_area[2], door_area[1] + door_area[3]), (0, 0, 255), 2)

            for tid, x, y, w, h in tracked_persons:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {tid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np
from collections import defaultdict

app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()

detected_vehicles = defaultdict(int)
danger_status = "No danger detected"
prev_positions = {}

# Helper functions
def get_centroid(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def box_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)

def intersection_over_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = box_area(*boxA)
    areaB = box_area(*boxB)
    smaller_area = min(areaA, areaB)
    if smaller_area == 0:
        return 0
    return interArea / smaller_area

def gen_frames():
    global detected_vehicles, danger_status, prev_positions
    cap = cv2.VideoCapture('head_on_collision_135.mp4')
    vehicle_id_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Keep previous danger_status unless new collision is not detected
        if danger_status != "⚠️ Collision Detected":
            danger_status = "No danger detected"

        frame_resized = cv2.resize(frame, (640, 360))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = model([img_rgb])
        df = results.pandas().xyxy[0]

        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 360

        current_positions = {}
        detected_vehicles.clear()
        boxes = []

        for i, row in df.iterrows():
            label = row['name']
            conf = row['confidence']
            if label in ['car', 'motorcycle', 'bus', 'truck'] and conf > 0.4:
                x1 = int(row['xmin'] * scale_x)
                y1 = int(row['ymin'] * scale_y)
                x2 = int(row['xmax'] * scale_x)
                y2 = int(row['ymax'] * scale_y)
                centroid = get_centroid(x1, y1, x2, y2)

                matched_id = None
                for vid, prev_centroid in prev_positions.items():
                    if euclidean_distance(centroid, prev_centroid) < 40:
                        matched_id = vid
                        break

                if matched_id is None:
                    vehicle_id_counter += 1
                    matched_id = vehicle_id_counter

                current_positions[matched_id] = centroid
                boxes.append((matched_id, x1, y1, x2, y2))

                detected_vehicles[label] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} ID:{matched_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Collision detection
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                id1, x1, y1, x2, y2 = boxes[i]
                id2, a1, b1, a2, b2 = boxes[j]

                boxA = (x1, y1, x2, y2)
                boxB = (a1, b1, a2, b2)

                overlap_ratio = intersection_over_area(boxA, boxB)

                if overlap_ratio >= 0.75:
                    if id1 in prev_positions and id2 in prev_positions:
                        prev_c1 = prev_positions[id1]
                        prev_c2 = prev_positions[id2]
                        new_c1 = current_positions[id1]
                        new_c2 = current_positions[id2]

                        prev_dist = euclidean_distance(prev_c1, prev_c2)
                        new_dist = euclidean_distance(new_c1, new_c2)

                        if new_dist < prev_dist - 5:
                            danger_status = "⚠️ Collision Detected"
                            cv2.putText(frame, danger_status, (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            break
            else:
                continue
            break

        prev_positions = current_positions

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vehicle_data')
def vehicle_data():
    return {"vehicles": dict(detected_vehicles), "danger": danger_status}

if __name__ == "__main__":
    app.run(debug=True)

import cv2
import os
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch import nn
from datetime import datetime
import face_recognition
import mediapipe as mp
import firebase_admin
from firebase_admin import credentials, firestore
import requests

# ----------------- Anti-Spoofing Model ------------------
class DepthWiseSeparable(nn.Module):
    def __init__(self, nin, nout, stride):
        super().__init__()
        self.conv = nn.Conv2d(nin, nin, 3, stride, 1, groups=nin, bias=False)
        self.bn = nn.BatchNorm2d(nin)
        self.pointwise = nn.Conv2d(nin, nout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nout)
        self.prelu = nn.PReLU(nout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.prelu(x)
        return x

class MiniFASNetV1SE(nn.Module):
    def __init__(self, image_height=80, image_width=80):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU(16)
        self.dw1 = DepthWiseSeparable(16, 16, 1)
        self.dw2 = DepthWiseSeparable(16, 32, 2)
        self.dw3 = DepthWiseSeparable(32, 64, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))

class AntiSpoofDetector:
    def __init__(self, model_path="4_0_0_80x80_MiniFASNetV1SE.pth"):
        state = torch.load(model_path, map_location="cpu")
        state = state["state_dict"] if "state_dict" in state else state
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        self.model = MiniFASNetV1SE()
        self.model.load_state_dict(state)
        self.model.eval()

    def is_real_face(self, face_img):
        face_resized = cv2.resize(face_img, (80, 80))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            prob = F.softmax(self.model(tensor), dim=1)[0][1].item()
        return prob > 0.4

# ----------------- Face Recognition ------------------
class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        print("[INFO] Loading known faces...")
        for dirpath, _, filenames in os.walk(images_path):
            for file in filenames:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(dirpath, file)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(os.path.basename(dirpath))

    def detect_known_faces(self, frame):
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)
        names = []
        for enc in encodings:
            name = "Unknown"
            if self.known_face_encodings:
                distances = face_recognition.face_distance(self.known_face_encodings, enc)
                best_match = np.argmin(distances)
                if distances[best_match] < 0.5:
                    name = self.known_face_names[best_match]
            names.append(name)
        locations = [(t*4, r*4, b*4, l*4) for (t, r, b, l) in locations]
        return locations, names

# ----------------- EAR Detection ------------------
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def calculate_EAR(landmarks, left_indices, right_indices):
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C)
    left = [landmarks[i] for i in left_indices]
    right = [landmarks[i] for i in right_indices]
    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

# ----------------- Main Attendance ------------------
if __name__ == "__main__":
    with open("Dheeraj.yml", "r") as f:
        config = yaml.safe_load(f)

    camera_url = config["camera"]["url"]
    java_api_url = config["java_api"]["url"]
    auth_token = config["java_api"]["token"]

    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    sfr = SimpleFacerec()
    sfr.load_encoding_images("Faces")
    antispoof = AntiSpoofDetector()
    cap = cv2.VideoCapture(camera_url)

    seen_names = {}
    instructions = ["Please blink", "Turn right", "Turn left", "Smile"]
    current_instruction = np.random.randint(len(instructions))
    blink_detected = False

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_locations, face_names = sfr.detect_known_faces(frame)
            if face_locations:
                y1, x2, y2, x1 = face_locations[0]
                face_img = frame[y1:y2, x1:x2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    h, w, _ = frame.shape
                    points = [(int(p.x * w), int(p.y * h)) for p in lm]
                    EAR = calculate_EAR(points, LEFT_EYE_IDX, RIGHT_EYE_IDX)
                    if EAR < 0.25:
                        blink_detected = True

                cv2.putText(frame, f"Instruction: {instructions[current_instruction]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if antispoof.is_real_face(face_img):
                    name = face_names[0]
                    if name != "Unknown" and blink_detected:
                        if name not in seen_names:
                            timestamp = datetime.now().isoformat()
                            seen_names[name] = timestamp
                            db.collection("Attendance").add({"Name": name, "Timestamp": timestamp, "Attendance": "P"})

                            headers = {"Authorization": f"Bearer {auth_token}"}
                            payload = {"empId": name, "loginTime": timestamp}
                            try:
                                res = requests.post(java_api_url, json=payload, headers=headers)
                                print(f"[INFO] Marked {name}: {res.status_code}")
                            except Exception as e:
                                print("[ERROR]", e)

                            blink_detected = False

                        cv2.putText(frame, f"{name} - Present", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    else:
                        cv2.putText(frame, "Blink to confirm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
                else:
                    cv2.putText(frame, "Fake Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.imshow("Face Attendance", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

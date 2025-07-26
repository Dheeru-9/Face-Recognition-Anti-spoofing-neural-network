import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import torch
import torch.nn.functional as F
from torch import nn
import mediapipe as mp
import yaml
import requests
import time
import threading

# ----------------- Load Config ------------------
YAML_FILE_PATH = r".yml_Location"
with open(YAML_FILE_PATH, "r") as f:
    config = yaml.safe_load(f)

camera_url = config["camera"]["url"]
java_api_url = config["java_api"]["url"]
auth_url = config["java_api"]["auth_url"]
auth_username = config["java_api"]["username"]
auth_password = config["java_api"]["password"]
auth_token = config["java_api"]["token"]

# ----------------- Get New Token ------------------
def get_new_token():
    payload = {
        "userName": auth_username,
        "password": auth_password
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    try:
        response = requests.post(auth_url, json=payload, headers=headers)
        print("[DEBUG] Status Code:", response.status_code)
        print("[DEBUG] Response Text:", response.text)
        if response.status_code == 200:
            token = response.json().get("token")
            print("[INFO] Token received:", token)
            return token
        elif response.status_code == 401:
            print("[ERROR] Unauthorized. Check credentials.")
        elif response.status_code == 403:
            print("[ERROR] Forbidden. Access denied.")
        else:
            print(f"[ERROR] Unexpected status code: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print("[ERROR] Request failed:", e)
    return None

# ----------------- Update Token in YAML ------------------
def update_token_in_yaml(token):
    try:
        with open(YAML_FILE_PATH, "r") as f:
            config = yaml.safe_load(f) or {}

        if "java_api" not in config:
            config["java_api"] = {}

        config["java_api"]["token"] = token

        with open(YAML_FILE_PATH, "w") as f:
            yaml.dump(config, f)

        print("[INFO] Token updated in YAML")
    except Exception as e:
        print(f"[ERROR] Exception while updating YAML: {e}")

# ----------------- Token Refresh Logic ------------------
def refresh_token():
    global auth_token
    new_token = get_new_token()
    if new_token:
        auth_token = new_token
        update_token_in_yaml(new_token)

# ----------------- Token Refresh Every 5 Minutes ------------------
def refresh_token_every_5_minutes():
    while True:
        refresh_token()
        time.sleep(300)  # 5 minutes = 300 seconds
        print("refreshing", datetime.now())

# ----------------- Anti-Spoofing Model ------------------
class DepthWiseSeparable(nn.Module):
    def __init__(self, nin, nout, stride):
        super(DepthWiseSeparable, self).__init__()
        self.conv = nn.Conv2d(nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin, bias=False)
        self.bn = nn.BatchNorm2d(nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=False)
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
        super(MiniFASNetV1SE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU(16)
        self.dw1 = DepthWiseSeparable(16, 16, 1)
        self.dw2 = DepthWiseSeparable(16, 32, 2)
        self.dw3 = DepthWiseSeparable(32, 64, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class AntiSpoofDetector:
    def __init__(self, model_path="4_0_0_80x80_MiniFASNetV1SE.pth"):
        print("[INFO] Loading Anti-Spoofing model...")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        self.model = MiniFASNetV1SE(80, 80)
        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.eval()

    def is_real_face(self, face_img):
        face_resized = cv2.resize(face_img, (80, 80))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            output = self.model(face_tensor)
            prob = F.softmax(output, dim=1)[0][1].item()
        return prob > 0.4

# ----------------- Eye Aspect Ratio ------------------
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def calculate_EAR(landmarks, left_indices, right_indices):
    def eye_aspect_ratio(eye_points):
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        return (A + B) / (2.0 * C)
    left_eye = [landmarks[i] for i in left_indices]
    right_eye = [landmarks[i] for i in right_indices]
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

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
                        encoding = encodings[0]
                        name = os.path.basename(dirpath)
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                        print(f"[INFO] Loaded encoding for {name} from {file}")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            if self.known_face_encodings:
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < 0.5:
                    name = self.known_face_names[best_match_index]
            face_names.append(name)
        face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]
        return face_locations, face_names

# ----------------- Main ------------------
if __name__ == "__main__":
    threading.Thread(target=refresh_token_every_5_minutes, daemon=True).start()
    refresh_token()

    IMAGE_PATH = r"Image_Path"

    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    sfr = SimpleFacerec()
    sfr.load_encoding_images(IMAGE_PATH)
    antispoof = AntiSpoofDetector()
    cap = cv2.VideoCapture(camera_url)

    seen_names = dict()
    instructions = ["Please blink your eyes", "Please turn your face to the right", "Please turn your face to the left", "Please smile"]
    current_instruction_index = np.random.randint(len(instructions))
    blink_detected = False

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_locations, face_names = sfr.detect_known_faces(frame)
            if face_locations:
                y1, x2, y2, x1 = face_locations[0]
                face_img = frame[y1:y2, x1:x2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    h, w, _ = frame.shape
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                    EAR = calculate_EAR(points, LEFT_EYE_IDX, RIGHT_EYE_IDX)
                    if EAR < 0.25:
                        blink_detected = True

                cv2.putText(frame, f"Instruction: {instructions[current_instruction_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if antispoof.is_real_face(face_img):
                    name = face_names[0]
                    if name != "Unknown" and blink_detected:
                        if name not in seen_names:
                            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                            seen_names[name] = timestamp
                            db.collection("Attendance").add({"Name": name, "Timestamp": timestamp, "Attendance": "P"})

                            payload = {"empId": name, "loginTime": timestamp}
                            headers = {"Authorization": f"Bearer {auth_token}"}

                            try:
                                response = requests.post(java_api_url, json=payload, headers=headers)
                                if response.status_code in [200, 201]:
                                    print(f"[INFO] {name} marked Present via Java API at {timestamp}")
                                else:
                                    print(f"[ERROR] Java API failed. Status: {response.status_code}, Response: {response.text}")
                            except Exception as e:
                                print(f"[ERROR] Could not connect to Java API: {e}")

                            blink_detected = False

                        cv2.putText(frame, f"{name} - Present", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Please blink your eyes to confirm", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "2D Fake Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No face detected, please show your face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

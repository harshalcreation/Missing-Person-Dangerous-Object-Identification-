import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog, messagebox
import threading
from threading import Thread
from playsound import playsound
import os
import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import csv
from datetime import datetime
from PIL import Image, ImageTk
import mediapipe as mp
from geopy.geocoders import Nominatim
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api
from pymongo import MongoClient
# from utils import Utility
# from encryption_algo import encrypt_data
# from whatsapp import WhatsAppNotifier

class CloudinaryManager:
    def __init__(self):
        cloudinary.config(
            cloud_name="dcvsavnx3",
            api_key="399294449426652",
            api_secret="7ZJuKbNHl0KsWPMAcKb4OWDH-j8"
        )

    def upload_image(self, file_path, public_id):
        response = cloudinary.uploader.upload(file_path, public_id=public_id, folder="Wanderlust_DEV")
        return response["url"]

# MongoDB credentials
class MongoDB:
    def __init__(self):
        self.client = MongoClient("mongodb+srv://alhanmsiddique:jU0FQ5M89o3hlL1w@cluster0.o7nko.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        self.db = self.client["test"]
        self.collection = self.db["listings"]

    def get_all_suspects(self):
        return list(self.collection.find())

    def add_suspect(self, suspect_details):
        try:
            self.collection.insert_one(suspect_details)
        except Exception as e:
            raise Exception(f"Error adding suspect to MongoDB: {str(e)}")

# Global variables
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names

    print("Fetching known faces from Cloudinary...")

    try:
        response = cloudinary.api.resources(
            type="upload",
            prefix="Wanderlust_DEV",
            resource_type="image"
        )

        if not os.path.exists("faces_folder"):
            os.makedirs("faces_folder")

        for resource in response.get("resources", []):
            try:
                image_url = resource.get("url")
                public_id = resource.get("public_id")

                if image_url:
                    print(f"Fetching image: {public_id} from {image_url}...")

                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        image_path = os.path.join("faces_folder", f"{public_id}.jpg")
                        with open(image_path, "wb") as image_file:
                            for chunk in response.iter_content(1024):
                                image_file.write(chunk)

                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            encoding = encodings[0]
                            known_face_encodings.append(encoding)
                            known_face_names.append(public_id)
                            print(f"Loaded face for {public_id} from Cloudinary")
                        else:
                            print(f"No face found in image {public_id}. Skipping.")
                    else:
                        print(f"Failed to fetch image for {public_id}. HTTP status code: {response.status_code}")
            except Exception as e:
                print(f"Error processing image for {resource.get('public_id', 'unknown')}: {e}")
    except Exception as e:
        print(f"Error fetching images from Cloudinary: {e}")

# Function to play alert sound
def play_alert_sound():
    playsound("alert.mp3")

# Process a single frame for face recognition and object detection
def process_frame(frame, model):
    global known_face_encodings, known_face_names

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[best_match_index]
            print(f"Recognized: {name}")

            # Log the recognition
            log_to_csv("recognized_faces.csv", [name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            play_alert_sound()

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    results = model.predict(frame, conf=0.5)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            class_name = result.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

# Log recognized faces and objects to CSV
def log_to_csv(filename, data):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

# GUI setup
def create_gui():
    root = tk.Tk()
    root.title("Surveillance System")
    root.geometry("800x600")

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=tk.BOTH)

    tab1 = tk.Frame(notebook)
    notebook.add(tab1, text="Surveillance")

    tab1_buttons = tk.Frame(tab1)
    tab1_buttons.pack(side=tk.BOTTOM, pady=20)

    tk.Button(tab1_buttons, text="Start", command=lambda: start_surveillance_threaded()).pack(side=tk.LEFT, padx=10)

    root.mainloop()

def start_surveillance_threaded():
    surveillance_thread = Thread(target=start_surveillance)
    surveillance_thread.start()

def start_surveillance():
    video_capture = cv2.VideoCapture(0)
    model = YOLO('yolov5n.pt')

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        process_frame(frame, model)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_known_faces()
    create_gui()
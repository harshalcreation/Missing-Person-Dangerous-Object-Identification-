import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog
import threading
from playsound import playsound
import speech_recognition as sr
import os
import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import csv
from datetime import datetime
from PIL import Image, ImageTk

# Global variables
known_face_encodings = []
known_face_names = []
dangerous_objects = ["knife", "gun", "bomb"]
stop_processing_flag = False
video_processing_running = False
recognized_faces = {}
activity_log = []

# Load known faces
def load_known_faces():
    global known_face_encodings, known_face_names
    print("Loading known faces...")
    face_folder = "faces_folder"
    for filename in os.listdir(face_folder):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(face_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(filename.split(".")[0])
                print(f"Loaded {filename.split('.')[0]} from {image_path}")
            else:
                print(f"No face found in {image_path}. Skipping.")

# Function to play alert sound
def play_alert_sound():
    alert_path = "alert.mp3"  # Path to the alert sound file
    playsound(alert_path)

# Log recognized faces and objects to CSV
def log_to_csv(filename, data):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Process a single frame for both face recognition and object detection
def process_frame(frame, model, class_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Recognition
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[best_match_index]
            log_to_csv('recognized_faces.csv', [name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            play_alert_sound()  # Alert sound for recognized face

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Object Detection
    results = model.predict(frame, conf=0.5)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown ({class_id})"

            label = f"{class_name} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if class_name in dangerous_objects:
                cv2.putText(frame, "DANGER!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                play_alert_sound()
                log_to_csv('recognized_objects.csv', [class_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

    return frame

# Start surveillance
def start_surveillance():
    global video_processing_running
    if video_processing_running:
        print("Surveillance already running!")
        return

    video_processing_running = True
    print("Starting surveillance...")

    def surveillance_thread():
        global stop_processing_flag
        stop_processing_flag = False

        video_capture = cv2.VideoCapture(0)
        model = YOLO('yolov5n.pt')
        class_names = model.names

        while not stop_processing_flag:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame.")
                break

            processed_frame = process_frame(frame, model, class_names)
            update_feed(processed_frame)

        video_capture.release()
        video_processing_running = False
        print("Surveillance stopped.")

    threading.Thread(target=surveillance_thread, daemon=True).start()

# Stop surveillance
def stop_processing():
    global stop_processing_flag
    stop_processing_flag = True
    print("Stopping surveillance...")

# Update the GUI feed
def update_feed(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
    feed_label.config(image=frame_image)
    feed_label.image = frame_image

# Add face to system
def add_face():
    name = simpledialog.askstring("Input", "Enter the name of the person:")
    if name:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg")])
        if file_path:
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                save_path = os.path.join("faces_folder", f"{name}.jpg")
                cv2.imwrite(save_path, cv2.imread(file_path))
                print(f"Added face for {name}!")
                log_to_csv('activity_log.csv', [f"Added face: {name}", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

# Create GUI
def create_gui():
    global feed_label
    root = tk.Tk()
    root.title("Surveillance System")
    root.geometry("800x600")

    # Header with Blinkers logo
    header = tk.Frame(root, bg="white", height=100)
    header.pack(fill=tk.X, side=tk.TOP)

    logo_path = "blinkers_logo.png"
    logo_image = Image.open(logo_path).resize((80, 80), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(header, image=logo_photo, bg="white")
    logo_label.image = logo_photo
    logo_label.pack(side=tk.LEFT, padx=10, pady=10)

    header_title = tk.Label(header, text="Surveillance System", font=("Arial", 24, "bold"), bg="white")
    header_title.pack(side=tk.LEFT, padx=20)

    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=tk.BOTH)

    # Tab 1: Real-Time Surveillance
    tab1 = tk.Frame(notebook)
    notebook.add(tab1, text="Real-Time Surveillance")

    feed_label = tk.Label(tab1, bg="black")
    feed_label.pack(expand=True, fill=tk.BOTH)

    tab1_buttons_frame = tk.Frame(tab1)
    tab1_buttons_frame.pack(side=tk.BOTTOM, pady=20)

    tk.Button(tab1_buttons_frame, text="Start Surveillance", command=start_surveillance).pack(side=tk.LEFT, padx=10)
    tk.Button(tab1_buttons_frame, text="Stop Surveillance", command=stop_processing).pack(side=tk.LEFT, padx=10)
    tk.Button(tab1_buttons_frame, text="Add Face", command=add_face).pack(side=tk.LEFT, padx=10)

    # Tab 2: Recorded Video Processing
    tab2 = tk.Frame(notebook)
    notebook.add(tab2, text="Recorded Video Processing")
    tk.Label(tab2, text="Feature Coming Soon!", font=("Arial", 16)).pack(expand=True)

    # Tab 3: View Activity Log
    tab3 = tk.Frame(notebook)
    notebook.add(tab3, text="View Activity Log")

    activity_log_text = tk.Text(tab3, state=tk.DISABLED, wrap=tk.WORD)
    activity_log_text.pack(expand=True, fill=tk.BOTH)

    root.mainloop()

# Start the program
if __name__ == "__main__":
    load_known_faces()
    create_gui()
